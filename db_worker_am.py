import os
import json
import time
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
import ccxt
from openai import OpenAI
from supabase import create_client, Client
from datetime import datetime

# ==========================================
# [설정] 깃허브 Secrets에서 키 가져오기
# ==========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 1. 기술적 지표 추출
# ==========================================
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period="45d", interval="1d", progress=False)
        if df.empty or len(df) < 20: return None

        if isinstance(df.columns, pd.MultiIndex):
            close_series = df['Close'][ticker]
            vol_series = df['Volume'][ticker]
        else:
            close_series = df['Close']
            vol_series = df['Volume']

        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        vol_avg = vol_series.head(30).mean()
        vol_now = vol_series.tail(5).mean()
        vol_ratio = round(float(vol_now / vol_avg), 2)

        return {
            "current_price": round(float(close_series.iloc[-1]), 2),
            "rsi": round(float(rsi.iloc[-1]), 2),
            "vol_ratio": vol_ratio,
            "price_history": [round(float(p), 2) for p in close_series.tail(15).tolist()]
        }
    except Exception as e:
        print(f"  ⚠ 데이터 수집 오류 ({ticker}): {e}")
        return None

# ==========================================
# 2. AI 비서 분석 엔진
# ==========================================
def run_ai_analysis(ticker, name, data):
    prompt = f"""
    당신은 Plus-T의 핵심 AI 분석 팀입니다. [{name}({ticker})]를 분석하십시오.
    현재가: {data['current_price']}, RSI: {data['rsi']}, 거래량강도: {data['vol_ratio']}
    최근 가격 흐름: {data['price_history']}

    다음 3명의 AI 비서 의견을 포함하여 반드시 JSON 형식으로만 답변하십시오:
    1. T-Shield (방어형): 리스크 관리 및 하락 위험성 경고
    2. T-Bull (공격형): 상승 모멘텀, 매수 타점, 수급 분석
    3. T-Core (통계형): 'The Proof' 모델 기반의 과거 유사 패턴 승률 분석

    응답 JSON 키값은 아래와 정확히 일치해야 합니다:
    {{
        "t_index_score": (0-100 사이의 정수 점수),
        "win_rate": (상승 확률을 나타내는 0-100 사이의 소수점 숫자),
        "t_shield_opinion": "T-Shield의 분석 코멘트",
        "t_bull_opinion": "T-Bull의 분석 코멘트",
        "t_core_opinion": "T-Core의 분석 코멘트",
        "final_verdict": "투자 입문자를 위한 직관적이고 쉬운 한 줄 요약",
        "target_price": (목표가, 숫자만 입력)
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 주식 입문자를 돕는 친절하고 전문적인 금융 AI입니다. 복잡한 용어는 배제하고 토스(Toss) 앱처럼 세련되고 깔끔한 톤을 유지하세요."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  ⚠ AI 분석 실패 ({name}): {e}")
        return None

# ==========================================
# 3. 150개 타겟 종목 리스트 구성
# ==========================================
def get_ticker_list():
    tickers = []
    
    us_list = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD", "NFLX", "COIN",
               "PLTR", "SOFI", "TSM", "AVGO", "ORCL", "V", "MA", "UNH", "JNJ", "WMT",
               "PG", "XOM", "CVX", "LLY", "ABBV", "COST", "HD", "MRK", "BAC", "KO",
               "PEP", "TMO", "ADBE", "LIN", "NKE", "MCD", "CSCO", "CRM", "PFE", "ABT",
               "DHR", "ACN", "CMCSA", "DIS", "T", "VZ", "INTC", "QCOM", "TXN", "AMAT",
               "MU", "SBUX", "INTU", "ISRG", "LRCX", "AMGN", "MDLZ", "BKNG", "GILD", "PYPL"]
    for t in us_list: tickers.append({"ticker": t, "name": t, "market": "US"})

    try:
        krx = fdr.StockListing('KRX').head(60)
        for _, row in krx.iterrows():
            code = row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ")
            tickers.append({"ticker": code, "name": row['Name'], "market": "KR"})
    except: pass

    try:
        bn = ccxt.binance()
        crypto = sorted(bn.fetch_tickers().items(), key=lambda x: x[1].get('quoteVolume', 0), reverse=True)
        count = 0
        for symbol, _ in crypto:
            if '/USDT' in symbol and count < 30:
                tickers.append({"ticker": symbol.replace('/', '-'), "name": symbol.split('/')[0], "market": "CRYPTO"})
                count += 1
    except: pass
    
    return tickers

# ==========================================
# 4. 메인 실행 프로세스 (Insert 누적형 적용)
# ==========================================
def start_worker():
    all_targets = get_ticker_list()
    # 날짜 출력 (예: 2026-05-01)
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"🚀 Plus-T AI 엔진 가동 ({today_str}): 총 {len(all_targets)}개 종목 분석 및 DB 적재 시작!\n")

    for i, item in enumerate(all_targets):
        print(f"[{i+1}/{len(all_targets)}] {item['name']} 데이터 분석 중...")
        
        data = get_stock_data(item['ticker'])
        if not data: continue
        
        analysis = run_ai_analysis(item['ticker'], item['name'], data)
        if not analysis: continue
        
        is_popular_flag = True if i < 10 else False 

        try:
            # 💡 핵심 수정: upsert 대신 insert를 사용하여 데이터를 '누적'시킵니다.
            supabase.table("tb_t_index").insert({
                "ticker": item['ticker'],
                "stock_name": item['name'],
                "market_type": item['market'],
                "is_popular": is_popular_flag,
                **analysis, 
                # created_at은 SQL에서 자동으로 오늘 날짜(예: 2026-05-01)를 찍어줍니다.
            }).execute()
            
            print(f"  ✅ 기록 완료 (T-Index: {analysis['t_index_score']}점)")
        
        except Exception as e:
            # SQL에서 중복 에러가 났을 때 처리 (이미 오늘 날짜에 저장된 경우)
            if "duplicate key value violates unique constraint" in str(e):
                print(f"  ℹ️ {item['name']}은(는) 이미 오늘 기록이 있습니다. 건너뜁니다.")
            else:
                print(f"  ❌ DB 저장 실패 ({item['name']}): {e}")
            
        time.sleep(1.5) 

    print("\n🎉 오늘의 Plus-T 분석 기록이 모두 안전하게 저장되었습니다!")

if __name__ == "__main__":
    start_worker()