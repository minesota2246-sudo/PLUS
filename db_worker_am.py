import os
import json
import time
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
from openai import OpenAI
from supabase import create_client, Client
from datetime import datetime

# [설정] 환경 변수 (GitHub Secrets 등에 설정된 값)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 1. 심층 데이터 수집 (지표 + 뉴스)
# ==========================================
def get_analysis_data(ticker):
    try:
        # A. 주가 데이터 수집 (최근 60일)
        df = yf.download(ticker, period="60d", interval="1d", progress=False)
        if df.empty or len(df) < 20: return None

        close = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close']
        vol = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume']

        # 지표 계산 (T-Shield용)
        ma20 = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        upper_bb = ma20 + (std * 2)
        lower_bb = ma20 - (std * 2)
        delta = close.diff()
        rsi = 100 - (100 / (1 + (delta.where(delta > 0, 0).rolling(14).mean() / -delta.where(delta < 0, 0).rolling(14).mean())))
        vol_ratio = round(float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]), 2)

        # B. 뉴스 헤드라인 수집 (T-Bull용)
        news_headlines = []
        try:
            ticker_obj = yf.Ticker(ticker)
            news_headlines = [n['title'] for n in ticker_obj.news[:3]]
        except: news_headlines = ["최근 뉴스 없음"]

        return {
            "current_price": round(float(close.iloc[-1]), 2),
            "ma20": round(float(ma20.iloc[-1]), 2),
            "bb_upper": round(float(upper_bb.iloc[-1]), 2),
            "bb_lower": round(float(lower_bb.iloc[-1]), 2),
            "rsi": round(float(rsi.iloc[-1]), 2),
            "vol_ratio": vol_ratio,
            "news": news_headlines
        }
    except Exception as e:
        print(f"  ⚠ {ticker} 데이터 수집 실패: {e}")
        return None

# ==========================================
# 2. AI 3인방 분석 엔진 (역할 철저 분리)
# ==========================================
def run_plus_t_ai(ticker, name, data):
    news_str = "\n".join([f"- {h}" for h in data['news']])
    prompt = f"""
    Plus-T AI 분석 팀 가동. [{name}({ticker})] 분석.
    
    [데이터 시트]
    1. 지표: 현재가 {data['current_price']}, 20일선 {data['ma20']}, RSI {data['rsi']}, 거래량강도 {data['vol_ratio']}
    2. 뉴스: {news_str}

    [역할 분담 지시]
    - 🛡️ T-Shield (차트 전문가): '지표'만 보고 분석하세요. 현재 위치가 기술적으로 저평가인지 과열인지만 판단합니다.
    - 🐂 T-Bull (뉴스/심리 전문가): '지표'는 무시하고 '뉴스'만 보세요. 시장의 반응과 기대감, 호재/악재의 크기만 분석합니다.
    - 🧠 T-Core (종합 컨트롤러): Shield의 지표와 Bull의 심리를 저울질하여 최종 승률(win_rate)과 투자 한줄평을 작성하세요.

    반드시 JSON 형식으로만 응답:
    {{
        "t_index_score": (0-100), "win_rate": (0-100),
        "t_shield_opinion": "...", "t_bull_opinion": "...", "t_core_opinion": "...",
        "final_verdict": "...", "target_price": (숫자), "support_price": (숫자)
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "너는 주식 분석 전문가 3인방을 1인 3역으로 연기한다."},
                      {"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except: return None

# ==========================================
# 3. 150개 전략 종목 리스트
# ==========================================
def get_150_tickers():
    # 미국 75개 (빅테크 + 반도체 + 코인관련 + 고배당 + ETF)
    us_75 = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "COST", "NFLX",
        "AMD", "TSM", "ARM", "MU", "INTC", "ASML", "LRCX", "AMAT", "QCOM", "TXN",
        "COIN", "MSTR", "MARA", "RIOT", "HOOD", "PYPL", "SQ", "SOFI", "PLTR", "SNOW",
        "JPM", "BAC", "GS", "V", "MA", "LLY", "UNH", "PFE", "ABBV", "JNJ",
        "WMT", "KO", "PEP", "NKE", "SBUX", "XOM", "CVX", "NEE", "CAT", "GE",
        "QQQ", "SPY", "SOXL", "TQQQ", "SCHD", "JEPI", "TSLY", "BITO", "ARKK", "TLT",
        "ABNB", "BKNG", "UBER", "DASH", "PANW", "CRWD", "ZS", "FTNT", "OKTA", "U",
        "LULU", "MELI", "SE", "PDD", "MCD"
    ]
    
    # 한국 75개 (시총 상위 순)
    kr_75 = []
    try:
        df_krx = fdr.StockListing('KRX').head(75)
        for _, row in df_krx.iterrows():
            code = row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ")
            kr_75.append({"ticker": code, "name": row['Name'], "market": "KR"})
    except: pass
    
    final = [{"ticker": t, "name": t, "market": "US"} for t in us_75]
    final.extend(kr_75)
    return final

# ==========================================
# 4. 메인 실행 루프
# ==========================================
def main():
    targets = get_150_tickers()
    print(f"🚀 Plus-T AI 엔진 작동 시작 (총 {len(targets)} 종목 누적 분석)")
    
    for i, item in enumerate(targets):
        print(f"[{i+1}/150] {item['name']} 분석 중...")
        data = get_analysis_data(item['ticker'])
        if not data: continue
        
        analysis = run_plus_t_ai(item['ticker'], item['name'], data)
        if not analysis: continue
        
        try:
            supabase.table("tb_t_index").insert({
                "ticker": item['ticker'], "stock_name": item['name'],
                "market_type": item['market'], "is_popular": True if i < 20 else False,
                "current_price": data['current_price'], "ma20": data['ma20'],
                "rsi": data['rsi'], "vol_ratio": data['vol_ratio'],
                **analysis
            }).execute()
            print(f"  ✅ 기록 완료")
        except Exception as e: print(f"  ❌ 에러: {e}")
        
        time.sleep(1) # API 부하 방지

if __name__ == "__main__":
    main()