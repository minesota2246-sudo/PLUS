import os
import json
import time
import pandas as pd
import yfinance as yf
import FinanceDataReader as fdr
from openai import OpenAI
from supabase import create_client, Client
from datetime import datetime

# ==========================================
# [설정] 환경 변수 세팅
# ==========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================
# 1. 심층 데이터 수집 (입체적 지표 + 뉴스)
# ==========================================
def get_analysis_data(ticker):
    try:
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
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Division by zero 방지
        rs = gain / loss.replace(0, 0.001)
        rsi = 100 - (100 / (1 + rs))
        
        vol_avg = vol.rolling(window=20).mean()
        vol_ratio = round(float(vol.iloc[-1] / vol_avg.iloc[-1]), 2) if vol_avg.iloc[-1] > 0 else 1.0

        # 뉴스 헤드라인 수집 (T-Bull용)
        news_headlines = []
        try:
            ticker_obj = yf.Ticker(ticker)
            news_data = ticker_obj.news[:3]
            news_headlines = [n['title'] for n in news_data] if news_data else ["최근 주요 뉴스 없음"]
        except: news_headlines = ["최근 주요 뉴스 없음"]

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
# 2. AI 3인방 분석 엔진 (역할 철저 분리 & 커뮤니티 투심 적용)
# ==========================================
def run_plus_t_ai(ticker, name, data):
    news_str = "\n".join([f"- {h}" for h in data['news']])
    prompt = f"""
    당신은 Plus-T의 수석 주식 분석 팀입니다. [{name}({ticker})]에 대해 분석하세요.
    
    [데이터 시트]
    1. 가격/차트: 현재가 {data['current_price']}, 20일선 {data['ma20']}, 상단밴드 {data['bb_upper']}, 하단밴드 {data['bb_lower']}, RSI {data['rsi']}, 거래량강도 {data['vol_ratio']}배
    2. 실시간 뉴스: {news_str}

    [AI 비서별 역할 및 절대 금기사항]
    
    1. 🛡️ T-Shield (기술 지표 통합 분석가):
       - 임무: 차트를 '입체적'으로 분석하십시오. RSI, 20일선과의 이격도, 볼린저 밴드 내의 위치, 예상되는 '지지선'을 하나의 스토리로 엮어서 설명하세요. (예: "RSI가 침체권이고 볼린저 하단을 터치했으나, 20일선과 이격이 커 단기 기술적 반등이 기대됨. 지지선은 X원.")
       - 금기: 뉴스 언급 절대 금지.
       
    2. 🐂 T-Bull (뉴스 및 커뮤니티 심리 전문가):
       - 임무: 제공된 '실시간 뉴스'와 '거래량 강도({data['vol_ratio']}배)'를 바탕으로, 현재 종토방이나 레딧 같은 온라인 투자 커뮤니티의 대중 심리와 반응을 찰지게 추론하고 묘사하세요. (예: "이 뉴스 한 방에 커뮤니티가 폭발했습니다! 거래량이 평소보다 {data['vol_ratio']}배 터진 걸 보니 개미들의 FOMO가 시작됐습니다.")
       - 금기: 'RSI', '이평선', '밴드' 등 차트 숫자 언급 절대 금지.

    3. 🧠 T-Core (전략 통합 컨트롤러):
       - 임무: Shield의 차트 로직과 Bull의 커뮤니티 투심을 저울질하여 종합적인 상승 확률(win_rate)을 냉정하게 산출하고 결론을 내립니다.

    🚨 [경고] 당신은 금융 분석가입니다. '슬라이더', '앱 UI', '광고 기획', '서비스 운영' 등 투자와 무관한 헛소리를 할 경우 즉시 시스템에서 삭제됩니다. 오직 '주가 방향'과 '투자 전략'만 논하십시오.

    반드시 JSON 형식으로만 응답:
    {{
        "t_index_score": (0-100), 
        "win_rate": (0-100),
        "t_shield_opinion": "🛡️ Shield: (지표+차트 입체적 분석)",
        "t_bull_opinion": "🐂 Bull: (뉴스+커뮤니티 반응 분석)",
        "t_core_opinion": "🧠 Core: (종합 결론)",
        "final_verdict": "투자자를 위한 직관적 한 줄 요약",
        "target_price": (숫자만), 
        "support_price": (숫자만)
    }}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "너는 냉철한 주식 분석가다. 앱 개발자나 마케터처럼 굴지 마라."},
                      {"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            temperature=0.4 # 창의성을 조절하여 헛소리 차단 및 찰진 커뮤니티 반응 유도
        )
        return json.loads(response.choices[0].message.content)
    except: return None

# ==========================================
# 3. 트렌드 자동 반영 150개 종목 추출
# ==========================================
def get_dynamic_tickers():
    final_list = []
    
    # --- [1] 미국 (우량주 50 + 필수 트렌드 25) ---
    try:
        sp500 = fdr.StockListing('S&P500').head(50)
        for _, row in sp500.iterrows():
            final_list.append({"ticker": row['Symbol'], "name": row['Name'], "market": "US"})
    except: pass
    
    trend_us = [
        "TSLA", "ARM", "PLTR", "COIN", "MSTR", "MARA", "RIOT", "HOOD", "SOFI", "U",
        "QQQ", "SPY", "SOXL", "TQQQ", "SCHD", "JEPI", "TSLY", "BITO", "TLT", "ARKK",
        "CRWD", "PANW", "SNOW", "PDD", "MELI"
    ]
    existing_us_tickers = [item['ticker'] for item in final_list]
    for t in trend_us:
        if t not in existing_us_tickers:
            final_list.append({"ticker": t, "name": t, "market": "US"})
            
    final_list = final_list[:75] # 딱 75개 맞춤

    # --- [2] 한국 (우량주 50 + 거래량 폭발 트렌드주 25) ---
    kr_list = []
    try:
        df_krx = fdr.StockListing('KRX')
        
        # 1. 시가총액 상위 50개
        top_50_cap = df_krx.head(50)
        for _, row in top_50_cap.iterrows():
            code = row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ")
            kr_list.append({"ticker": code, "name": row['Name'], "market": "KR"})
            
        # 2. 오늘 거래량 폭발 상위 25개 (테마/트렌드)
        top_volume = df_krx.sort_values(by='Volume', ascending=False).head(50)
        existing_kr_codes = [item['name'] for item in kr_list]
        
        added_vol = 0
        for _, row in top_volume.iterrows():
            if added_vol >= 25: break
            if row['Name'] not in existing_kr_codes:
                code = row['Code'] + (".KS" if row['Market'] == 'KOSPI' else ".KQ")
                kr_list.append({"ticker": code, "name": row['Name'], "market": "KR"})
                added_vol += 1
    except Exception as e: print(f"KR 데이터 수집 에러: {e}")
    
    final_list.extend(kr_list[:75])
    return final_list

# ==========================================
# 4. 메인 실행 루프
# ==========================================
def main():
    targets = get_dynamic_tickers()
    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"🚀 Plus-T AI 엔진 가동 ({today_str}): 총 {len(targets)} 종목 입체/심리 분석 시작!\n")
    
    for i, item in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] {item['name']} 분석 중...")
        data = get_analysis_data(item['ticker'])
        if not data: continue
        
        analysis = run_plus_t_ai(item['ticker'], item['name'], data)
        if not analysis: continue
        
        try:
            supabase.table("tb_t_index").insert({
                "ticker": item['ticker'], "stock_name": item['name'],
                "market_type": item['market'], 
                "is_popular": True if i < 10 else False, # 대시보드 메인 노출용
                "current_price": data['current_price'], "ma20": data['ma20'],
                "rsi": data['rsi'], "vol_ratio": data['vol_ratio'],
                **analysis
            }).execute()
            print(f"  ✅ [승률: {analysis.get('win_rate', 0)}%] 분석 DB 기록 완료")
        except Exception as e: print(f"  ❌ DB 저장 에러: {e}")
        
        time.sleep(1.2) # API 부하 방지용 쿨타임

if __name__ == "__main__":
    main()