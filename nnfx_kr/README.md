# nnfx_kr — NNFX 프레임워크 국내주식 버전 (롱온리, 일봉)

구성: Baseline(EMA/HMA) + C1(RSI) + C2(MACD 히스토그램) + Volume + Exit(Supertrend) + ATR 리스크 관리
(손절 1.5ATR, 1ATR에서 절반 익절 후 본전 스톱, 거래당 위험 2%).

## 사용
    pip install pandas numpy requests pykrx
    python -m nnfx_kr.cli backtest    --code 005930 --start 20150101
    python -m nnfx_kr.cli optimize    --code 005930      # 앞 70%에서 최적화, 뒤 30%로 검증
    python -m nnfx_kr.cli walkforward --code 005930
    python -m nnfx_kr.cli forward     --code 005930 --name 삼성전자 --forward-start 2026-07-01
    # --csv 파일.csv (date,open,high,low,close,volume) 도 가능, 옵션 없으면 합성 데이터

## 국내주식 반영
KRX 호가단위(2023 통합), 매수/매도 수수료, 매도세(`krx.Costs`, 세율은 현행 확인 후 조정), 슬리피지 1틱,
공매도 제한으로 롱온리, 신호는 종가 확정 후 다음 거래일 시가 체결.

## 알림 / 자동매매
- 텔레그램: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` 환경변수. 없으면 콘솔 출력.
- 기본 브로커는 `PaperBroker`(주문을 `paper_orders.jsonl`에 기록만 함).
- `KISBroker`(실주문)는 의도적으로 미구현. 모의투자에서 충분히 검증한 뒤 직접 연결할 것.

## 주의
합성 데이터 결과는 의미 없음. 최적화 결과는 반드시 검증구간/포워드 성과로만 판단하고, 과최적화에 유의.
투자 판단과 손실의 책임은 사용자에게 있습니다.
