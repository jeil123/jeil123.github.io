"""텔레그램 알림 + 브로커 인터페이스. 기본은 모의(Paper)이며 실주문은 명시적 활성화가 필요하다."""
import json
import os
import time
from pathlib import Path


def telegram(text: str) -> bool:
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        print("[telegram 미설정] " + text)
        return False
    import requests
    r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=10)
    return r.ok


class PaperBroker:
    def __init__(self, log="paper_orders.jsonl"):
        self.log = Path(log)

    def order(self, code, side, qty, note=""):
        rec = dict(ts=time.strftime("%F %T"), code=code, side=side, qty=qty, note=note)
        with self.log.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


class KISBroker:
    """한국투자증권 Open API 연결용 자리. 의도적으로 미구현:
    실계좌 자동주문은 앱키/계좌 설정, 모의투자 검증, 주문 한도 확인 후 직접 켜야 한다."""
    def __init__(self, *a, **k):
        if os.getenv("NNFX_LIVE") != "1":
            raise RuntimeError("실거래는 NNFX_LIVE=1 로 명시적으로 켜야 합니다.")
        raise NotImplementedError("KIS 주문 어댑터는 아직 구현하지 않았습니다.")


def dispatch(code, name, state, broker=None):
    broker = broker or PaperBroker()
    o = state.get("next_open_order") or {}
    if not o:
        telegram(f"[{name} {code}] 신규 주문 없음 ({state['last_bar']})")
        return
    msg = f"[{name} {code}] 다음 거래일 시가 {o['action']}"
    if o["action"] == "BUY":
        msg += f" / 손절 {o['stop']:,.0f}원 / ATR {o['atr']:,.0f}"
    telegram(msg)
    broker.order(code, o["action"], 0, note=json.dumps(o))
