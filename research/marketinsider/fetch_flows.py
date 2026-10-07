"""네이버 증권 모바일 API에서 종목별 일별 투자자 순매수량·외국인 보유율을 내려받는다 (2016~)."""
import os, sys, json, time, pickle, requests
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ep_common as C
CACHE = C.CACHE; OUT = os.path.join(CACHE, "flows.pkl")
S = requests.Session(); S.headers["User-Agent"] = "Mozilla/5.0"
def num(x):
    x = str(x).replace(",", "").replace("%", "").replace("+", "")
    try: return float(x)
    except: return None
def fetch(code):
    rows = []; bz = None
    for _ in range(80):
        u = f"https://m.stock.naver.com/api/stock/{code}/trend?pageSize=60" + (f"&bizdate={bz}" if bz else "")
        for k in range(4):
            try:
                r = S.get(u, timeout=20)
                if r.status_code == 200: d = r.json(); break
            except Exception: pass
            time.sleep(1 + k)
        else: return code, rows, "fail"
        if not d: break
        rows += d; bz = d[-1]["bizdate"]
        if bz < "20160101": break
        time.sleep(0.15)
    return code, rows, "ok"
if __name__ == "__main__":
    data, _ = C.load_all()
    codes = sorted({t.split(".")[0] for t in data})
    res = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    todo = [c for c in codes if c not in res]
    print(len(codes), "종목, 남은", len(todo), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(5) as ex:
        for i, (c, rows, st) in enumerate(ex.map(fetch, todo), 1):
            if st == "ok": res[c] = rows
            if i % 10 == 0:
                pickle.dump(res, open(OUT, "wb")); print(i, c, len(rows), st, round(time.time() - t0), "s", flush=True)
    pickle.dump(res, open(OUT, "wb")); print("done", len(res), flush=True)
