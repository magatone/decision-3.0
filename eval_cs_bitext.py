"""CS 문의 분류를 Jev 또는 로컬 d3로 돌리고 정답과 비교한다.

Jev와 d3 서버가 같은 /v1/systemone 형식이라 주소와 모델 이름만 바꿔서 같은 평가를 돌린다.

  python eval_cs_bitext.py sample <bitext|banking77>        # 평가셋 만들기 (한 번만)
  set JEV_API_KEY=...                                        # 키는 환경변수로만 받는다
  python eval_cs_bitext.py run <bitext|banking77> <jev|d3> [건수]
  python eval_cs_bitext.py report <bitext|banking77> <jev|d3>
  python eval_cs_bitext.py selftest
"""
import collections, concurrent.futures as cf, csv, json, os, random, sys, time, urllib.error, urllib.request
from pathlib import Path

BACKENDS = {
    "jev": {"url": "https://api.typesafe.ai/v1/systemone", "model": "jev-latest", "key_env": "JEV_API_KEY", "workers": 4},
    "d3": {"url": "http://127.0.0.1:8899/v1/systemone", "model": "d3-mini", "key_env": None, "workers": 4},
    # WSL에서 laya-serve를 8898로 띄운다. 기본 보기 영역(192토큰)에선 Banking77의 77개 보기가 뭉개져서
    # 1024로 늘린다. 학습도 같은 값으로 해야 전후 비교가 공정하다.
    "laya-base": {"url": "http://127.0.0.1:8898/v1/systemone", "model": "laya", "key_env": None, "workers": 4,
                  "extra": {"max_len": 1024, "head_max_len": 1024}},
    "laya-ft": {"url": "http://127.0.0.1:8898/v1/systemone", "model": "laya-ft", "key_env": None, "workers": 4,
                "extra": {"max_len": 1024, "head_max_len": 1024}},
    # 한국어용 laya-multilingual(322M). 댓글이 짧고 보기가 예/아니오라 기본 길이 설정 그대로 쓴다(학습도 동일하게).
    "laya-ml-base": {"url": "http://127.0.0.1:8898/v1/systemone", "model": "multilingual", "key_env": None, "workers": 4},
    "laya-ml-ft": {"url": "http://127.0.0.1:8898/v1/systemone", "model": "laya-ml-ft", "key_env": None, "workers": 4},
}

# Bitext: 두 모델에 똑같은 설명을 준다. 이름만 주면 USER·SALES 같은 묶음은 뜻이 모호해서.
BITEXT_CRITERIA = {
    "ACCOUNT": "Opening, closing or changing an account, order history, recovering a password",
    "APP_WEBSITE": "Technical problems with the app or website, or how to use the app",
    "CART": "Adding or removing products in the shopping cart",
    "CONTACT": "Wants to reach customer service or talk to a human agent",
    "DELIVERY": "Shipping costs, delivery time or tracking, damaged, missing or wrong items in a delivery",
    "FEEDBACK": "Submitting feedback, a product review or a product idea",
    "ORDER": "Cancelling, changing or tracking an order, requesting an invoice",
    "PAYMENT": "How to pay, payment methods, problems making a payment",
    "PRODUCT": "Product information, availability, exchanging a product, a defective product",
    "RETURNS": "Returning a product, refunds, return or refund policy",
    "SALES": "Sales periods and discounts",
    "STORE": "Store locations and opening hours",
    "USER": "Correcting inaccurate personal data (right to rectification)",
}

# Banking77: 77개 설명을 손으로 쓰면 내 해석이 정답 힌트로 들어가서, 라벨 이름을 띄어쓰기로만 바꿔 준다.
BANKING77_LABELS = sorted({r["category"] for r in csv.DictReader(open("data/datasets/banking77-test.csv", encoding="utf-8"))}) \
    if Path("data/datasets/banking77-test.csv").exists() else []
# 처리 경로가 비슷한 것끼리 묶은 8개 그룹. 77개는 틀려도 그룹은 맞는지 따로 보려고.
BANKING77_GROUPS = {
    "card_order": "activate_my_card card_about_to_expire card_arrival card_delivery_estimate card_linking get_disposable_virtual_card get_physical_card getting_spare_card getting_virtual_card order_physical_card visa_or_mastercard",
    "card_problem": "apple_pay_or_google_pay atm_support card_acceptance card_not_working card_swallowed change_pin compromised_card contactless_not_working disposable_card_limits lost_or_stolen_card lost_or_stolen_phone passcode_forgotten pin_blocked virtual_card_not_working",
    "payment_issue": "card_payment_not_recognised cash_withdrawal_not_recognised declined_card_payment declined_cash_withdrawal direct_debit_payment_not_recognised pending_card_payment pending_cash_withdrawal reverted_card_payment? transaction_charged_twice wrong_amount_of_cash_received",
    "fee_fx": "card_payment_fee_charged card_payment_wrong_exchange_rate cash_withdrawal_charge country_support exchange_charge exchange_rate exchange_via_app extra_charge_on_statement fiat_currency_support supported_cards_and_currencies top_up_by_bank_transfer_charge top_up_by_card_charge transfer_fee_charged wrong_exchange_rate_for_cash_withdrawal",
    "transfer": "balance_not_updated_after_bank_transfer beneficiary_not_allowed cancel_transfer declined_transfer failed_transfer pending_transfer receiving_money transfer_into_account transfer_not_received_by_recipient transfer_timing",
    "top_up": "automatic_top_up balance_not_updated_after_cheque_or_cash_deposit pending_top_up top_up_by_cash_or_cheque top_up_failed top_up_limits top_up_reverted topping_up_by_card verify_top_up",
    "identity_account": "age_limit edit_personal_details terminate_account unable_to_verify_identity verify_my_identity verify_source_of_funds why_verify_identity",
    "refund": "Refund_not_showing_up request_refund",
}
GROUP_OF = {lab: g for g, labs in BANKING77_GROUPS.items() for lab in labs.split()}

# K-MHaS(한국어 뉴스 댓글 혐오 표현, 한 댓글에 라벨 여러 개). 카테고리마다 예/아니오 질문 하나씩, 한 요청에 8개를 묶는다.
KMHAS_NAMES = ["origin", "physical", "politics", "profanity", "age", "gender", "race", "religion", "not_hate_speech"]
# 문장을 통째로 적는다. "{desc}이 있습니까"처럼 조사를 붙여 만들면 "비하이"처럼 틀린 문장이 나온다.
KMHAS_QUESTIONS = {c: {"type": "noul", "instructions": q} for c, q in {
    "origin": "이 댓글에 출신 지역이나 국적을 이유로 한 차별이나 비하가 있습니까?",
    "physical": "이 댓글에 외모나 신체를 비하하는 내용이 있습니까?",
    "politics": "이 댓글에 정치 성향을 이유로 한 비하나 혐오가 있습니까?",
    "profanity": "이 댓글에 욕설이나 저속한 표현이 있습니까?",
    "age": "이 댓글에 나이나 세대를 이유로 한 비하가 있습니까?",
    # v2(2026-11-15): K-MHaS는 성적 지향을 성별에 넣는다("동성애" 비하 = gender). v1 문구엔 빠져 있어 성별 재현율이 51%였다.
    "gender": "이 댓글에 성별이나 성적 지향을 이유로 한 차별이나 비하가 있습니까?",
    # v2: v1에선 "캐나다 백인" 같은 인종 비하를 출신으로 판정해서, 특정 나라·지역이 아니라 인종을 겨냥한다는 점을 적었다.
    "race": "이 댓글에 백인, 흑인처럼 인종이나 피부색을 이유로 한 비하가 있습니까?",
    "religion": "이 댓글에 종교를 이유로 한 비하가 있습니까?",
}.items()}

DATASETS = {
    "kmhas": {
        "out": Path("docs/test-results/kmhas-20261115"),
        "questions": KMHAS_QUESTIONS,
        "multilabel": True,
    },
    "bitext": {
        "out": Path("docs/test-results/cs-bitext-20261115"),
        "question": {"type": "choice", "instructions": "Which category does this customer message belong to?", "criteria": BITEXT_CRITERIA},
    },
    "banking77": {
        "out": Path("docs/test-results/cs-banking77-20261115"),
        "question": {"type": "choice", "instructions": "Which intent does this bank customer message express?",
                     "criteria": {lab: lab.replace("_", " ").rstrip("?") for lab in BANKING77_LABELS}},
    },
}


def questions_of(d):
    return d.get("questions") or {"q": d["question"]}


def read_kmhas(split):
    rows = list(csv.reader(open(f"data/datasets/kmhas/kmhas_{split}.txt", encoding="utf-8"), delimiter="\t"))[1:]
    # gold는 혐오 카테고리 목록. "혐오 아님"(8)은 빈 목록이 된다.
    return [(t, sorted(KMHAS_NAMES[int(x)] for x in lab.split(",") if int(x) != 8)) for t, lab in rows]


def sample(ds, seed=42):
    out = DATASETS[ds]["out"]
    out.mkdir(parents=True, exist_ok=True)
    if ds == "kmhas":
        # 공식 테스트셋 21,939건에서 3,000건 무작위. 비율을 건드리지 않아야 실제 분포(혐오 아님 약 54%)대로 평가된다.
        rows = read_kmhas("test")
        idx = sorted(random.Random(seed).sample(range(len(rows)), 3000))
        picked = [{"id": i, "text": rows[i][0], "gold": rows[i][1]} for i in idx]
    elif ds == "banking77":
        # 공식 테스트셋(의도당 40건, 3,080건)을 그대로 쓴다. 학습셋과 겹치지 않음.
        picked = [{"id": i, "text": r["text"], "gold": r["category"]}
                  for i, r in enumerate(csv.DictReader(open("data/datasets/banking77-test.csv", encoding="utf-8")))]
        assert len(GROUP_OF) == 77 and set(GROUP_OF) == set(BANKING77_LABELS), set(GROUP_OF) ^ set(BANKING77_LABELS)
    else:
        rows = list(csv.DictReader(open("data/datasets/bitext-retail.csv", encoding="utf-8")))
        by_cat = collections.defaultdict(list)
        for i, r in enumerate(rows):
            by_cat[r["category"]].append({"id": i, "text": r["instruction"], "gold": r["category"], "intent": r["intent"]})
        assert set(by_cat) == set(BITEXT_CRITERIA), set(by_cat) ^ set(BITEXT_CRITERIA)
        # 묶음별로 같은 수를 뽑는다. 원본 비율대로 뽑으면 DELIVERY(7개 의도)가 SALES(1개)의 7배가 돼서 큰 묶음 점수가 전체를 덮는다.
        per = 5000 // len(by_cat)
        rnd = random.Random(seed)
        picked = [x for c in sorted(by_cat) for x in rnd.sample(by_cat[c], per)]
        # ponytail: 이 id 목록은 나중에 LoRA 학습 데이터에서 빼야 함(평가셋 누출 방지). 같은 원문의 오타 변형은 아직 못 걸러냄.
    (out / "sample.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in picked), encoding="utf-8")
    print(f"{ds}: {len(picked)}건 -> {out / 'sample.jsonl'}")


def trainset(ds, n=None, seed=42):
    """laya-train용 JSONL. 평가와 똑같은 질문(키 "q", 같은 보기 설명)을 써야 학습 효과만 따로 잴 수 있다."""
    if ds == "kmhas":
        rows = read_kmhas("train")
        if n:
            rows = random.Random(seed).sample(rows, n)
        out = Path(f"data/datasets/kmhas-train{'-' + str(n) if n else ''}.jsonl")
        out.write_text("".join(json.dumps({"state": t, "questions": KMHAS_QUESTIONS,
                                           "expected": {c: c in gold for c in KMHAS_QUESTIONS}}, ensure_ascii=False) + "\n"
                               for t, gold in rows), encoding="utf-8")
        print(f"{ds}: 학습 {len(rows)}건(질문 {len(rows) * len(KMHAS_QUESTIONS)}개) -> {out}")
        return
    q = DATASETS[ds]["question"]
    if ds == "banking77":
        rows = [(r["text"], r["category"]) for r in csv.DictReader(open("data/datasets/banking77-train.csv", encoding="utf-8"))]
    else:
        held = {json.loads(l)["id"] for l in (DATASETS[ds]["out"] / "sample.jsonl").open(encoding="utf-8")}
        rows = [(r["instruction"], r["category"]) for i, r in enumerate(csv.DictReader(open("data/datasets/bitext-retail.csv", encoding="utf-8")))
                if i not in held]
    out = Path(f"data/datasets/{ds}-train.jsonl")
    out.write_text("".join(json.dumps({"state": t, "questions": {"q": q}, "expected": {"q": y}}, ensure_ascii=False) + "\n"
                           for t, y in rows), encoding="utf-8")
    print(f"{ds}: 학습 {len(rows)}건 -> {out}")


def ask(cfg, key, questions, text):
    body = json.dumps({"model": cfg["model"], "state": text, "questions": questions, **cfg.get("extra", {})}).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    for attempt in range(5):
        req = urllib.request.Request(cfg["url"], data=body, headers=headers, method="POST")
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                res = json.load(r)
            return res, time.perf_counter() - t
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
        time.sleep(2 ** attempt)


def run(ds, name, limit=None):
    cfg, d = BACKENDS[name], DATASETS[ds]
    key = None
    if cfg["key_env"]:
        # 환경변수가 우선이고, 없으면 프로젝트 .env(git 제외)에서 읽는다. 키 값은 어디에도 출력하지 않는다.
        key = os.environ.get(cfg["key_env"])
        if not key and Path(".env").exists():
            for line in Path(".env").read_text(encoding="utf-8").splitlines():
                k, _, v = line.partition("=")
                if k.strip() == cfg["key_env"]:
                    key = v.strip().strip('"').strip("'")
        if not key:
            sys.exit(f"{cfg['key_env']}가 없음. 프로젝트 폴더의 .env 에 {cfg['key_env']}=키 한 줄을 넣을 것.")
    items = [json.loads(l) for l in (d["out"] / "sample.jsonl").open(encoding="utf-8")]
    out = d["out"] / f"{name}.jsonl"
    done = {json.loads(l)["id"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [x for x in items if x["id"] not in done][:limit]
    print(f"{ds}/{name}: 남은 {len(todo)}건 (이미 {len(done)}건)")
    errors = 0
    with out.open("a", encoding="utf-8") as f, cf.ThreadPoolExecutor(cfg["workers"]) as pool:
        futs = {pool.submit(ask, cfg, key, questions_of(d), x["text"]): x for x in todo}
        for k, fut in enumerate(cf.as_completed(futs), 1):
            x = futs[fut]
            try:
                res, sec = fut.result()
                ans = res["answers"]
                for a in ans.values():
                    if "error" in a:  # 질문 오류는 HTTP 200 + answers.<id>.error 로 옴
                        raise RuntimeError(a["error"])
                if d.get("multilabel"):
                    row = {**x, "probs": {c: ans[c]["noul"] for c in d["questions"]}, "sec": round(sec, 3), "usage": res.get("usage")}
                else:
                    a = ans["q"]
                    row = {**x, "pred": a["choice"], "confidence": a.get("confidence"),
                           "probabilities": a.get("probabilities"), "sec": round(sec, 3), "usage": res.get("usage")}
            except Exception as e:
                errors += 1
                print(f"  오류 id={x['id']}: {e}")
                continue  # 오류 건은 기록하지 않음 -> 다시 실행하면 그것만 재시도
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            if k % 500 == 0:
                print(f"  {k}/{len(todo)}")
    print(f"끝. 오류 {errors}건")


def score(rows):
    acc = sum(r["pred"] == r["gold"] for r in rows) / len(rows)
    per = {}
    for c in sorted({r["gold"] for r in rows}):
        g = [r for r in rows if r["gold"] == c]
        per[c] = sum(r["pred"] == c for r in g) / len(g)
    confusions = collections.Counter((r["gold"], r["pred"]) for r in rows if r["pred"] != r["gold"])
    # 확신도 구간별 실제 정확도: "90%라고 했을 때 진짜 90% 맞나"(사람한테 넘길 기준선 정하는 근거)
    bins = collections.defaultdict(list)
    for r in rows:
        if r.get("confidence") is not None:
            bins[min(int(r["confidence"] * 10), 9)].append(r["pred"] == r["gold"])
    calib = {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": (len(v), sum(v) / len(v)) for b, v in sorted(bins.items())}
    return acc, per, confusions, calib


def score_multilabel(rows, cats, t=0.5):
    """카테고리별 정밀도·재현율·F1, macro F1, '혐오 있음/없음' 정확도, 라벨 전체 일치율."""
    per = {}
    for c in cats:
        tp = sum(r["probs"][c] >= t and c in r["gold"] for r in rows)
        fp = sum(r["probs"][c] >= t and c not in r["gold"] for r in rows)
        fn = sum(r["probs"][c] < t and c in r["gold"] for r in rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"n": tp + fn, "p": p, "r": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0}
    macro = sum(v["f1"] for v in per.values()) / len(per)
    pred = [{c for c in cats if r["probs"][c] >= t} for r in rows]
    binary = sum(bool(p) == bool(r["gold"]) for p, r in zip(pred, rows)) / len(rows)
    exact = sum(p == set(r["gold"]) for p, r in zip(pred, rows)) / len(rows)
    return per, macro, binary, exact


def report(ds, name):
    rows = [json.loads(l) for l in (DATASETS[ds]["out"] / f"{name}.jsonl").open(encoding="utf-8")]
    if DATASETS[ds].get("multilabel"):
        per, macro, binary, exact = score_multilabel(rows, list(DATASETS[ds]["questions"]))
        secs = sorted(r["sec"] for r in rows)
        print(f"{ds}/{name}: {len(rows)}건, macro F1 {macro:.3f}, 혐오 여부 정확도 {binary:.1%}, 라벨 전체 일치 {exact:.1%}, "
              f"요청 중앙값 {secs[len(secs) // 2]:.2f}초 (기준 0.5)")
        for c, v in per.items():
            print(f"  {c:10s} 정답 {v['n']:4d}건  정밀도 {v['p']:.1%}  재현율 {v['r']:.1%}  F1 {v['f1']:.3f}")
        return
    acc, per, conf, calib = score(rows)
    secs = sorted(r["sec"] for r in rows)
    tok = sum((r.get("usage") or {}).get("input_tokens", 0) for r in rows)
    print(f"{ds}/{name}: {len(rows)}건, 정확도 {acc:.1%}, 요청 중앙값 {secs[len(secs) // 2]:.2f}초, 입력 토큰 합계 {tok:,}")
    if ds == "banking77":
        g = sum(GROUP_OF.get(r["pred"]) == GROUP_OF[r["gold"]] for r in rows) / len(rows)
        print(f"8개 그룹 기준 정확도 {g:.1%}")
        worst = sorted(per.items(), key=lambda kv: kv[1])[:12]
        print("약한 의도:", ", ".join(f"{c} {v:.0%}" for c, v in worst))
    else:
        print("묶음별:", ", ".join(f"{c} {v:.0%}" for c, v in per.items()))
    print("많이 헷갈린 쌍(정답->예측):", ", ".join(f"{g}->{p} {n}" for (g, p), n in conf.most_common(10)))
    print("확신도 구간별 (건수, 실제 정확도):", {k: (n, round(a, 3)) for k, (n, a) in calib.items()})
    for t in (0.5, 0.7, 0.8, 0.9):
        s = [r for r in rows if (r.get("confidence") or 0) >= t]
        if s:
            print(f"  기준선 {t}: 자동 처리 {len(s) / len(rows):.1%}, 그 안의 정확도 {sum(r['pred'] == r['gold'] for r in s) / len(s):.1%}")


def selftest():
    rows = [{"gold": "A", "pred": "A", "confidence": 0.95}, {"gold": "A", "pred": "B", "confidence": 0.55},
            {"gold": "B", "pred": "B", "confidence": 0.91}, {"gold": "B", "pred": "B", "confidence": 0.5}]
    acc, per, conf, calib = score(rows)
    assert acc == 0.75 and per == {"A": 0.5, "B": 1.0} and conf[("A", "B")] == 1
    assert calib == {"0.5-0.6": (2, 0.5), "0.9-1.0": (2, 1.0)}, calib
    ml = [{"gold": ["x"], "probs": {"x": 0.9, "y": 0.1}}, {"gold": [], "probs": {"x": 0.6, "y": 0.2}},
          {"gold": ["x", "y"], "probs": {"x": 0.8, "y": 0.3}}, {"gold": [], "probs": {"x": 0.1, "y": 0.1}}]
    per, macro, binary, exact = score_multilabel(ml, ["x", "y"])
    assert per["x"] == {"n": 2, "p": 2 / 3, "r": 1.0, "f1": 0.8} and per["y"]["f1"] == 0.0, per
    assert binary == 0.75 and exact == 0.5 and macro == 0.4, (binary, exact, macro)
    print("selftest ok")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["selftest"]
    if cmd == "sample":
        sample(args[0])
    elif cmd == "trainset":
        trainset(args[0], int(args[1]) if len(args) > 1 else None)
    elif cmd == "run":
        run(args[0], args[1], int(args[2]) if len(args) > 2 else None)
    elif cmd == "report":
        report(args[0], args[1])
    else:
        selftest()
