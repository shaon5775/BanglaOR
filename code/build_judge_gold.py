"""Idea 1 (judge reliability): sample 400 chat answers from P1 (1,200) + P2 (1,000) for human gold labels.
Stratified: 200 that the big judge called a refusal, 200 it did not; spread over the 5 forms and 5 models.
Usage: py -3 -I build_judge_gold.py <paper5_dir>
Out: data/judge_gold_400.csv (all info, used by the notebook)
     LABEL_ME_judge_gold_400.csv (Excel-friendly sheet for the owner: question, meaning, answer, label)
"""
import csv, random, sys
from pathlib import Path
from collections import defaultdict

root = Path(sys.argv[1]); random.seed(5)
rows = []
for src, f in [("P1", "results/pilot_v1/pilot_judged.csv"), ("P2", "results/p2/p2_chat_judged.csv")]:
    for r in csv.DictReader(open(root / f, encoding="utf-8")):
        bigref = r["judge"] in ("FULL_REFUSAL", "PARTIAL_REFUSAL")
        rows.append(dict(src=src, model=r["model"].split("/")[-1], id=r["id"], form=r["form"], type=r["type"],
                         category=r["category"], english_meaning=r["english_meaning"], prompt=r["prompt"],
                         response=r["response"], kw_refusal=r["kw_refusal"], big_judge=r["judge"],
                         big_judge_refusal=bigref))
strata = defaultdict(list)
for r in rows: strata[(r["big_judge_refusal"], r["form"])].append(r)
pick = []
for flag in (True, False):
    keys = sorted(k for k in strata if k[0] == flag)
    pool = {k: random.sample(strata[k], len(strata[k])) for k in keys}
    got = 0
    while got < 200:            # round-robin over forms so every form is represented
        for k in keys:
            if pool[k] and got < 200: pick.append(pool[k].pop()); got += 1
for r in pick: r["stratum_size"] = len(strata[(r["big_judge_refusal"], r["form"])])
random.shuffle(pick)
for i, r in enumerate(pick, 1): r["row"] = i
cols = ["row", "src", "model", "id", "form", "type", "category", "english_meaning", "prompt", "response", "kw_refusal",
        "big_judge", "big_judge_refusal", "stratum_size"]
with open(root / "data" / "judge_gold_400.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow({c: r[c] for c in cols}) for r in pick]
with open(root / "LABEL_ME_judge_gold_400.csv", "w", newline="", encoding="utf-8-sig") as f:   # BOM so Excel shows Bangla
    w = csv.writer(f)
    w.writerow(["row", "question", "english_meaning", "answer", "YOUR_LABEL (A / D / P / R)", "note (optional)"])
    for r in pick: w.writerow([r["row"], r["prompt"], r["english_meaning"], r["response"][:1500], "", ""])
from collections import Counter
print(len(pick), Counter(r["form"] for r in pick), Counter(r["big_judge_refusal"] for r in pick), Counter(r["model"] for r in pick))
