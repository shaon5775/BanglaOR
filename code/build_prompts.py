"""Build BanglaOR prompt CSV from the two markdown files (seeds + 3 forms).
Usage: py -3 -I build_prompts.py <paper5_dir>
Output: <paper5_dir>/data/BanglaOR_v1_400.csv  (id, seed, category, type, form, prompt)
"""
import csv, re, sys
from pathlib import Path

root = Path(sys.argv[1])
seeds_md = (root / "BanglaOR_seeds_v1.md").read_text(encoding="utf-8").splitlines()
forms_md = (root / "BanglaOR_forms_v1.md").read_text(encoding="utf-8").splitlines()

def rows(lines):
    cat = None
    for ln in lines:
        m = re.match(r"^## (\d+)\. (.+)$", ln)
        if m:
            cat = m.group(2).strip()
            continue
        m = re.match(r"^\| (\d+) \|(.+)\|\s*$", ln)
        if m:
            yield int(m.group(1)), cat, [c.strip() for c in m.group(2).split("|")]

seed = {}
for n, cat, cells in rows(seeds_md):
    bn, en, typ = cells
    seed[n] = dict(category=cat, type=typ, en=en, bangla_script=bn)
for n, cat, cells in rows(forms_md):
    banglish, mixed, kolkata = cells
    seed[n].update(banglish=banglish, mixed=mixed, kolkata_spoken=kolkata)

assert sorted(seed) == list(range(1, 101)), "seed numbers must be 1..100"
out = root / "data"
out.mkdir(exist_ok=True)
with open(out / "BanglaOR_v1_400.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "seed", "category", "type", "form", "english_meaning", "prompt"])
    for n in range(1, 101):
        s = seed[n]
        for form in ["bangla_script", "banglish", "mixed", "kolkata_spoken"]:
            assert s.get(form), f"missing {form} for seed {n}"
            w.writerow([f"{n:03d}_{form}", n, s["category"], s["type"], form, s["en"], s[form]])
print("wrote", out / "BanglaOR_v1_400.csv", "rows:", 4 * len(seed))
