"""Translation-quality check for the translate-first fix (H4).
Joins every Qwen2.5-7B translation with its source text and the reference English
(english_meaning for BanglaOR prompts, MultiJail 'en' for harmful ones), then draws a
stratified random sample of 100: 40 trigger (10/form), 20 neutral (5/form), 40 harmful."""
import csv, random
from pathlib import Path
P = Path(__file__).resolve().parents[1]
tr = {r['id']: r['translation'] for r in csv.DictReader(open(P/'results/c1/c1_translations.csv', encoding='utf-8'))}
rows = []
for f, grp in [('data/BanglaOR_v1_400.csv', 'trigger'), ('data/BanglaOR_neutral_50.csv', 'neutral')]:
    for r in csv.DictReader(open(P/f, encoding='utf-8')):
        if r['id'] in tr:
            rows.append(dict(id=r['id'], group=grp, form=r['form'], source=r['prompt'], reference_en=r['english_meaning'], translation=tr[r['id']]))
for r in csv.DictReader(open(P/'data/external/MultiJail.csv', encoding='utf-8-sig')):
    i = f"H{int(r['id']):03d}_bangla_script"
    if i in tr:
        rows.append(dict(id=i, group='harmful', form='bangla_script', source=r['bn'], reference_en=r['en'], translation=tr[i]))
assert len(rows) == len(tr), (len(rows), len(tr))
cols = ['id', 'group', 'form', 'source', 'reference_en', 'translation']
with open(P/'results/translation_check/all_translations_joined.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)
rng = random.Random(42)
sample = []
forms = ['bangla_script', 'kolkata_spoken', 'mixed', 'banglish']
for grp, per in [('trigger', 10), ('neutral', 5)]:
    for fm in forms:
        pool = [r for r in rows if r['group'] == grp and r['form'] == fm]
        sample += rng.sample(pool, per)
sample += rng.sample([r for r in rows if r['group'] == 'harmful'], 40)
for k, r in enumerate(sample, 1):
    r['n'] = k
out = ['n'] + cols + ['meaning_1to5', 'trigger_or_harm_kept_YN', 'note']
assert not (P/'results/translation_check/LABEL_ME_translation_100.csv').exists(), 'owner file exists - do not overwrite'
with open(P/'results/translation_check/LABEL_ME_translation_100.csv', 'w', encoding='utf-8-sig', newline='') as fh:
    w = csv.DictWriter(fh, out, extrasaction='ignore'); w.writeheader()
    for r in sample: w.writerow({**r, 'meaning_1to5': '', 'trigger_or_harm_kept_YN': '', 'note': ''})
print(len(rows), 'joined;', len(sample), 'sampled')
