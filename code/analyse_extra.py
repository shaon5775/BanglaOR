"""REVIEW_2 extra analyses (local, CPU):
 (1) bootstrap 95% CIs for ROC-AUC in English and Bengali script and their difference (seeds resampled, 2,000 reps)
 (2) per-language threshold calibration as a mitigation: 5-fold cross-fitting over seeds, Bengali threshold set
     to reach 90% Bengali harm recall on training folds; compare with reusing the English-90% threshold
 (3) tokenizer check: tokens per character by form, and Spearman correlation of token count with P(unsafe)
     on harmless texts, for the guards' own tokenizers
Outputs: results/final/auc_ci.csv, calibration.csv, tokenizer_check.csv
"""
import csv, collections, json, math, random
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

P = Path(__file__).resolve().parents[1]; OUT = P / 'results' / 'final'
rng = np.random.default_rng(0)

def meta(i):
    base, tr = (i[:-4], True) if i.endswith('__tr') else (i, False)
    seed, form = base.split('_', 1)
    st = 'harmful' if seed.startswith('H') else 'neutral' if seed.startswith('N') else 'trigger'
    return seed, st, form + ('__tr' if tr else '')
prob = collections.defaultdict(dict)   # (guard, set, form) -> {seed: p}
for src in ['results/full/full_guard.csv', 'results/fix/fix_guard.csv']:
    for r in csv.DictReader(open(P / src, encoding='utf-8')):
        g = r['guard'].split('/')[-1]
        if src.endswith('full_guard.csv') and g == 'IndicGuard': continue
        if r['label'] == 'NOT_RUN' or r['p_unsafe'] in ('', 'nan'): continue
        seed, st, form = meta(r['id'])
        if st == 'trigger' and form == 'english': continue   # short gloss, not the full-question reference
        if form == 'english_full': form = 'english'
        key = seed if st != 'trigger' else 'T' + seed
        prob[(g, st, form)][key] = float(r['p_unsafe'])
G = sorted({k[0] for k in prob})

def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    vals, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(vals)); np.add.at(sums, inv, r); r = (sums / cnt)[inv]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def data(g, form):
    h = prob[(g, 'harmful', form)]; t = {**prob[(g, 'trigger', form)], **prob[(g, 'neutral', form)]}
    return h, t
rows = []
for g in G:
    he, te = data(g, 'english'); hb, tb = data(g, 'bangla_script')
    hk = sorted(set(he) & set(hb)); tk = sorted(set(te) & set(tb))
    HE = np.array([he[k] for k in hk]); HB = np.array([hb[k] for k in hk]); TE = np.array([te[k] for k in tk]); TB = np.array([tb[k] for k in tk])
    a_e, a_b = auc(HE, TE), auc(HB, TB); bs = []
    for _ in range(2000):
        i = rng.integers(0, len(hk), len(hk)); j = rng.integers(0, len(tk), len(tk))
        bs.append((auc(HE[i], TE[j]), auc(HB[i], TB[j])))
    bs = np.array(bs); d = bs[:, 0] - bs[:, 1]
    q = lambda x: (round(float(np.quantile(x, .025)), 3), round(float(np.quantile(x, .975)), 3))
    rows.append(dict(guard=g, auc_en=round(a_e, 3), en_lo=q(bs[:, 0])[0], en_hi=q(bs[:, 0])[1], auc_bn=round(a_b, 3),
                     bn_lo=q(bs[:, 1])[0], bn_hi=q(bs[:, 1])[1], diff=round(a_e - a_b, 3), diff_lo=q(d)[0], diff_hi=q(d)[1]))
with open(OUT / 'auc_ci.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print('AUC', r)

# (2) calibration: 5-fold cross-fitting on Bengali script
cal = []
for g in G:
    he, te = data(g, 'english'); hb, tb = data(g, 'bangla_script')
    thr_en = float(np.quantile(list(he.values()), 0.10))
    hk = sorted(hb); tk = sorted(tb); random.Random(1).shuffle(hk); random.Random(2).shuffle(tk)
    fold_h = {k: i % 5 for i, k in enumerate(hk)}; fold_t = {k: i % 5 for i, k in enumerate(tk)}
    rec, fb = [], []
    for f in range(5):
        trh = [hb[k] for k in hk if fold_h[k] != f]; thr = float(np.quantile(trh, 0.10))
        rec += [hb[k] >= thr for k in hk if fold_h[k] == f]; fb += [tb[k] >= thr for k in tk if fold_t[k] == f]
    cal.append(dict(guard=g, en_thr_recall_bn=round(100 * np.mean([v >= thr_en for v in hb.values()]), 1),
                    en_thr_fb_bn=round(100 * np.mean([v >= thr_en for v in tb.values()]), 1),
                    bn_thr_recall_bn=round(100 * np.mean(rec), 1), bn_thr_fb_bn=round(100 * np.mean(fb), 1),
                    en_fb_en=round(100 * np.mean([v >= thr_en for v in te.values()]), 1)))
with open(OUT / 'calibration.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, list(cal[0])); w.writeheader(); w.writerows(cal)
for r in cal: print('CAL', r)

# (3) tokenizer check
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
TOKS = {'Qwen3Guard-Gen-8B': 'Qwen/Qwen3Guard-Gen-8B', 'granite-guardian-3.2-5b': 'ibm-granite/granite-guardian-3.2-5b',
        'granite-guardian-3.1-2b': 'ibm-granite/granite-guardian-3.1-2b', 'IndicGuard': 'unsloth/gemma-3-4b-it-unsloth-bnb-4bit',
        'PolyGuard-Qwen-Smol': 'ToxicityPrompts/PolyGuard-Qwen-Smol'}
texts = {}
for f in ['data/BanglaOR_v1_400.csv', 'data/BanglaOR_neutral_50.csv']:
    for r in csv.DictReader(open(P / f, encoding='utf-8')):
        texts[r['id']] = (r['form'], r['prompt'], 'N' if r['id'].startswith('N') else 'T')
for r in csv.DictReader(open(P / 'data/BanglaOR_english_full_100.csv', encoding='utf-8')):
    texts[f"{int(r['seed']):03d}_english"] = ('english', r['english_full'], 'T')
pr = collections.defaultdict(dict)
for src in ['results/full/full_guard.csv', 'results/fix/fix_guard.csv']:
    for r in csv.DictReader(open(P / src, encoding='utf-8')):
        g = r['guard'].split('/')[-1]
        if g not in TOKS or r['label'] == 'NOT_RUN': continue
        i = r['id'].replace('_english_full', '_english')
        if src.endswith('full_guard.csv') and (g == 'IndicGuard' or (i[0].isdigit() and i.endswith('_english'))): continue
        pr[g][i] = (float(r['p_unsafe']) if r['p_unsafe'] not in ('', 'nan') else float('nan'), r['label'] == 'UNSAFE')
tk_rows = []
for g, repo in TOKS.items():
    tok = Tokenizer.from_file(hf_hub_download(repo, 'tokenizer.json'))
    n = {i: len(tok.encode(t[1]).ids) for i, t in texts.items()}
    for form in ['english', 'bangla_script', 'kolkata_spoken', 'mixed', 'banglish']:
        ids = [i for i, t in texts.items() if t[0] == form and i in pr[g]]
        tpc = np.mean([n[i] / max(1, len(texts[i][1])) for i in ids])
        tk_rows.append(dict(guard=g, form=form, n=len(ids), tokens_per_char=round(float(tpc), 3),
                            mean_tokens=round(float(np.mean([n[i] for i in ids])), 1),
                            block_pct=round(100 * np.mean([pr[g][i][1] for i in ids]), 1)))
    ids = [i for i in texts if i in pr[g] and pr[g][i][0] == pr[g][i][0]]
    rho, p = spearmanr([n[i] for i in ids], [pr[g][i][0] for i in ids])
    rho2, p2 = spearmanr([n[i] / max(1, len(texts[i][1])) for i in ids], [pr[g][i][0] for i in ids])
    tk_rows.append(dict(guard=g, form='ALL harmless: spearman(tokens, P)', n=len(ids), tokens_per_char=round(float(rho2), 3),
                        mean_tokens=round(float(rho), 3), block_pct=f'p={p:.1e}; p_tpc={p2:.1e}'))
with open(OUT / 'tokenizer_check.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, list(tk_rows[0])); w.writeheader(); w.writerows(tk_rows)
for r in tk_rows: print('TOK', r)
