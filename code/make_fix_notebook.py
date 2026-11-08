"""Reviewer-fix GPU notebook (8 Nov 2026, REVIEW_1 majors 2-5). Re-uses the guard code of the FULL study.
New texts for every guard:
  (a) english_full : 100 full English trigger questions (data/BanglaOR_english_full_100.csv)        [Major 3]
  (b) banglish_rom : MultiJail Bengali prompts romanized with aksharamukha (rule-based, no new text) [Major 4]
  (c) social       : 600 NON-hateful real Bengali Facebook/YouTube comments (Romim et al. 2021)      [Major 5]
IndicGuard additionally re-runs ALL 2,295 original texts with NO time cap                             [Major 2]
Output: /kaggle/working/fix_guard.csv (+ fix_items.csv). Analysis is done locally (code/analyse_fix.py).
Usage: py make_fix_notebook.py <paper5_dir>
"""
import json, sys, io, contextlib
from pathlib import Path

root = Path(sys.argv[1])
ns = {}
src = (root / 'code' / 'make_full_notebook.py').read_text(encoding='utf-8')
old_argv = sys.argv; sys.argv = ['x', str(root)]
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, 'make_full_notebook.py', 'exec'), ns)
sys.argv = old_argv
full = ns['cells']
ef = (root / 'data' / 'BanglaOR_english_full_100.csv').read_text(encoding='utf-8')

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
md("""# BanglaOR — reviewer-fix run (IndicGuard full, English full questions, romanized harm, real comments)
GPU T4 x2, Internet On. Resumes per guard (/kaggle/working/fix_guard.csv)."""),
code(full[1]['source'] + "\n!pip install -q aksharamukha kagglehub"),
full[2],
code("EF = " + repr(ef) + """
from aksharamukha import transliterate
ef = pd.read_csv(io.StringIO(EF), dtype=str)
A = pd.DataFrame({'id': [f'{int(s):03d}_english_full' for s in ef.seed], 'seed': ef.seed.astype(str), 'set': 'trigger',
                  'form': 'english_full', 'prompt': ef.english_full})
hb = items[(items.set == 'harmful') & (items.form == 'bangla_script')]
import re as _re
rom = [_re.sub(r'(aa|ii|uu|ee|oo)', lambda m: m.group(0)[0], transliterate.process('Bengali', 'RomanReadable', t)) for t in hb.prompt]
B = pd.DataFrame({'id': hb.id.str.replace('_bangla_script', '_banglish_rom'), 'seed': hb.seed, 'set': 'harmful',
                  'form': 'banglish_rom', 'prompt': rom})
print(B.prompt.head(3).tolist())
# real social-media comments (Romim et al. 2021, Kaggle copy); keep NON-hateful ones only
import kagglehub, glob
C = pd.DataFrame(columns=['id', 'seed', 'set', 'form', 'prompt'])
try:
    p = kagglehub.dataset_download('naurosromim/bengali-hate-speech-dataset')
    f = [x for x in glob.glob(p + '/**/*.csv', recursive=True)][0]
    d = pd.read_csv(f); print(f, d.shape, list(d.columns)); print(d.head(3))
    tcol = max([c for c in d.columns if d[c].dtype == object], key=lambda c: d[c].astype(str).str.len().mean())
    lcol = [c for c in d.columns if c.lower() in ('hate', 'label', 'class', 'hate_speech')][0]
    print('text col', tcol, 'label col', lcol, d[lcol].value_counts().to_dict())
    nonhate = d[d[lcol].astype(str).str.lower().isin(['0', 'non-hate', 'nonhate', 'not hate', 'nh', 'false'])]
    nonhate = nonhate[nonhate[tcol].astype(str).str.len().between(15, 300)].sample(600, random_state=0)
    lat = nonhate[tcol].astype(str).map(lambda s: sum(ch.isascii() and ch.isalpha() for ch in s) / max(1, sum(ch.isalpha() for ch in s)))
    C = pd.DataFrame({'id': [f'S{i:04d}_social' for i in range(len(nonhate))], 'seed': [f'S{i:04d}' for i in range(len(nonhate))],
                      'set': 'social', 'form': ['social_roman' if l > 0.5 else 'social_bengali' for l in lat],
                      'prompt': nonhate[tcol].astype(str).values})
    if 'category' in nonhate: C['category'] = nonhate['category'].values
    print(C.form.value_counts())
except Exception as ex:
    print('SOCIAL DATA FAILED', repr(ex)[:300])
NEW = pd.concat([A, B, C], ignore_index=True)
NEW.to_csv('/kaggle/working/fix_items.csv', index=False)
print(len(NEW), 'new texts'); print(NEW.groupby(['set', 'form']).size())"""),
full[3],
full[4],
code("""# ---- run: every guard on the NEW texts; IndicGuard also on ALL original texts (no time cap) ----
GOUT = W + 'fix_guard.csv'
gd = pd.read_csv(GOUT) if os.path.exists(GOUT) else pd.DataFrame(columns=['guard', 'id', 'raw', 'label', 'p_unsafe'])
for name, (build, mx, pos, neg, after, parse), bs in GUARDS:
    if (gd.guard == name).any(): print('skip', name); continue
    X = pd.concat([NEW, ALL[['id', 'seed', 'set', 'form', 'prompt']]], ignore_index=True) if 'IndicGuard' in name else NEW
    P_ = X.prompt.fillna('').tolist(); IDS = X.id.tolist()
    try:
        t0 = time.time()
        if 'IndicGuard' in name:
            from peft import PeftModel
            tok = AutoTokenizer.from_pretrained(name); tok.padding_side = 'left'
            base = AutoModelForCausalLM.from_pretrained('unsloth/gemma-3-4b-it-unsloth-bnb-4bit', device_map={'': 0}, torch_dtype=torch.float32)
            model = PeftModel.from_pretrained(base, name).eval()
        else:
            tok, model = load(name)
        texts = [build(tok, q) for q in P_]
        raws, probs = gen_scored(tok, model, texts, mx, pos, neg, after, bs, max_min=None)
        lab = [parse(r) for r in raws]
        if name.startswith('Qwen/Qwen3Guard'): lab = ['UNSAFE' if 'Safety: Unsafe' in r else ('SAFE' if 'Safety:' in r else 'PARSE_FAIL') for r in raws]
        gd = pd.concat([gd, pd.DataFrame({'guard': name, 'id': IDS, 'raw': raws, 'label': lab, 'p_unsafe': probs})]); gd.to_csv(GOUT, index=False)
        print(name, len(IDS), 'texts', round((time.time()-t0)/60, 1), 'min; parse fails', sum(l == 'PARSE_FAIL' for l in lab), flush=True)
        del model, tok; free()
        if 'IndicGuard' in name: del base; free()
    except Exception as ex: print('FAILED', name, repr(ex)[:400]); free()
    import shutil
    for d in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--*')): shutil.rmtree(d, ignore_errors=True)
g = pd.read_csv(GOUT).merge(pd.concat([NEW, ALL])[['id', 'set', 'form']].drop_duplicates('id'), on='id')
g['blocked'] = g.label == 'UNSAFE'
display(g.groupby(['guard', 'set', 'form']).blocked.mean().mul(100).round(1).unstack(['set', 'form']))"""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(root / 'notebooks' / 'BanglaOR_FIX_reviewer_kaggle.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('wrote notebooks/BanglaOR_FIX_reviewer_kaggle.ipynb')
