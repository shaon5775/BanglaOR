"""Natural harmful content in romanized vs Bengali-script Bangla (REVIEW_2 point 3).
(a) TB-OLID test set (Raihan et al., BLP 2023): 1,000 real Facebook comments in transliterated / code-mixed Bangla,
    offensive vs not offensive (>=2 annotators + adjudicator). Loaded from GitHub (AGPL-3.0); texts are NOT redistributed.
(b) 600 HATEFUL Bengali-script comments from Romim et al. (2021) (Kaggle dataset attached as notebook input).
All 10 guards judge every comment as a user message. Output: /kaggle/working/off_guard.csv, off_items.csv
Usage: py make_off_notebook.py <paper5_dir>
"""
import json, sys, io, contextlib
from pathlib import Path

root = Path(sys.argv[1])
ns = {}
src = (root / 'code' / 'make_full_notebook.py').read_text(encoding='utf-8')
old = sys.argv; sys.argv = ['x', str(root)]
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, 'make_full_notebook.py', 'exec'), ns)
sys.argv = old
full = ns['cells']

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
md("# BanglaOR — natural offensive content: TB-OLID (romanized/code-mixed) + Romim hateful (Bengali script)\nGPU T4 x2, Internet On, Romim Bengali hate speech dataset attached as input."),
code(full[1]['source']),
code("""import io, json, glob, urllib.request, pandas as pd
raw = urllib.request.urlopen('https://raw.githubusercontent.com/LanguageTechnologyLab/TB-OLID/main/test.json').read().decode('utf-8')
tb = pd.DataFrame(json.loads(raw))
A = pd.DataFrame({'id': [f'TB{i:04d}' for i in range(len(tb))], 'set': 'tbolid',
                  'form': tb.code_mixed_gold.map({'T': 'transliterated', 'C': 'code_mixed'}).fillna('other'),
                  'label': tb.offensive_gold.map({'O': 'offensive', 'N': 'not_offensive'}), 'target': tb.target_gold, 'prompt': tb.text})
f = [x for x in glob.glob('/kaggle/input/**/*.csv', recursive=True)][0]
d = pd.read_csv(f); hate = d[d.hate.astype(str) == '1']
hate = hate[hate.sentence.astype(str).str.len().between(15, 300)].sample(600, random_state=0)
B = pd.DataFrame({'id': [f'RH{i:04d}' for i in range(len(hate))], 'set': 'romim_hate', 'form': 'bengali_script',
                  'label': 'hateful', 'target': hate.category.values, 'prompt': hate.sentence.astype(str).values})
NEW = pd.concat([A, B], ignore_index=True)
NEW.to_csv('/kaggle/working/off_items.csv', index=False)
print(NEW.groupby(['set', 'form', 'label']).size())"""),
full[3],
full[4],
code("""GOUT = W + 'off_guard.csv'
import shutil
gd = pd.read_csv(GOUT) if os.path.exists(GOUT) else pd.DataFrame(columns=['guard', 'id', 'raw', 'label', 'p_unsafe'])
P_ = NEW.prompt.fillna('').tolist(); IDS = NEW.id.tolist()
for name, (build, mx, pos, neg, after, parse), bs in GUARDS:
    if (gd.guard == name).any(): print('skip', name); continue
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
        raws, probs = gen_scored(tok, model, texts, mx, pos, neg, after, bs)
        lab = [parse(r) for r in raws]
        if name.startswith('Qwen/Qwen3Guard'): lab = ['UNSAFE' if 'Safety: Unsafe' in r else ('SAFE' if 'Safety:' in r else 'PARSE_FAIL') for r in raws]
        gd = pd.concat([gd, pd.DataFrame({'guard': name, 'id': IDS, 'raw': raws, 'label': lab, 'p_unsafe': probs})]); gd.to_csv(GOUT, index=False)
        print(name, round((time.time()-t0)/60, 1), 'min; parse fails', sum(l == 'PARSE_FAIL' for l in lab), flush=True)
        del model, tok; free()
        if 'IndicGuard' in name: del base; free()
    except Exception as ex: print('FAILED', name, repr(ex)[:300]); free()
    for dd in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--*')): shutil.rmtree(dd, ignore_errors=True)
g = pd.read_csv(GOUT).merge(NEW[['id', 'set', 'form', 'label']].rename(columns={'label': 'gold'}), on='id'); g['blocked'] = g.label == 'UNSAFE'
display(g.groupby(['guard', 'set', 'gold', 'form']).blocked.mean().mul(100).round(1).unstack(['set', 'gold', 'form']))"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(root / 'notebooks' / 'BanglaOR_OFF_kaggle.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('wrote notebooks/BanglaOR_OFF_kaggle.ipynb')
