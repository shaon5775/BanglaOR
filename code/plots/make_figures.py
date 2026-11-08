"""All paper figures for BanglaOR (SNAM submission). Reads only the result CSVs; writes vector PDFs to
paper/figures/. Palette: validated 3-slot categorical (blue/orange/aqua, colour-blind safe all-pairs),
sequential blue ramp for heat-maps; marker shapes + direct labels as secondary encoding (greyscale-safe)."""
import csv, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

P = Path(__file__).resolve().parents[2]
OUT = P / 'paper' / 'figures'; OUT.mkdir(parents=True, exist_ok=True)
EN, BN, TR, VI = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
INK, MUTED, GRID, AXIS = '#1f1f1d', '#6b6a64', '#e1e0d9', '#c3c2b7'
SEQ = LinearSegmentedColormap.from_list('blue', ['#f7f9fc', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'axes.linewidth': 0.6,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
MM = 1 / 25.4
FULL, COL = 174 * MM, 84 * MM

GUARDS = [  # (file key in summary csv, raw key in full_guard.csv, display name)
    ('granite-guardian-3.1-2b', 'ibm-granite/granite-guardian-3.1-2b', 'Granite Guardian 3.1-2B'),
    ('granite-guardian-3.2-5b', 'ibm-granite/granite-guardian-3.2-5b', 'Granite Guardian 3.2-5B'),
    ('granite-guardian-3.3-8b', 'ibm-granite/granite-guardian-3.3-8b', 'Granite Guardian 3.3-8B'),
    ('granite-guardian-4.1-8b', 'ibm-granite/granite-guardian-4.1-8b', 'Granite Guardian 4.1-8B'),
    ('Qwen3Guard-Gen-0.6B', 'Qwen/Qwen3Guard-Gen-0.6B', 'Qwen3Guard-Gen-0.6B'),
    ('Qwen3Guard-Gen-4B', 'Qwen/Qwen3Guard-Gen-4B', 'Qwen3Guard-Gen-4B'),
    ('Qwen3Guard-Gen-8B', 'Qwen/Qwen3Guard-Gen-8B', 'Qwen3Guard-Gen-8B'),
    ('PolyGuard-Qwen-Smol', 'ToxicityPrompts/PolyGuard-Qwen-Smol', 'PolyGuard-Qwen-Smol'),
    ('PolyGuard-Qwen', 'ToxicityPrompts/PolyGuard-Qwen', 'PolyGuard-Qwen'),
    ('IndicGuard', 'l3cube-pune/IndicGuard', 'IndicGuard'),
]
FORMS = ['english', 'bangla_script', 'kolkata_spoken', 'mixed', 'banglish']
FORM_LBL = ['English', 'Bengali\nscript', 'Kolkata\nspoken', 'Code-\nmixed', 'Banglish\n(romanized)']

S = collections.defaultdict(dict)
for r in csv.DictReader(open(P / 'results/final/summary.csv', encoding='utf-8')):
    S[(r['guard'], r['set'])][r['form']] = float(r['block_pct'])

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight'); fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig); print('wrote', name)

# ---------- Fig 1: benchmark + pipeline overview ----------
def fig_overview():
    fig, ax = plt.subplots(figsize=(FULL, 62 * MM)); ax.set_xlim(0, 174); ax.set_ylim(0, 62); ax.axis('off')
    def box(x, y, w, h, title, lines, fc='#f3f6fb', ec='#86b6ef'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6,rounding_size=1.5', fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 3.2, title, ha='center', va='top', fontsize=7.5, fontweight='bold')
        for i, l in enumerate(lines):
            ax.text(x + 1.8, y + h - 8.6 - i * 3.95, l, ha='left', va='top', fontsize=5.6, color=INK)
    box(1, 4, 54, 54, 'BanglaOR benchmark', [
        'Harmless trigger set: 100 seeds x 5 forms', '  (kill, sacrifice, bomb, poison ...)',
        'Neutral control set: 50 seeds x 5 forms', '  (no trigger words)',
        'Harmful: MultiJail 315 x en/bn/roman',
        'Translate-first copies: 915 texts', 'Real comments: 600 (Facebook/YouTube)',
        'Forms: English | Bengali script |', '  Kolkata spoken | code-mixed | Banglish', '= 3,310 texts per guard'])
    box(63, 4, 50, 54, 'Systems under audit', [
        '10 open safety guards', '  Granite Guardian 3.1 / 3.2 / 3.3 / 4.1',
        '  Qwen3Guard-Gen 0.6B / 4B / 8B', '  PolyGuard-Qwen-Smol / -Qwen', '  IndicGuard',
        '7 light toxicity classifiers', '5 chat models (pilots)', '6 refusal judges',
        '', 'Label + P(unsafe) per text', 'Free Kaggle T4 GPUs, greedy'], fc='#fdf3ee', ec='#f0a07f')
    box(121, 4, 52, 54, 'Pre-specified questions', [
        'RQ1 language penalty (false blocks)', 'RQ2 harm recall, English vs Bengali',
        'RQ3 one threshold for both languages?', 'RQ4 translate-first mitigation',
        'RQ5 drift across guard versions', '', 'Exact McNemar + Holm, Wilson CIs,',
        "ROC-AUC, Cochran's Q", '', 'Gates: pilot -> confirmation -> full'], fc='#eef8f3', ec='#7fd3b1')
    for x0 in (55.8, 113.8):
        ax.annotate('', xy=(x0 + 6.4, 31), xytext=(x0 + 0.8, 31), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1))
    save(fig, 'fig1_overview')

# ---------- Fig 2: false-block heat-maps ----------
def fig_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 82 * MM), gridspec_kw={'wspace': 0.08})
    for ax, st, title in zip(axes, ['trigger', 'neutral'], ['(a) Harmless prompts with trigger words', '(b) Neutral prompts (no trigger words)']):
        M = np.array([[S[(g, st)].get(f, np.nan) for f in FORMS] for g, _, _ in GUARDS])
        ax.imshow(M, cmap=SEQ, vmin=0, vmax=85, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]; ax.text(j, i, f'{int(v + 0.5)}', ha='center', va='center', fontsize=7,
                                     color='white' if v > 45 else INK, fontweight='bold' if v >= 20 else 'normal')
        ax.set_xticks(range(5)); ax.set_xticklabels(FORM_LBL, fontsize=6.8)
        ax.set_yticks(range(len(GUARDS))); ax.set_yticklabels([d for _, _, d in GUARDS] if st == 'trigger' else [], fontsize=7)
        ax.set_title(title, fontsize=8, loc='left'); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True); ax.set_yticks(np.arange(-.5, len(GUARDS), 1), minor=True)
        ax.grid(which='minor', color='white', lw=1.5); ax.tick_params(which='minor', length=0)
    save(fig, 'fig2_false_block_heatmap')

# ---------- Fig 3: harmful recall dumbbell ----------
def fig_recall():
    rows = [(d, S[(g, 'harmful')]['english'], S[(g, 'harmful')]['bangla_script'], S[(g, 'harmful')]['bangla_script__tr'], S[(g, 'harmful')]['banglish_rom']) for g, _, d in GUARDS]
    rows.sort(key=lambda r: r[2] - r[1])
    fig, ax = plt.subplots(figsize=(FULL * 0.78, 78 * MM))
    for i, (d, e, b, t, rm) in enumerate(rows):
        ax.scatter(rm, i, s=30, marker='D', color=VI, zorder=2, edgecolor='white', lw=0.8)
        ax.plot([min(e, b), max(e, b)], [i, i], color=AXIS, lw=2, zorder=1)
        ax.scatter(e, i, s=34, marker='o', color=EN, zorder=3, edgecolor='white', lw=0.8)
        ax.scatter(b, i, s=34, marker='s', color=BN, zorder=3, edgecolor='white', lw=0.8)
        ax.scatter(t, i, s=40, marker='^', color=TR, zorder=2, edgecolor='white', lw=0.8)
        ax.text(101, i, f'{b - e:+.0f}', va='center', fontsize=7, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(-2, 100); ax.set_xlabel('Harmful MultiJail prompts caught (%)')
    ax.text(101, -0.9, 'Bn - En', fontsize=6.5, color=MUTED, va='center')
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    h = [plt.Line2D([], [], marker=m, ls='', color=c, ms=6, label=l) for m, c, l in
         [('o', EN, 'English'), ('s', BN, 'Bengali script'), ('D', VI, 'romanized Bengali'), ('^', TR, 'translated to English')]]
    ax.legend(handles=h, loc='lower left', frameon=False, fontsize=6.8, ncol=4, bbox_to_anchor=(-0.02, 1.0), columnspacing=0.9, handletextpad=0.3)
    save(fig, 'fig3_harm_recall')

# ---------- Fig 4: ROC curves ----------
def roc(pos, neg):
    s = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(-s); y = y[o]
    tpr = np.r_[0, np.cumsum(y) / y.sum()]; fpr = np.r_[0, np.cumsum(1 - y) / (1 - y).sum()]
    return fpr, tpr, np.trapz(tpr, fpr)

def fig_roc():
    raw = collections.defaultdict(list)
    for src in ['results/full/full_guard.csv', 'results/fix/fix_guard.csv']:
        for r in csv.DictReader(open(P / src, encoding='utf-8')):
            if r['label'] == 'NOT_RUN' or r['p_unsafe'] in ('', 'nan'): continue
            g = r['guard']; i = r['id']; form = i.split('_', 1)[1]
            st = 'harmful' if i.startswith('H') else 'neutral' if i.startswith('N') else 'trigger'
            if src.endswith('full_guard.csv') and st == 'trigger' and form == 'english': continue
            if form == 'english_full': form = 'english'
            raw[(g, form, st)].append(float(r['p_unsafe']))
    pick = [('ibm-granite/granite-guardian-3.2-5b', 'Granite Guardian 3.2-5B'), ('ToxicityPrompts/PolyGuard-Qwen-Smol', 'PolyGuard-Qwen-Smol'),
            ('ibm-granite/granite-guardian-4.1-8b', 'Granite Guardian 4.1-8B'), ('Qwen/Qwen3Guard-Gen-8B', 'Qwen3Guard-Gen-8B')]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 50 * MM), sharey=True, gridspec_kw={'wspace': 0.12})
    for ax, (g, d) in zip(axes, pick):
        for form, c, ls, lab in [('english', EN, '-', 'English'), ('bangla_script', BN, '--', 'Bengali')]:
            fpr, tpr, a = roc(raw[(g, form, 'harmful')], raw[(g, form, 'trigger')] + raw[(g, form, 'neutral')])
            ax.plot(fpr, tpr, color=c, ls=ls, lw=1.6, label=f'{lab} (AUC {a:.2f})')
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.6, ls=':')
        ax.set_title(d, fontsize=7.2); ax.set_xlabel('False-block rate'); ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=6, frameon=False, handlelength=1.6)
        ax.grid(color=GRID, lw=0.4)
    axes[0].set_ylabel('Harm recall')
    save(fig, 'fig4_roc')

# ---------- Fig 5: Granite version drift ----------
def fig_drift():
    vers = ['3.1-2B', '3.2-5B', '3.3-8B', '4.1-8B']; keys = [g for g, _, _ in GUARDS[:4]]
    fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.9, 66 * MM), gridspec_kw={'wspace': 0.3})
    ax = axes[0]
    for st_form, c, m, lab in [('english', EN, 'o', 'English'), ('bangla_script', BN, 's', 'Bengali script'), ('banglish_rom', VI, 'D', 'romanized'), ('bangla_script__tr', TR, '^', 'translated')]:
        y = [S[(k, 'harmful')][st_form] for k in keys]; ax.plot(vers, y, color=c, marker=m, lw=1.8, ms=5, label=lab)
    ax.set_ylim(-3, 100); ax.set_ylabel('Harmful prompts caught (%)'); ax.set_title('(a) Harm recall', fontsize=8, loc='left')
    ax = axes[1]
    for form, c, m, lab in [('english', EN, 'o', 'English'), ('bangla_script', BN, 's', 'Bengali script'), ('kolkata_spoken', '#b04a1e', 'D', 'Kolkata spoken'), ('banglish', '#7a5a00', 'v', 'Banglish')]:
        y = [S[(k, 'neutral')][form] for k in keys]; ax.plot(vers, y, color=c, marker=m, lw=1.6, ms=5, label=lab)
    ax.set_ylim(-3, 60); ax.set_ylabel('Neutral prompts blocked (%)'); ax.set_title('(b) False blocks, no trigger words', fontsize=8, loc='left')
    for ax in axes:
        ax.set_xlabel('Granite Guardian version'); ax.grid(axis='y', color=GRID, lw=0.5); ax.set_xlim(-0.25, 3.25)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=6.4, handlelength=1.8)
    save(fig, 'fig5_version_drift')

# ---------- Fig 6: translate-first ----------
def fig_translate():
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 70 * MM), sharey=True, gridspec_kw={'wspace': 0.08})
    names = [d for _, _, d in GUARDS]
    for ax, mode in zip(axes, ['fb', 'rec']):
        for i, (g, _, d) in enumerate(GUARDS):
            if mode == 'fb':
                vals = {f: S[(g, 'trigger')].get(f) for f in FORMS[1:]}; worst = max(vals, key=vals.get)
                a, b, e = vals[worst], S[(g, 'trigger')][worst + '__tr'], S[(g, 'trigger')]['english']
            else:
                a, b, e = S[(g, 'harmful')]['bangla_script'], S[(g, 'harmful')]['bangla_script__tr'], S[(g, 'harmful')]['english']
            ax.annotate('', xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=1, shrinkA=3, shrinkB=3))
            ax.scatter(a, i, s=30, marker='s', color=BN, zorder=3); ax.scatter(b, i, s=36, marker='^', color=TR, zorder=3)
            ax.scatter(e, i, s=26, marker='|', color=EN, zorder=4, lw=1.6)
        ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels(names, fontsize=7); axes[0].invert_yaxis()
    axes[0].set_xlabel('Trigger prompts blocked (%)\n(worst Bangla form)'); axes[0].set_title('(a) False blocks', fontsize=8, loc='left')
    axes[1].set_xlabel('Harmful Bengali prompts caught (%)\n '); axes[1].set_title('(b) Harm recall', fontsize=8, loc='left')
    h = [plt.Line2D([], [], marker='s', ls='', color=BN, ms=5, label='Bangla, original'), plt.Line2D([], [], marker='^', ls='', color=TR, ms=6, label='after translate-first'),
         plt.Line2D([], [], marker='|', ls='', color=EN, ms=8, mew=1.6, label='English reference')]
    axes[0].legend(handles=h, loc='lower left', bbox_to_anchor=(0, 1.07), ncol=3, frameon=False, fontsize=7)
    save(fig, 'fig6_translate_first')

# ---------- Fig 7: comprehension vs refusal, and judge validity ----------
def fig_judges():
    rows = list(csv.DictReader(open(P / 'results/judge_gold/claude_labels_400.csv', encoding='utf-8-sig')))
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 62 * MM), gridspec_kw={'wspace': 0.55, 'width_ratios': [1, 1.15]})
    ax = axes[0]
    for j, f in enumerate(FORMS):
        s = [r for r in rows if r['form'] == f]; n = len(s); c = collections.Counter(r['claude_label'] for r in s)
        a, d, rr = 100 * c['A'] / n, 100 * c['D'] / n, 100 * (c['P'] + c['R']) / n
        ax.barh(j, a, color=EN, height=0.62, edgecolor='white', lw=1)
        ax.barh(j, d, left=a, color=BN, height=0.62, edgecolor='white', lw=1, hatch='///')
        ax.barh(j, rr, left=a + d, color=INK, height=0.62, edgecolor='white', lw=1)
        if d >= 8: ax.text(a + d / 2, j, f'{d:.0f}%', ha='center', va='center', fontsize=6.6, color='white', fontweight='bold')
        ax.text(101, j, f'n={n}', va='center', fontsize=6.2, color=MUTED)
    ax.set_yticks(range(5)); ax.set_yticklabels([l.replace('-\n', '-').replace('\n', ' ') for l in FORM_LBL], fontsize=7); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel('Share of chat-model answers (%)'); ax.set_title('(a) What small chat models do', fontsize=8, loc='left')
    h = [plt.Rectangle((0, 0), 1, 1, color=EN, label='answered'), plt.Rectangle((0, 0), 1, 1, fc=BN, hatch='///', ec='white', label='did not understand'),
         plt.Rectangle((0, 0), 1, 1, color=INK, label='refused / partly')]
    ax.legend(handles=h, loc='lower left', bbox_to_anchor=(-0.02, 1.07), ncol=3, frameon=False, fontsize=6.3, handlelength=1.2, columnspacing=0.8)
    ax = axes[1]
    J = list(csv.DictReader(open(P / 'results/judge_gold/judge_scores_vs_claude.csv', encoding='utf-8')))
    J.sort(key=lambda r: float(r['says_refusal']))
    y = np.arange(len(J))
    ax.barh(y + 0.19, [100 * int(r['says_refusal']) / 400 for r in J], height=0.36, color=BN, label='called refusal (%)')
    ax.barh(y - 0.19, [100 * float(r['precision']) for r in J], height=0.36, color=EN, hatch='\\\\\\', edgecolor='white', label='precision (%)')
    ax.axvline(100 * 9 / 400, color=INK, lw=1, ls='--'); ax.text(100 * 9 / 400 + 1.2, len(J) - 0.42, 'true refusal rate (2.3%)', fontsize=6.2, va='bottom')
    ax.set_yticks(y); ax.set_yticklabels([r['judge'].replace('Big judge ', '').replace(' (used in P1/P2)', '') for r in J], fontsize=7); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel('% of the 400 sampled answers / precision'); ax.set_title('(b) Refusal judges vs. reference labels', fontsize=8, loc='left')
    ax.legend(loc='upper right', frameon=False, fontsize=6.4)
    for a in axes: a.grid(axis='x', color=GRID, lw=0.5); a.set_axisbelow(True)
    save(fig, 'fig7_judges')

def fig_social():
    rows = list(csv.DictReader(open(P / 'results/final/social.csv', encoding='utf-8')))
    rows = [r for r in rows if r['form'] == 'all']; name = {g: d for g, _, d in GUARDS}
    rows.sort(key=lambda r: float(r['block_pct']))
    fig, ax = plt.subplots(figsize=(FULL * 0.72, 62 * MM))
    y = np.arange(len(rows)); v = [float(r['block_pct']) for r in rows]
    lo = [v[i] - float(r['ci_low']) for i, r in enumerate(rows)]; hi = [float(r['ci_high']) - v[i] for i, r in enumerate(rows)]
    ax.barh(y, v, height=0.62, color=BN, xerr=[lo, hi], error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
    for i, x in enumerate(v): ax.text(x + hi[i] + 1.5, i, f'{x:.0f}%', va='center', fontsize=6.8)
    ax.set_yticks(y); ax.set_yticklabels([name.get(r['guard'], r['guard']) for r in rows], fontsize=7)
    ax.set_xlim(0, 100); ax.set_xlabel('Non-hateful real comments blocked (%), n = 600')
    ax.grid(axis='x', color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, 'fig8_social_comments')

if __name__ == '__main__':
    fig_overview(); fig_heatmap(); fig_recall(); fig_roc(); fig_drift(); fig_translate(); fig_judges(); fig_social()
