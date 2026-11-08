# BanglaOR: auditing open safety guards on Bengali script, Kolkata speech, code-mixed and romanized Bangla

This repository holds the data, code and model outputs for the paper

> Shaon Bandyopadhyay. *Blocked in Bangla, Blind in Bengali: Auditing Open Safety Guards across Script,
> Colloquial, Code-Mixed and Romanized Bangla.* Manuscript submitted to *Social Network Analysis and Mining* (Springer).

**Content warning:** some files contain model outputs for harmful prompts taken from the public MultiJail
benchmark. They are released only so that the results can be checked.

## What BanglaOR is
- **Trigger set:** 100 harmless questions that contain an alarming word or idiom (kill, sacrifice, bomb, poison…).
  Each question is written in five forms: English, Bengali script, Kolkata spoken Bangla, code-mixed Bangla–English
  and Banglish (romanized). See `data/trigger_bangla_4forms.csv` and `data/trigger_english_full.csv`.
- **Neutral set:** 50 harmless questions with no alarming word, in the same five forms (`data/neutral_5forms.csv`).
- **Harmful set:** the 315 English and Bengali prompts of MultiJail (Deng et al., ICLR 2024, MIT licence). They
  are **not redistributed here**. The notebooks load them from Hugging Face (`DAMO-NLP-SG/MultiJail`), and the
  romanized versions are rebuilt with the rule-based transliteration in `code/make_fix_notebook.py`.
- **Translations:** Qwen2.5-7B-Instruct English translations of all non-harmful Bangla texts
  (`data/translations_qwen25_7b.csv`). Translations of harmful prompts are left out.
- **Real comments:** 600 comments labelled non-hateful, sampled from the Bengali hate-speech corpus of
  Romim et al. (2021; Kaggle `naurosromim/bengali-hate-speech-dataset`). The texts are **not redistributed**.
  `code/make_social_notebook.py` reproduces the sample (random seed 0), and `results/raw/social_guard.csv` holds
  the guard verdicts by sample id.

All Bangla forms were written by the author, a native speaker from West Bengal.

## Layout
| Folder | Contents |
|---|---|
| `data/` | benchmark texts and translations |
| `rules/` | analysis rules written before the confirmation and full studies (internal remarks removed; wording otherwise unchanged) |
| `notebooks/` | the Kaggle notebooks used for every stage (pilot, P2, confirmation, full study, reviewer-fix run, real comments, light filters and judges) |
| `code/` | the scripts that build the notebooks, the merged analysis (`analyse_fix.py`) and the figures (`plots/make_figures.py`) |
| `results/raw/` | per-text guard verdicts and P(unsafe) for all 10 guards (`full_guard.csv`, `fix_guard.csv`, `social_guard.csv`) |
| `results/final/` | every number in the paper: summaries with Wilson CIs, McNemar + Holm tests, ROC/threshold results, translate-first, version drift, romanized harm, real comments, Qwen3Guard strict reading |
| `results/pilots/` | pilot (P1), retry (P2) and confirmation (C1) outputs |
| `results/light_filters/`, `results/judges/`, `results/translation_check/` | light toxicity filters, refusal-judge study and translation-quality ratings |

The scripts were written for the working layout of the project, so some file paths inside them refer to the
original folder names (for example `results/judge_gold/`); the mapping to this repository is shown in the table
above.

## Guards audited
Granite Guardian 3.1-2B, 3.2-5B, 3.3-8B and 4.1-8B; Qwen3Guard-Gen 0.6B, 4B and 8B; PolyGuard-Qwen-Smol and
PolyGuard-Qwen; L3Cube IndicGuard. All are openly downloadable from Hugging Face; exact identifiers and settings are
in the paper's appendix and in `code/make_full_notebook.py`.

## Labels made with AI assistance
The reference labels of the 400 chat-model answers (`results/judges/reference_labels_400.csv`) and the 100
translation ratings (`results/translation_check/translation_ratings_100.csv`) were produced with the help of an AI
assistant (Claude, Anthropic) and checked by the author. They are a single, AI-assisted reference, not a
multi-annotator human gold standard; independent relabelling is welcome.

## Licences
- Code: MIT (see `LICENSE`).
- BanglaOR texts, translations, labels and model outputs: CC BY 4.0 (see `DATA_LICENSE.md`).
- MultiJail and the Romim et al. comment corpus keep their own licences and are not included.

## Citation
See `CITATION.cff`. Please also cite MultiJail (Deng et al., 2024) and Romim et al. (2021) if you use those parts.
