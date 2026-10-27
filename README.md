# Logic4Life: Sleep Guideline Formalization

Companion repository for the prompts and back-translation evaluation code used in our **AMIA 2026** paper:

**[Ensemble Logic for Symbolic Representation of Sleep Medicine Guidelines](https://arxiv.org/abs/2607.21331)**

Jiahao Fan, Xiaojin Li, Yan Huang, Xubing Hao, Licong Cui, and Guo-Qiang Zhang

*Proceedings of the 2026 American Medical Informatics Association Symposium, to appear.*


## Overview

The study represents sleep-scoring rules from the American Academy of Sleep Medicine (AASM) Manual using **Rational Ensemble Logic (QEL)**. QEL makes physiological conditions, thresholds, durations, exclusions, and temporal relationships explicit.

The back-translation experiment reconstructs clinician-facing English descriptions from these formal specifications and compares them with the original scoring narratives. It covers 12 scoring rules across arousals, respiratory events, stage N2 sleep, and limb movements.

## Prompts and code

The implementation is available as a single Python file: [back_translation.py](back_translation.py). It includes:

- **Generation prompts:** the instructions for translating QEL formulas into English, a controlled terminology map, an example, and the structured input/output format.
- **Back-translation code:** asynchronous generation of repeated translations for each rule, with structured JSON outputs containing the generated text, individual clauses, assumptions, and warnings.
- **Evaluation code:** embedding-based semantic similarity, lexical sequence matching, ROUGE-L, and bootstrap summaries at the rule and overall levels.

For each rule, the generation input contains its identifier, event name, QEL formula, and any supporting macro definitions. The source narrative is reserved for evaluation and is not included in the generation payload. The prompt instructs the model to preserve numeric thresholds, durations, negations, and temporal relationships using the supplied terminology.

## Usage

Use Python 3.10 or newer. Install the generation dependencies and set `OPENAI_API_KEY` in your environment:

```bash
python -m pip install openai "pydantic>=2"
```

Provide the formulas and reference narratives in a JSON file. The following `rules.json` is a synthetic example demonstrating the input format:

```json
{
  "rules": [
    {
      "id": 1,
      "name": "Example sleep event",
      "formula": "\\exists x\\, (\\Box_{10}\\,\\mathrm{Sleep})_x",
      "reference_text": "Score Example sleep event if the subject is asleep continuously for at least 10 seconds.",
      "macro_definitions": {}
    }
  ]
}
```

Each rule requires a unique positive integer `id` and nonempty `name`, `formula`, and `reference_text` strings. Optional fields are `macro_definitions` (a string-to-string map), `supporting_rule_ids` (a list of integers), and `formula_warnings` (a list of strings). Include any referenced macro definitions needed to interpret a formula.

Validate the input, inspect the embedded prompt, or run generation and evaluation:

```bash
python back_translation.py --input rules.json --validate-only
python back_translation.py --print-prompt
python back_translation.py --input rules.json --output-dir results
```

Validation and prompt inspection work without an API key or third-party packages. Generation calls the OpenAI API; reference narratives are sent to the embedding model during evaluation. By default, the script generates 40 translations for every rule in the input. Use `--rule-ids`, `--runs`, `--concurrency`, and the model options to customize a run; `--help` lists all options. Choose a new or empty output directory for each run.

Outputs are:

- `detailed_results.csv`: one row per rule and run, including `back_translation`, `back_translation_clauses`, the reference text in `gold_text`, and evaluation metrics.
- `summary_by_rule.csv`: means, standard deviations, and confidence intervals for each rule.
- `summary_overall.json`: overall metrics and experiment settings.
- `raw_outputs/`: structured model responses for individual translations.

Lexical similarity uses character-level sequence matching. ROUGE-L uses case-sensitive whitespace tokens. Bootstrap intervals resample the 40 outputs within each rule, and the pooled outputs for the overall summary, with 5,000 resamples, 95% confidence, and seed 42 by default. The seed controls bootstrap resampling; it does not make model generation deterministic.

## Experiment configuration

The script defaults match the model and sampling configuration used in the study.

| Setting | Value |
| --- | --- |
| Generation model | `gpt-5-mini-2025-08-07` |
| Embedding model | `text-embedding-3-small` |
| Scoring rules | 12 |
| Generations per rule | 40 |
| Total back-translations | 480 |


## Back-translation results

Overall results

| Metric | Mean | 95% confidence interval |
| --- | ---: | ---: |
| Embedding cosine similarity | 79.33 | 78.99–79.67 |
| Lexical sequence matching | 14.93 | 13.80–16.06 |
| ROUGE-L F1 | 18.28 | 17.60–18.95 |

The evaluation measures similarity between reconstructed and reference descriptions. The higher semantic similarity alongside lower lexical overlap indicates that the generated descriptions often use different wording from the source narratives.

## Citation

If you use this work, please cite the paper:

```bibtex
@article{fan2026ensemble,
  title         = {Ensemble Logic for Symbolic Representation of Sleep Medicine Guidelines},
  author        = {Fan, Jiahao and Li, Xiaojin and Huang, Yan and Hao, Xubing and Cui, Licong and Zhang, Guo-Qiang},
  journal       = {arXiv preprint arXiv:2607.21331},
  year          = {2026},
  eprint        = {2607.21331},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LO},
  doi           = {10.48550/arXiv.2607.21331},
  url           = {https://arxiv.org/abs/2607.21331}
}
```
