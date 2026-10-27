# Logic4Life: Sleep Guideline Formalization

Companion repository for the prompts and back-translation evaluation code used in our **AMIA 2026** paper:

**[Ensemble Logic for Symbolic Representation of Sleep Medicine Guidelines](https://arxiv.org/abs/2607.21331)**

Jiahao Fan, Xiaojin Li, Yan Huang, Xubing Hao, Licong Cui, and Guo-Qiang Zhang

*Proceedings of the 2026 American Medical Informatics Association Symposium, to appear.*

**Release status:** This initial release contains the README. The prompt and code files will be added in a subsequent update.

## Overview

The study represents sleep-scoring rules from the American Academy of Sleep Medicine (AASM) Manual using **Rational Ensemble Logic (QEL)**. QEL makes physiological conditions, thresholds, durations, exclusions, and temporal relationships explicit.

The back-translation experiment reconstructs clinician-facing English descriptions from these formal specifications and compares them with the original scoring narratives. It covers 12 scoring rules across arousals, respiratory events, stage N2 sleep, and limb movements.

## Prompts and code

The materials to be released include:

- **Generation prompts:** the instructions for translating QEL formulas into English, a controlled terminology map, an example, and the structured input/output format.
- **Back-translation code:** asynchronous generation of repeated translations for each rule, with structured JSON outputs containing the generated text, individual clauses, assumptions, and warnings.
- **Evaluation code:** embedding-based semantic similarity, lexical sequence matching, ROUGE-L, and bootstrap summaries at the rule and overall levels.

For each rule, the generation input contains its identifier, event name, QEL formula, and any supporting macro definitions. The source narrative is reserved for evaluation and is not included in the generation payload. The prompt instructs the model to preserve numeric thresholds, durations, negations, and temporal relationships using the supplied terminology.

## Experiment configuration

The final pinned experiment is identified as `v99_pinned` in the research outputs.

| Setting | Value |
| --- | --- |
| Generation model | `gpt-5-mini-2025-08-07` |
| Embedding model | `text-embedding-3-small` |
| Scoring rules | 12 |
| Generations per rule | 40 |
| Total back-translations | 480 |
| Bootstrap resamples | 5,000 |
| Confidence interval | 95% |
| Bootstrap seed | 42 |

## Back-translation results

Overall results from `v99_pinned` are shown below. Similarity scores are reported on a 0–100 scale.

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
