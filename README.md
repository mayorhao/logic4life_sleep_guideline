# Logic4Life: Sleep Guideline Formalization

Companion repository for the prompts and back-translation evaluation code used in our **AMIA 2026** paper:

**[Ensemble Logic for Symbolic Representation of Sleep Medicine Guidelines](https://arxiv.org/abs/2607.21331)**

Jiahao Fan, Xiaojin Li, Yan Huang, Xubing Hao, Licong Cui, and Guo-Qiang Zhang

*Proceedings of the 2026 American Medical Informatics Association Symposium, to appear.*


## Overview

The study represents sleep-scoring rules from the American Academy of Sleep Medicine (AASM) Manual using **Rational Ensemble Logic (QEL)**. QEL makes physiological conditions, thresholds, durations, exclusions, and temporal relationships explicit.

The back-translation experiment reconstructs clinician-facing English descriptions from these formal specifications and compares them with the original scoring narratives. It covers 12 scoring rules across arousals, respiratory events, stage N2 sleep, and limb movements.

## Prompts and code

The Python package provides a command-line interface and reusable modules for:

- **Generation prompts:** the instructions for translating QEL formulas into English, a controlled terminology map, an example, and the structured input/output format.
- **Back-translation code:** asynchronous generation of repeated translations for each rule, with structured JSON outputs containing the generated text, individual clauses, assumptions, and warnings.
- **Evaluation code:** embedding-based semantic similarity, lexical sequence matching, ROUGE-L, and bootstrap summaries at the rule and overall levels.

For each rule, the generation input contains its identifier, event name, QEL formula, and any supporting macro definitions. The source narrative is reserved for evaluation and is not included in the generation payload. The prompt instructs the model to preserve numeric thresholds, durations, negations, and temporal relationships using the supplied terminology.

## Project structure

```text
config/default.json          Experiment defaults
examples/rules.json          Synthetic example input
src/logic4life_sleep/
    cli.py                   Command-line interface
    config.py                Configuration loading and validation
    pipeline.py              Generation, embeddings, and output files
    metrics.py               Similarity metrics and bootstrap summaries
    prompt.py                Original generation prompt
    rules.py                 Input validation and generation payloads
back_translation.py          Source-checkout entry point
requirements.txt             Pinned runtime dependencies
requirements-dev.txt         Development and build dependencies
pyproject.toml               Package metadata and command-line entry point
tests/                       Offline unit and integration tests
.github/workflows/tests.yml  Automated tests and package builds
CITATION.cff                 Machine-readable paper citation
```

## Installation

Python 3.10 or newer is required. Create a virtual environment and install the project:

```bash
git clone https://github.com/mayorhao/logic4life_sleep_guideline.git
cd logic4life_sleep_guideline
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The package installation reads its runtime dependencies from [requirements.txt](requirements.txt). For use directly from a checkout, installing `python -m pip install -r requirements.txt` is also supported.

## Configuration

[config/default.json](config/default.json) contains the study's model, generation, concurrency, retry, and bootstrap settings. Create a copy to configure another experiment. Settings are applied in this order:

1. Built-in study defaults.
2. Values from the JSON file supplied with `--config`.
3. Explicit command-line overrides, such as `--runs 5`.

Unknown keys and invalid types are rejected before any API call. Input and output paths are supplied on the command line and are relative to the current working directory. Each run writes the resolved settings to `config.json` in its output directory.

Set the API key through your environment before generation:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
```

[.env.example](.env.example) documents the environment variable. The application reads the process environment and does not automatically load `.env` files. Local environment files and generated outputs are excluded by `.gitignore`.

## Usage

Validate the bundled synthetic example without making API calls:

```bash
logic4life-sleep --config config/default.json --input examples/rules.json --validate-only
```

Provide your study rules in `rules.json`, then run generation and evaluation:

```bash
logic4life-sleep --config config/default.json --input rules.json --output-dir results
```

To override a configuration value or inspect the original prompt:

```bash
logic4life-sleep --config config/default.json --input rules.json --runs 5 --output-dir results-small
logic4life-sleep --print-prompt
logic4life-sleep --help
```

The installed module entry point `python -m logic4life_sleep` accepts the same options. The existing `python back_translation.py` entry point remains available from the repository root, including input validation and prompt inspection without third-party dependencies.

By default, the script generates 40 translations for every rule in the input. Choose a new or empty output directory for each run. Generation calls the OpenAI API; reference narratives are sent to the embedding model during evaluation.

## Input format

[examples/rules.json](examples/rules.json) is a synthetic example demonstrating the input format:

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

Each rule requires a unique positive integer `id` and nonempty `name`, `formula`, and `reference_text` strings. Optional fields are `macro_definitions` (a string-to-string map), `supporting_rule_ids` (a list of integers), and `formula_warnings` (a list of strings). Include any referenced macro definitions needed to interpret a formula. Use `--rule-ids 1 2` to evaluate a subset.

## Outputs and evaluation

- `config.json`: resolved experiment settings, including command-line overrides.
- `detailed_results.csv`: one row per rule and run, including `back_translation`, `back_translation_clauses`, the reference text in `gold_text`, and evaluation metrics.
- `summary_by_rule.csv`: means, standard deviations, and confidence intervals for each rule.
- `summary_overall.json`: overall metrics and experiment settings.
- `raw_outputs/`: structured model responses for individual translations.

Lexical similarity uses character-level sequence matching. ROUGE-L uses case-sensitive whitespace tokens. Bootstrap intervals resample the outputs within each rule, and the pooled outputs for the overall summary, with 5,000 resamples, 95% confidence, and seed 42 by default. The seed controls bootstrap resampling; it does not make model generation deterministic.

## Development

Install the development dependencies, run the offline tests, and build a wheel and source distribution:

```bash
python -m pip install -e . -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m build
```

Tests cover configuration precedence, invalid inputs, metric behavior, reference separation, API retries, and generated files. API interactions use mocked responses and require no credentials. GitHub Actions runs the tests and package builds on Python 3.10, 3.11, and 3.12.

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
