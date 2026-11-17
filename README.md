# Cantill Spend Intelligence

A local, config-driven spend pipeline for Companies A and B, with a generic no-label path for a new company's taxonomy. Raw task data belongs under `data/` and is excluded from Git.

## Install

Use Python 3.10 or newer from the repository root:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the provided inputs under `data/company_a/` and `data/company_b/` with the filenames in their YAML configs.

## Run

```bash
python main.py --company A
python main.py --company B
python evaluate_company_a.py
python validateoutput.py
python -m pytest -q
```

The production runs write `outputs/company_a_output.csv` and `outputs/company_b_output.csv`. The output schema is exactly:

```text
record_id,include,exclusion_reason,supplier_key,item_key,spend,l1,l2,l3,l4,confidence
```

The assignment-named initial deliverables are staged separately as `submission/Madeeha__A.csv` and `submission/Madeeha__B.csv` after the production outputs are regenerated.

## Company A evaluation

`python evaluate_company_a.py` holds out 20% of the labeled ERP records with `random_state=42`, fits TF-IDF and the classifier on training records only, and writes metrics and a confusion matrix under `outputs/company_a_evaluation/`.

## Generic no-label configuration

`configs/company_c.yaml` is a schema-neutral template until the blind files arrive. Replace its file paths and generic field mappings only after inspecting those files. The generic configuration maps the transaction ID, amount, currency, optional date and supplier/item fields, text fields, and taxonomy hierarchy columns. Currency must be the base currency or have a configured transaction rate or monthly FX mapping.

To exercise the generic path before then, use only the synthetic fixture:

```bash
python main.py --config tests/fixtures/company_c_three_level/config.yaml
```

The no-label fallback builds TF-IDF vectors from complete taxonomy paths and selects the best cosine-similarity match. It supports three- and four-level trees, returns only paths from the supplied taxonomy, and leaves unused lower levels blank. Its confidence is a similarity score, not a calibrated probability. Generic or empty descriptions can receive a weak match; synthetic results are not evidence of Company C accuracy.
