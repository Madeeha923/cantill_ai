# Cantill Spend Intelligence

## About the Project

The goal was to create a pipeline that can take raw purchase records from different company systems, clean and process the data, identify duplicate transactions, and classify each purchase into the correct spending category.

I worked on three scenarios:

- **Company A:** Combined data from an ERP system and an invoice tool. The main focus was handling VAT, identifying duplicate purchases, and using supplier and purchase information to match records across both sources.
- **Company B:** Processed purchase records from multiple regions and currencies. This involved converting local amounts into USD using the available exchange rates and applying rules to identify non-procurement transactions.
- **Company C:** Built a reusable classification approach for a new company where labelled training data is not available. It uses the company's taxonomy to find the closest matching category.

The pipeline is designed to run locally, so the raw assignment data does not need to be uploaded to GitHub.

## How I Approached the Problem

I divided the work into a few main steps:

1. **Data loading and cleaning:** Read the input files, standardised column names, handled missing values, and converted amounts and dates into usable formats.

2. **Spend processing:** Applied company-specific rules to calculate spend correctly. For example, Company A's invoice tool amounts are converted from gross to net by removing VAT, while Company B's local currency amounts are converted into USD.

3. **Duplicate detection:** For Company A, ERP records are treated as the primary records. Invoice tool entries are checked against ERP records using purchase order details, supplier information, amounts, and descriptions.

4. **Spend inclusion:** Applied the available business rules to decide whether a transaction should be included in spend or excluded with a reason.

5. **Category classification:** Used the available labelled records to train a TF-IDF-based classifier for Companies A and B. For a company without labelled data, the pipeline compares transaction descriptions with the available taxonomy categories.

6. **Output generation:** Generated the final CSV with the required fields, including inclusion status, supplier and item keys, spend amount, category hierarchy, and confidence score.

## Project Structure

```text
cantill-spend-intelligence/
│
├── configs/                 # Company-specific configuration files
├── data/                    # Local input data (not committed to Git)
├── src/
│   └── spend_intelligence/  # Data processing and classification logic
├── tests/                   # Automated tests
├── outputs/                 # Generated results (not committed to Git)
├── submission/              # Assignment output files (not committed to Git)
│
├── main.py                  # Main entry point
├── evaluate_company_a.py    # Company A model evaluation
├── validateoutput.py        # Output validation
├── requirements.txt
├── APPROACH.md
├── DECISIONS.md
└── README.md
```

## Getting Started

You will need Python 3.10 or newer.

Clone the repository and install the required packages:

```bash
git clone https://github.com/Madeeha923/cantill_ai.git
cd cantill_ai

python -m venv .venv
```

Activate the environment:

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Pipeline

Place the provided input files in their respective folders under `data/`. The filenames and column mappings are defined in the YAML configuration files inside `configs/`.

Run the pipelines from the project root:

**Company A**
```bash
python main.py --company A
```

**Company B**
```bash
python main.py --company B
```

The generated results will be saved in the `outputs/` folder.

The final output follows this structure:

```text
record_id, include, exclusion_reason, supplier_key, item_key,
spend, l1, l2, l3, l4, confidence
```

Each input record is represented once. Excluded records have zero spend, and category values are taken from the supplied taxonomy.

## Model Evaluation and Validation

I also included separate scripts to check the results and evaluate the classification approach.

**Evaluate Company A:**

```bash
python evaluate_company_a.py
```

This evaluates the classifier using an 80/20 train-test split. The model is fitted only on the training portion, and the evaluation metrics and confusion matrix are saved under `outputs/company_a_evaluation/`.

The current held-out results are:

| Metric | Score |
|---|---:|
| Accuracy | 90.95% |
| Macro Precision | 88.03% |
| Macro Recall | 87.15% |
| Macro F1-score | 87.00% |
| Weighted F1-score | 91.00% |

**Validate generated outputs:**

```bash
python validateoutput.py
```

**Run automated tests:**

```bash
python -m pytest -q
```

The tests cover important cases such as duplicate handling, currency conversion, output structure, taxonomy matching, and the generic classification pipeline.

## Handling a New Company Without Training Labels

For Company C, I included a generic pipeline that can be configured once the input files and taxonomy are available.

The configuration template is:

```text
configs/company_c.yaml
```

It allows the user to specify the transaction ID, amount, currency, description fields, optional supplier and item fields, and taxonomy columns without changing the main pipeline code.

Since labelled examples are not available for this scenario, the classification fallback uses TF-IDF and cosine similarity to compare transaction descriptions against complete taxonomy paths. It supports both three-level and four-level taxonomies.

The confidence value in this approach represents a similarity score, not a calibrated probability. A weak or generic description may therefore receive a low-confidence category match.

The generic pipeline can be tested using the included sample configuration:

```bash
python main.py --config tests/fixtures/company_c_three_level/config.yaml
```

This runs a synthetic example to check that the pipeline works. It should not be treated as an evaluation of actual Company C classification accuracy.

## Additional Notes

- Raw input data and generated output files are excluded from Git.
- Company-specific processing rules and assumptions are documented in `DECISIONS.md`.
- The overall design, classification approach, and evaluation details are described in `APPROACH.md`.
- The pipeline does not require a paid LLM API; the current implementation uses local Python libraries.

