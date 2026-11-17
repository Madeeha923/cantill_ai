# Decisions and assumptions

## Amounts, currencies, and missing values

- Company A ERP values are already net ZAR. Invoice Tool values are gross and are divided by `1 + VAT`; a blank or invalid VAT value defaults to 15%.
- Company B USD rows pass through unchanged even if `currency_rate` is non-1. Non-USD rows use a positive transaction rate first, then the invoice-month `units_per_usd` rate; spend is local amount divided by the selected rate.
- Company B has five rows with missing non-USD FX and three rows with missing amounts. Preprocessing preserves `fx_missing`, `amount_missing`, and `spend_missing`; `outputs/company_b_missing_spend.csv` lists their IDs and reasons. The required output schema has numeric spend and uses zero for excluded rows; for included missing-spend rows the output builder also writes zero, so the sidecar is needed to distinguish missing values from genuine zero spend. No new exclusion reason was invented.
- Unparseable dates remain attached to their rows. Negative amounts are retained as credits or reversals.

## Inclusion and matching

- Company A ERP receipts remain canonical. Invoice Tool `Approved` and `Paid` rows are eligible; cancelled rows receive `CANCELLED`; all other statuses receive `NOT_APPROVED`. Status reasons take precedence over duplicate reasons.
- A tool row receives `DUPLICATE` only when it is eligible and matched to ERP or marked as a repeated export. Matching uses normalized PO, PO item, supplier, amount (the existing tolerance of max(1 ZAR, 1% of the larger amount)), and description evidence. With multiple ERP candidates, a unique best description match must also satisfy the amount check; ties remain included and are reported ambiguous. ERP receipt rows are never removed. These are heuristic matches, not ground-truth findings.
- Company B's input has no approval-status column. `paid_date` is present but is not used to infer approval or payment; document type is not used as approval evidence. Therefore the current B inclusion output does not independently establish the brief's Approved/Paid status rule for every invoice.
- Company B inclusion currently excludes intercompany rows when the GL description starts with an I/C marker and rows whose line description contains the whole word `tax` or `taxes`. All other rows are included by default. Seed-label agreement is reported in the final validation output; a seed match does not establish status for unlabeled rows.

## Identity and classification

- Company A supplied supplier/item keys seed ERP identities. Company B uses vendor and material identifiers, with normalized text fallbacks. When the generic config has no supplier/item fields or values, output uses `supplier-unknown` or `item-unknown`.
- Company A and B use supervised classification from their labels. The Company A hidden-label evaluation fits TF-IDF and its classifier only on the training split.
- The generic no-label pipeline reads taxonomy columns at runtime and compares each full path with configured line text using TF-IDF and cosine similarity. It supports three or four levels and leaves lower output levels blank for shorter paths. Similarity is bounded to [0,1] but is not a calibrated probability; the fallback always chooses a closest path, even for an empty or uninformative description.
- `configs/company_c.yaml` is a schema-neutral template. Actual paths and column mappings must be filled from the blind files when supplied. Synthetic fixture results verify pipeline mechanics only and are not Company C accuracy evidence.
