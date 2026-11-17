"""Held-out category evaluation for Company A's labelled ERP records."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from main import make_text
from src.spend_intelligence.config import load_config
from src.spend_intelligence.ingestion.loader import load_company_inputs
from src.spend_intelligence.preprocessing.company_a import prepare_company_a
from src.spend_intelligence.preprocessing.classification import TaxonomyClassifier


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs" / "company_a_evaluation"
LABEL_COLUMNS = ["category_l1", "category_l2", "category_l3"]
FEATURE_COLUMNS = [
    "material_short_text",
    "service_short_text",
    "material_number",
    "spend_type",
    "material_type",
]
RANDOM_STATE = 42
TEST_FRACTION = 0.20
INVALID_PREDICTION = "__INVALID_PREDICTION__"


def category_path(row, columns):
    """Create a path from non-empty category values, preserving spelling."""
    values = []
    for column in columns:
        value = row[column]
        if pd.notna(value) and str(value).strip():
            values.append(str(value).strip())
    return "|||".join(values)


def split_indices(labels):
    """Return deterministic train/test positions, keeping singletons in train."""
    labels = pd.Series(labels).reset_index(drop=True)
    counts = labels.value_counts()
    singleton_positions = set(
        labels.index[labels.isin(counts[counts == 1].index)].tolist()
    )
    eligible_positions = np.array(
        [position for position in labels.index if position not in singleton_positions],
        dtype=int,
    )

    if len(eligible_positions) < 2:
        raise ValueError(
            "Not enough labelled records with repeat categories for a held-out split."
        )

    eligible_labels = labels.iloc[eligible_positions]
    eligible_class_count = eligible_labels.nunique()
    test_count = max(1, math.ceil(TEST_FRACTION * len(eligible_positions)))
    can_stratify = (
        eligible_labels.value_counts().min() >= 2
        and test_count >= eligible_class_count
        and len(eligible_positions) - test_count >= eligible_class_count
    )

    if can_stratify:
        from sklearn.model_selection import train_test_split

        train_eligible, test_positions = train_test_split(
            eligible_positions,
            test_size=test_count,
            random_state=RANDOM_STATE,
            stratify=eligible_labels.to_numpy(),
        )
        method = "stratified 80/20; singleton categories kept in training"
        train_positions = np.concatenate(
            [np.asarray(train_eligible, dtype=int), np.array(sorted(singleton_positions), dtype=int)]
        )
    else:
        # Small or highly fragmented class distributions may not satisfy
        # sklearn's stratification size constraints. Split each repeat class
        # separately, always retaining at least one training example.
        rng = np.random.RandomState(RANDOM_STATE)
        train_parts = [np.array(sorted(singleton_positions), dtype=int)]
        test_parts = []
        for _, group in eligible_labels.groupby(eligible_labels, sort=False):
            positions = group.index.to_numpy()
            rng.shuffle(positions)
            group_test_count = min(
                len(positions) - 1,
                max(1, round(TEST_FRACTION * len(positions))),
            )
            test_parts.append(positions[:group_test_count])
            train_parts.append(positions[group_test_count:])
        train_positions = np.concatenate(train_parts)
        test_positions = np.concatenate(test_parts)
        method = (
            "per-category fallback; singleton categories kept in training "
            "and each repeat category represented in both partitions"
        )

    return (
        np.sort(np.asarray(train_positions, dtype=int)),
        np.sort(np.asarray(test_positions, dtype=int)),
        method,
        len(singleton_positions),
    )


def save_confusion_matrix(matrix, display_labels, category_labels):
    matrix_df = pd.DataFrame(
        matrix,
        index=display_labels,
        columns=display_labels,
    )
    matrix_df.index.name = "actual_category_path"
    matrix_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    index_df = pd.DataFrame(
        {
            "plot_label": [f"C{i + 1:02d}" for i in range(len(category_labels))]
            + (["INVALID"] if len(display_labels) > len(category_labels) else []),
            "category_path": category_labels
            + ([INVALID_PREDICTION] if len(display_labels) > len(category_labels) else []),
        }
    )
    index_df.to_csv(OUTPUT_DIR / "category_index.csv", index=False)

    tick_labels = index_df["plot_label"].tolist()
    fig, ax = plt.subplots(figsize=(max(10, len(tick_labels) * 0.28), 9))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", aspect="auto")
    ax.set_title("Company A held-out category confusion matrix")
    ax.set_xlabel("Predicted category (see category_index.csv)")
    ax.set_ylabel("Actual category (see category_index.csv)")
    ax.set_xticks(np.arange(len(tick_labels)), labels=tick_labels, rotation=90)
    ax.set_yticks(np.arange(len(tick_labels)), labels=tick_labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Record count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(ROOT / "configs" / "company_a.yaml")
    inputs = load_company_inputs(config, project_root=ROOT)
    prepared = prepare_company_a(inputs, config)

    erp = prepared["erp"].copy()
    erp["record_id"] = "ERP:" + erp["line_id"].astype(str)
    labels = inputs["labels"].copy()

    # Match the production Company A training join and feature construction.
    labelled = erp.merge(labels, on="record_id", how="inner")
    labelled["classification_text"] = make_text(labelled, FEATURE_COLUMNS)

    taxonomy = inputs["taxonomy"].copy()
    taxonomy_columns = [column for column in LABEL_COLUMNS if column in taxonomy]
    if taxonomy_columns != LABEL_COLUMNS:
        raise ValueError(
            f"Company A taxonomy must contain {LABEL_COLUMNS}; found {taxonomy_columns}."
        )

    taxonomy = taxonomy[LABEL_COLUMNS].fillna("").copy()
    taxonomy["category_path"] = taxonomy.apply(
        lambda row: category_path(row, LABEL_COLUMNS), axis=1
    )
    taxonomy_paths = list(dict.fromkeys(
        path for path in taxonomy["category_path"] if path
    ))
    taxonomy_path_set = set(taxonomy_paths)
    if not taxonomy_path_set:
        raise ValueError("Company A taxonomy has no complete category paths.")

    label_complete = labelled[LABEL_COLUMNS].notna().all(axis=1)
    for column in LABEL_COLUMNS:
        label_complete &= labelled[column].astype("string").str.strip().ne("")

    labelled["actual_path"] = labelled.apply(
        lambda row: category_path(row, LABEL_COLUMNS), axis=1
    )
    valid_taxonomy_label = labelled["actual_path"].isin(taxonomy_path_set)
    missing_label_count = int((~label_complete).sum())
    invalid_gold_label_count = int((label_complete & ~valid_taxonomy_label).sum())

    evaluation_frame = labelled.loc[label_complete & valid_taxonomy_label].copy()
    blank_feature_mask = evaluation_frame["classification_text"].str.strip().eq("")
    blank_feature_count = int(blank_feature_mask.sum())
    evaluation_frame = evaluation_frame.loc[~blank_feature_mask].reset_index(drop=True)

    if evaluation_frame.empty:
        raise ValueError("No complete, taxonomy-valid, non-empty labelled records to evaluate.")

    train_pos, test_pos, split_method, singleton_count = split_indices(
        evaluation_frame["actual_path"]
    )
    train = evaluation_frame.iloc[train_pos].copy()
    test = evaluation_frame.iloc[test_pos].copy()

    # TF-IDF is fitted only inside fit(), and fit() sees training records only.
    classifier = TaxonomyClassifier()
    classifier.fit(train, ["classification_text"], LABEL_COLUMNS)
    test_records = test[["record_id", "classification_text"]].copy()
    predicted = classifier.predict(
        test_records,
        ["classification_text"],
        taxonomy[LABEL_COLUMNS],
    )

    predicted_paths = predicted.apply(
        lambda row: category_path(row, ["l1", "l2", "l3"]), axis=1
    )
    invalid_prediction_mask = ~predicted_paths.isin(taxonomy_path_set)
    invalid_prediction_count = int(invalid_prediction_mask.sum())
    y_true = test["actual_path"].tolist()
    y_pred = predicted_paths.where(~invalid_prediction_mask, INVALID_PREDICTION).tolist()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_rows = {
        name: values if isinstance(values, dict) else {"accuracy": values}
        for name, values in report.items()
    }
    report_df = pd.DataFrame.from_dict(report_rows, orient="index")
    report_df.index.name = "category"
    report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=taxonomy_paths,
        zero_division=0,
    )
    per_category = pd.DataFrame(
        {
            "category_path": taxonomy_paths,
            "precision": per_precision,
            "recall": per_recall,
            "f1_score": per_f1,
            "test_support": per_support,
            "training_support": [int((train["actual_path"] == path).sum()) for path in taxonomy_paths],
        }
    )
    per_category.to_csv(OUTPUT_DIR / "per_category_metrics.csv", index=False)

    test_categories_without_support = int((per_category["test_support"] == 0).sum())
    low_test_support_categories = int(
        ((per_category["test_support"] > 0) & (per_category["test_support"] < 10)).sum()
    )
    low_training_support_categories = int((per_category["training_support"] < 10).sum())
    limitation_notes = []
    if test_categories_without_support:
        limitation_notes.append(
            f"{test_categories_without_support} taxonomy categories have no held-out "
            "test examples, so their recall cannot be estimated."
        )
    if low_test_support_categories:
        limitation_notes.append(
            f"{low_test_support_categories} represented categories have fewer than "
            "10 test examples; per-category metrics for them are noisy."
        )
    if low_training_support_categories:
        limitation_notes.append(
            f"{low_training_support_categories} categories have fewer than 10 training "
            "examples, limiting what the classifier can learn for those categories."
        )
    if not limitation_notes:
        limitation_notes.append("No taxonomy category has low train or test support under these thresholds.")
    (OUTPUT_DIR / "evaluation_notes.txt").write_text(
        "Held-out evaluation uses the fixed 80/20 split with random_state=42.\n"
        "TF-IDF and LinearSVC are fitted on the training partition only.\n"
        "Macro metrics use the categories present in the union of held-out truth and predictions.\n"
        + "\n".join(f"- {note}" for note in limitation_notes)
        + "\n",
        encoding="utf-8",
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1_score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "training_records": len(train),
        "test_records": len(test),
        "taxonomy_categories_represented_in_training": int(train["actual_path"].nunique()),
        "taxonomy_categories_represented_in_test": int(test["actual_path"].nunique()),
        "taxonomy_categories_without_test_support": test_categories_without_support,
        "test_categories_with_fewer_than_10_records": low_test_support_categories,
        "training_categories_with_fewer_than_10_records": low_training_support_categories,
        "missing_label_records_excluded": missing_label_count,
        "invalid_gold_paths_excluded": invalid_gold_label_count,
        "blank_feature_records_excluded": blank_feature_count,
        "singleton_categories_kept_in_training": singleton_count,
        "invalid_predictions": invalid_prediction_count,
        "taxonomy_category_count": len(taxonomy_path_set),
        "random_state": RANDOM_STATE,
        "test_fraction": TEST_FRACTION,
        "split_method": split_method,
    }
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    confusion_labels = taxonomy_paths + (
        [INVALID_PREDICTION] if invalid_prediction_count else []
    )
    matrix = confusion_matrix(y_true, y_pred, labels=confusion_labels)
    save_confusion_matrix(matrix, confusion_labels, taxonomy_paths)

    print("Company A held-out category evaluation complete")
    print(f"Split: {split_method}")
    print(f"Training records: {len(train):,}")
    print(f"Test records: {len(test):,}")
    print(f"Missing labels excluded: {missing_label_count:,}")
    print(f"Invalid gold paths excluded: {invalid_gold_label_count:,}")
    print(f"Blank feature records excluded: {blank_feature_count:,}")
    print(f"Invalid predictions: {invalid_prediction_count:,}")
    print(f"Taxonomy categories without test support: {test_categories_without_support}")
    print(f"Test categories with fewer than 10 records: {low_test_support_categories}")
    print(f"Training categories with fewer than 10 records: {low_training_support_categories}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['macro_precision']:.4f}")
    print(f"Macro recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1-score: {metrics['macro_f1_score']:.4f}")
    print(f"Weighted F1-score: {metrics['weighted_f1_score']:.4f}")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
