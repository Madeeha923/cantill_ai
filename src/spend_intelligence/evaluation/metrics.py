from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def hidden_label_evaluation(texts, labels, taxonomy, test_size=0.2, random_state=42):
    """Hold out Company A labels, fit from the remainder, and report accuracy by level."""
    from ..classification.classifier import TaxonomyClassifier

    paths = labels.fillna("").astype(str).apply(lambda row: " || ".join(v for v in row if v), axis=1)
    train_idx, test_idx = train_test_split(range(len(paths)), test_size=test_size, random_state=random_state,
                                           stratify=paths)
    model = TaxonomyClassifier(taxonomy)
    model.fit(texts.iloc[train_idx], labels.iloc[train_idx])
    predicted = model.predict(texts.iloc[test_idx])
    rows = []
    truth = labels.iloc[test_idx].fillna("").astype(str)
    for level, column in enumerate(labels.columns):
        actual = truth[column].tolist()
        guess = [path[level] if len(path) > level else "" for path, _ in predicted]
        rows.append({"level": column, "accuracy": accuracy_score(actual, guess)})
    return rows
