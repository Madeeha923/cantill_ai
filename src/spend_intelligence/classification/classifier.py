from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import normalize

from .taxonomy_loader import taxonomy_paths


def _path_id(path):
    return " || ".join(path)


class TaxonomyClassifier:
    """Supervised text classifier with an unseen-tree, zero-label similarity fallback."""
    def __init__(self, tree: pd.DataFrame, random_state: int = 42):
        self.tree = tree
        self.level_columns = list(tree.columns)
        self.paths = taxonomy_paths(tree)
        self.random_state = random_state
        self.vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=80000, sublinear_tf=True)),
        ])
        self.model = None
        self.fallback_vectorizer = None
        self.fallback_matrix = None
        self.classes_ = []

    def fit(self, texts, labels: pd.DataFrame | None = None):
        if labels is not None and not labels.empty:
            label_columns = [c for c in self.level_columns if c in labels.columns]
            target = labels[label_columns].fillna("").astype(str).apply(lambda row: tuple(v for v in row if v), axis=1)
            valid = target.map(bool)
            texts = pd.Series(texts).loc[valid].fillna("").astype(str)
            target = target.loc[valid]
            if len(texts) and target.nunique() > 1:
                y = target.map(_path_id)
                self.model = ComplementNB(alpha=0.15, norm=True)
                matrix = self.vectorizer.fit_transform(texts)
                self.model.fit(matrix, y)
                self.classes_ = list(self.model.classes_)
        if self.model is None:
            self._fit_fallback()
        return self

    def _fit_fallback(self):
        paths = self.paths
        descriptions = [" ".join(path) for path in paths]
        self.fallback_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
        self.fallback_matrix = normalize(self.fallback_vectorizer.fit_transform(descriptions))

    def predict(self, texts):
        texts = pd.Series(texts).fillna("").astype(str)
        if self.model is not None:
            matrix = self.vectorizer.transform(texts)
            probabilities = self.model.predict_proba(matrix)
            indices = probabilities.argmax(axis=1)
            rows = []
            for idx, class_idx in enumerate(indices):
                path = tuple(self.model.classes_[class_idx].split(" || "))
                rows.append((path, float(probabilities[idx, class_idx])))
            return rows
        matrix = normalize(self.fallback_vectorizer.transform(texts))
        similarities = (matrix @ self.fallback_matrix.T).toarray()
        rows = []
        for row in similarities:
            idx = int(row.argmax())
            confidence = float(np.clip(row[idx], 0.0, 1.0))
            rows.append((self.paths[idx], confidence))
        return rows
