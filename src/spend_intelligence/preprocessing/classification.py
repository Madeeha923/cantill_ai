
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity


class TaxonomyClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            min_df=1,
            max_features=100000,
            sublinear_tf=True
        )
        self.model = LinearSVC()
        self.path_map = {}
        self.is_fitted = False

    @staticmethod
    def _clean_text(series):
        return series.fillna("").astype(str).str.strip()

    @staticmethod
    def _make_path(row, columns):
        values = [
            str(row[col]).strip()
            for col in columns
            if pd.notna(row[col]) and str(row[col]).strip()
        ]
        return "|||".join(values)

    def fit(self, df, text_columns, label_columns):
        data = df.copy()

        data["training_text"] = (
            data[text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.strip()
        )

        data["category_path"] = data.apply(
            lambda row: self._make_path(row, label_columns),
            axis=1
        )

        data = data[
            (data["training_text"] != "")
            & (data["category_path"] != "")
        ]

        if data.empty:
            raise ValueError("No valid labelled training records found.")

        self.path_map = {
            path: path.split("|||")
            for path in data["category_path"].unique()
        }

        X = self.vectorizer.fit_transform(data["training_text"])
        y = data["category_path"]

        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(self, df, text_columns, taxonomy):
        if not self.is_fitted:
            raise RuntimeError("Train the classifier before prediction.")

        result = df.copy()

        text = (
            result[text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.strip()
        )

        X = self.vectorizer.transform(text)
        predicted_paths = self.model.predict(X)

        scores = self.model.decision_function(X)

        if scores.ndim == 1:
            confidence = np.ones(len(result))
        else:
            sorted_scores = np.sort(scores, axis=1)
            margin = sorted_scores[:, -1] - sorted_scores[:, -2]
            confidence = 1 / (1 + np.exp(-margin))

        taxonomy_columns = [
            col for col in taxonomy.columns
            if col.lower() in {
                "category_l1", "category_l2",
                "category_l3", "category_l4"
            }
        ]

        valid_paths = set(
            taxonomy.apply(
                lambda row: self._make_path(row, taxonomy_columns),
                axis=1
            )
        )

        for index, path in enumerate(predicted_paths):
            if path not in valid_paths:
                predicted_paths[index] = ""

        category_values = [
            self.path_map.get(path, [])
            for path in predicted_paths
        ]

        for level in range(4):
            result[f"l{level + 1}"] = [
                values[level] if len(values) > level else ""
                for values in category_values
            ]

        result["confidence"] = np.where(
            predicted_paths != "",
            confidence,
            0.0
        )

        return result

    @staticmethod
    def taxonomy_fallback(df, text_columns, taxonomy, taxonomy_columns=None):
        """Classify without labels using full-path TF-IDF cosine similarity.

        The returned confidence is a similarity score in [0, 1], not a
        calibrated probability. ``taxonomy_columns`` allows arbitrary source
        hierarchy column names while output remains l1 through l4.
        """
        result = df.copy()

        if taxonomy_columns is None:
            taxonomy_columns = [
                col for col in taxonomy.columns
                if col.lower() in {
                    "category_l1", "category_l2",
                    "category_l3", "category_l4"
                }
            ]
        if not 3 <= len(taxonomy_columns) <= 4:
            raise ValueError("Taxonomy fallback requires three or four hierarchy columns.")
        missing_taxonomy_columns = set(taxonomy_columns) - set(taxonomy.columns)
        if missing_taxonomy_columns:
            raise ValueError(
                f"Taxonomy is missing configured columns: {sorted(missing_taxonomy_columns)}"
            )
        missing_text_columns = set(text_columns) - set(result.columns)
        if missing_text_columns:
            raise ValueError(
                f"Records are missing configured text columns: {sorted(missing_text_columns)}"
            )

        paths = []
        for _, row in taxonomy.iterrows():
            values = []
            blank_seen = False
            for column in taxonomy_columns:
                value = "" if pd.isna(row[column]) else str(row[column]).strip()
                if not value:
                    blank_seen = True
                elif blank_seen:
                    raise ValueError(
                        "Taxonomy hierarchy has a populated level below a blank level."
                    )
                else:
                    values.append(value)
            if values:
                paths.append(values)

        if not paths:
            raise ValueError("Taxonomy contains no usable category paths.")

        path_text = [" ".join(path) for path in paths]

        descriptions = (
            result[text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.strip()
        )

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), analyzer="word")
        try:
            taxonomy_vectors = vectorizer.fit_transform(path_text)
        except ValueError as error:
            raise ValueError("Taxonomy paths contain no usable text for TF-IDF.") from error
        description_vectors = vectorizer.transform(descriptions.tolist())

        similarities = cosine_similarity(
            description_vectors,
            taxonomy_vectors
        )

        best_indices = similarities.argmax(axis=1)

        for level in range(4):
            result[f"l{level + 1}"] = [
                paths[i][level] if len(paths[i]) > level else ""
                for i in best_indices
            ]

        result["confidence"] = np.clip(similarities.max(axis=1), 0.0, 1.0)

        return result
