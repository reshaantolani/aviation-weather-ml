from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from aviation_weather_ml.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ]
    categorical_pipeline = Pipeline(steps=categorical_steps)

    transformers = [
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ]

    return ColumnTransformer(transformers=transformers)


def build_logistic_model() -> Pipeline:
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    steps = [
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ]

    return Pipeline(steps=steps)


def build_random_forest_model() -> Pipeline:
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    steps = [
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ]

    return Pipeline(steps=steps)
