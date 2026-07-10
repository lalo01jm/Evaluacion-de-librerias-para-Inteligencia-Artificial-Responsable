import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


TARGET = "two_year_recid"
AUDIT_COLUMNS = [
    "entity_id", "model_id", "feature_set", "fold",
    "race", "sex", "age_cat", "label_value", "score",
]


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = np.nan_to_num(tp / (tp + fp))
    recall = np.nan_to_num(tp / (tp + fn))
    f1 = np.nan_to_num(2 * precision * recall / (precision + recall))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def train_model(
    compas_preprocessed,
    compas_audit_context,
    model=None,
    model_name="LR_8_features",
    feature_set=8,
    n_splits=10,
    random_state=42,
    target=TARGET,
):
    required = {"entity_id", "race", "sex", "age_cat", "label_value"}
    missing = required.difference(compas_audit_context.columns)
    if missing:
        raise ValueError(f"Faltan columnas en compas_audit_context: {sorted(missing)}")
    if len(compas_preprocessed) != len(compas_audit_context):
        raise ValueError("Los DataFrames no tienen el mismo número de filas.")
    if not np.array_equal(compas_preprocessed[target].to_numpy(), compas_audit_context["label_value"].to_numpy()):
        raise ValueError("Las etiquetas no coinciden fila a fila.")

    model = LogisticRegression(max_iter=2000) if model is None else model
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = compas_preprocessed.drop(columns=[target])
    y = compas_preprocessed[target]

    fold_metrics = []
    audit_folds = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        fold_model = clone(model)
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = fold_model.predict(X.iloc[test_idx]).astype(int)

        metrics = compute_metrics(y.iloc[test_idx], preds)
        metrics["fold"] = fold
        fold_metrics.append(metrics)

        audit_fold = compas_audit_context.iloc[test_idx].copy()
        audit_fold.insert(1, "model_id", model_name)
        audit_fold.insert(2, "feature_set", str(feature_set))
        audit_fold.insert(3, "fold", fold)
        audit_fold["score"] = preds
        audit_folds.append(audit_fold)

    metrics_df = pd.DataFrame(fold_metrics).drop(columns="fold")
    audit_predictions = pd.concat(audit_folds, ignore_index=True).loc[:, AUDIT_COLUMNS]

    result = {
        "model": model_name,
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": cv.n_splits,
            "shuffle": cv.shuffle,
            "random_state": cv.random_state,
        },
        "metrics": metrics_df.mean().to_dict(),
        "metrics_std": metrics_df.std(ddof=1).to_dict(),
        "fold_metrics": fold_metrics,
    }
    return result, audit_predictions
