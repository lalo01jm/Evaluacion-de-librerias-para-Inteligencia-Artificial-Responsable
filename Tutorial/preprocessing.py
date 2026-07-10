import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from fairlearn.preprocessing import CorrelationRemover, PrototypeRepresentationLearner


def Preprocesar(csv_path, preprocessing_method=None):
    target = "two_year_recid"
    features = ["sex", "age", "juv_fel_count", "juv_misd_count", "priors_count", "c_charge_degree", "race"]
    sensitive = ["race", "sex"]

    df = pd.read_csv(csv_path)
    df = df[
        df["days_b_screening_arrest"].between(-30, 30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    audit = df[["id", "race", "sex", "age_cat", target]].rename(
        columns={"id": "entity_id", target: "label_value"}
    ).reset_index(drop=True)

    data = df[features + [target]].copy()
    cat_cols = data.select_dtypes(include=["object", "category"]).columns
    num_cols = data.columns.difference(cat_cols)

    data[num_cols] = data[num_cols].apply(pd.to_numeric, errors="coerce")
    data[num_cols] = data[num_cols].fillna(data[num_cols].median())
    data[cat_cols] = OrdinalEncoder().fit_transform(data[cat_cols].fillna("Unknown")).astype(int)
    data[target] = data[target].astype(int)

    X, y = data[features], data[target]
    method = _get_preprocessor(preprocessing_method, sensitive)

    if isinstance(method, CorrelationRemover):
        X = pd.DataFrame(method.fit_transform(X), columns=[c for c in X.columns if c not in sensitive])
    elif isinstance(method, PrototypeRepresentationLearner):
        groups = X[sensitive].astype(str).agg("_".join, axis=1)
        X = pd.DataFrame(method.fit_transform(X, y, sensitive_features=groups))
        X.columns = [f"prototype_{i}" for i in range(X.shape[1])]
    elif method is not None:
        raise ValueError("preprocessing_method debe ser None, CorrelationRemover o PrototypeRepresentationLearner")

    X[target] = y.to_numpy()
    return X.reset_index(drop=True), audit


def _get_preprocessor(method, sensitive):
    if method in (None, "None"):
        return None
    if method in ("CorrelationRemover", CorrelationRemover):
        return CorrelationRemover(sensitive_feature_ids=sensitive)
    if method in ("PrototypeRepresentationLearner", PrototypeRepresentationLearner):
        return PrototypeRepresentationLearner(random_state=0)
    return method
