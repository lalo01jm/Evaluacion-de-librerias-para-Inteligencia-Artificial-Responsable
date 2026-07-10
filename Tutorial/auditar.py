from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness

ATTR_COLS = ["race", "sex", "age_cat"]
REF_GROUPS = {"race": "Caucasian", "sex": "Male", "age_cat": "25 - 45"}


def auditar_modelo(audit_predictions, attr_cols=ATTR_COLS, ref_groups=REF_GROUPS,
                   tau=0.80, alpha=0.05):
    """
    Audita un DataFrame de predicciones con Aequitas.

    Requiere columnas: score, label_value, race, sex, age_cat.
    Si no existe model_id, se crea uno genérico.
    """
    df = audit_predictions.copy()
    if "model_id" not in df:
        df["model_id"] = "modelo"

    cols = ["model_id", "score", "label_value", *attr_cols]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    df = df.loc[:, cols].copy()
    df[["score", "label_value"]] = df[["score", "label_value"]].astype(int)
    df[attr_cols] = df[attr_cols].astype(str)

    group_metrics, _ = Group().get_crosstabs(df, attr_cols=attr_cols)
    disparities = Bias().get_disparity_predefined_groups(
        group_metrics,
        original_df=df,
        ref_groups_dict=ref_groups,
        check_significance=True,
        alpha=alpha,
        mask_significance=True,
    )

    fairness = Fairness()
    group_fairness = fairness.get_group_value_fairness(disparities, tau=tau)
    attribute_fairness = fairness.get_group_attribute_fairness(group_fairness)
    overall_fairness = fairness.get_overall_fairness(attribute_fairness)

    return {
        "group_metrics": group_metrics,
        "disparities": disparities,
        "group_fairness": group_fairness,
        "attribute_fairness": attribute_fairness,
        "overall_fairness": overall_fairness,
    }
