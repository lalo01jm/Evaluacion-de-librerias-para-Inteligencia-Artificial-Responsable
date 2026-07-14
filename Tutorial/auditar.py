from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness


# Atributos protegidos que se analizarán durante la auditoría.
ATTR_COLS = ["race", "sex", "age_cat"]

# Grupo de referencia utilizado para calcular las disparidades de cada atributo.
REF_GROUPS = {"race": "Caucasian", "sex": "Male", "age_cat": "25 - 45"}


def auditar_modelo(
    audit_predictions,
    attr_cols=ATTR_COLS,
    ref_groups=REF_GROUPS,
    tau=0.80,
    alpha=0.05,
):
    """
    Audita las predicciones de un modelo mediante Aequitas.

    Parameters
    ----------
    audit_predictions : pandas.DataFrame
        Tabla con las predicciones y el contexto de auditoría. Debe incluir
        ``score``, ``label_value`` y las columnas indicadas en ``attr_cols``.
    attr_cols : list[str]
        Atributos protegidos que se evaluarán.
    ref_groups : dict[str, str]
        Grupo de referencia para cada atributo protegido.
    tau : float
        Umbral de paridad utilizado por Aequitas. Con el valor 0.80 se aplica
        la regla del 80 % para decidir si una disparidad se considera justa.
    alpha : float
        Nivel de significancia empleado en las pruebas estadísticas.

    Returns
    -------
    dict
        Diccionario con métricas absolutas por grupo, disparidades y resultados
        de equidad por grupo, atributo y modelo completo.
    """
    # Se crea una copia para evitar modificar el DataFrame original.
    df = audit_predictions.copy()

    # Aequitas necesita identificar el modelo. Si la columna no existe,
    # se agrega un identificador genérico.
    if "model_id" not in df:
        df["model_id"] = "modelo"

    # Se construye la lista exacta de columnas necesarias para la auditoría.
    cols = ["model_id", "score", "label_value", *attr_cols]

    # Se comprueba que estén presentes todas las columnas requeridas.
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    df = df.loc[:, cols].copy()
    df[["score", "label_value"]] = df[["score", "label_value"]].astype(int)

    # Los atributos protegidos se convierten a texto para tratarlos como grupos
    # categóricos, incluso si originalmente estaban codificados como números.
    df[attr_cols] = df[attr_cols].astype(str)

    # Group calcula la matriz de confusión y las métricas absolutas para cada
    # valor de cada atributo protegido: TPR, FPR, precisión, prevalencia, etc.
    group_metrics, _ = Group().get_crosstabs(df, attr_cols=attr_cols)

    # Bias compara cada grupo con su grupo de referencia y calcula razones de
    # disparidad. También verifica si las diferencias son estadísticamente
    # significativas usando el nivel alpha indicado.
    disparities = Bias().get_disparity_predefined_groups(
        group_metrics,
        original_df=df,
        ref_groups_dict=ref_groups,
        check_significance=True,
        alpha=alpha,
        mask_significance=True,
    )

    # Objeto encargado de convertir las disparidades en evaluaciones
    # de equidad de acuerdo con el umbral tau.
    fairness = Fairness()

    # Se determina la equidad de cada métrica para cada grupo.
    group_fairness = fairness.get_group_value_fairness(
        disparities,
        tau=tau,
    )

    # Se resume la equidad de todos los grupos pertenecientes a cada atributo.
    attribute_fairness = fairness.get_group_attribute_fairness(
        group_fairness,
    )

    # Se obtiene una evaluación global de equidad para el modelo auditado.
    overall_fairness = fairness.get_overall_fairness(
        attribute_fairness,
    )
    
    return {
        "group_metrics": group_metrics,
        "disparities": disparities,
        "group_fairness": group_fairness,
        "attribute_fairness": attribute_fairness,
        "overall_fairness": overall_fairness,
    }
