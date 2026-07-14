import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


# Nombre predeterminado de la variable objetivo.
TARGET = "two_year_recid"

# Orden final de las columnas que utilizará Aequitas.
AUDIT_COLUMNS = [
    "entity_id",
    "model_id",
    "feature_set",
    "fold",
    "race",
    "sex",
    "age_cat",
    "label_value",
    "score",
]

# Esta función es un ejemplo de cómo calcular métricas de clasificación binaria a partir de etiquetas y predicciones. 
# Se utiliza en la función `train_model` para evaluar el desempeño del modelo en cada fold de validación cruzada.
# (Hace una pequeña parte de lo que hace la librería Aequitas).
def compute_metrics(y_true, y_pred):
    """
    Calcula métricas de clasificación binaria a partir de etiquetas y predicciones.
    """
    # Se genera la matriz de confusión usando siempre el orden de clases [0, 1].
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    # Se extraen verdaderos negativos, falsos positivos, falsos negativos y
    # verdaderos positivos de la matriz de confusión.
    tn, fp, fn, tp = cm.ravel()

    precision = np.nan_to_num(
        tp / (tp + fp),
    )

    recall = np.nan_to_num(
        tp / (tp + fn),
    )

    f1 = np.nan_to_num(
        2 * precision * recall / (precision + recall),
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
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
    """
    Entrena y evalúa un clasificador mediante validación cruzada estratificada.

    Además de las métricas predictivas, genera una tabla con una predicción
    fuera de muestra para cada observación. Esa tabla queda lista para ser
    auditada con Aequitas.

    Parameters
    ----------
    compas_preprocessed : pandas.DataFrame
        Características numéricas y variable objetivo.
    compas_audit_context : pandas.DataFrame
        Identificador, atributos protegidos y etiqueta real sin transformar.
    model : estimator or None
        Clasificador compatible con scikit-learn. Si es ``None``, se utiliza
        regresión logística.
    model_name : str
        Identificador que aparecerá en la tabla de auditoría.
    feature_set : object
        Etiqueta descriptiva del conjunto de características utilizado.
    n_splits : int
        Número de particiones de la validación cruzada.
    random_state : int
        Semilla usada para reproducir la división de los datos.
    target : str
        Nombre de la columna objetivo.

    Returns
    -------
    tuple[dict, pandas.DataFrame]
        Resumen de entrenamiento y tabla de predicciones para auditoría.
    """
    # Se especifican las columnas mínimas que debe contener el contexto de auditoría.
    required = {
        "entity_id",
        "race",
        "sex",
        "age_cat",
        "label_value",
    }

    # Se comprueba que todas las columnas necesarias estén disponibles.
    missing = required.difference(
        compas_audit_context.columns,
    )
    if missing:
        raise ValueError(
            f"Faltan columnas en compas_audit_context: {sorted(missing)}"
        )

    # Ambos DataFrames deben representar exactamente las mismas observaciones.
    if len(compas_preprocessed) != len(compas_audit_context):
        raise ValueError(
            "Los DataFrames no tienen el mismo número de filas."
        )

    # También se verifica que las etiquetas coincidan fila por fila. Esta
    # comprobación evita asociar una predicción con el contexto de otra persona.
    labels_match = np.array_equal(
        compas_preprocessed[target].to_numpy(),
        compas_audit_context["label_value"].to_numpy(),
    )
    if not labels_match:
        raise ValueError(
            "Las etiquetas no coinciden fila a fila."
        )

    # Si no se proporcionó un estimador, se usa regresión logística como modelo base.
    model = (
        LogisticRegression(max_iter=2000)
        if model is None
        else model
    )

    # Se configura validación cruzada estratificada para conservar aproximadamente
    # la proporción de las clases en cada partición.
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Se separan las características de la variable objetivo.
    X = compas_preprocessed.drop(
        columns=[target],
    )
    y = compas_preprocessed[target]

    # Esta lista almacenará las métricas obtenidas en cada fold.
    fold_metrics = []

    # Esta lista almacenará el contexto y las predicciones de prueba de cada fold.
    audit_folds = []

    # Cada observación se usa una sola vez como dato de prueba, de modo que la
    # tabla final contiene predicciones fuera de muestra para todo el conjunto.
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        fold_model = clone(model)

        fold_model.fit(
            X.iloc[train_idx],
            y.iloc[train_idx],
        )

        preds = fold_model.predict(
            X.iloc[test_idx],
        ).astype(int)

        metrics = compute_metrics(
            y.iloc[test_idx],
            preds,
        )

        metrics["fold"] = fold
        fold_metrics.append(metrics)

        # Se recupera el contexto sensible de las observaciones de prueba.
        audit_fold = compas_audit_context.iloc[
            test_idx
        ].copy()

        # Se añade la información necesaria para identificar el modelo, el
        # conjunto de variables y el fold que produjo cada predicción.
        audit_fold.insert(
            1,
            "model_id",
            model_name,
        )
        audit_fold.insert(
            2,
            "feature_set",
            str(feature_set),
        )
        audit_fold.insert(
            3,
            "fold",
            fold,
        )

        # Aequitas utiliza la columna score para representar la decisión binaria.
        audit_fold["score"] = preds

        audit_folds.append(audit_fold)

    # Se crea una tabla numérica con las métricas de todos los folds. El número
    # de fold se elimina antes de calcular estadísticas agregadas.
    metrics_df = pd.DataFrame(
        fold_metrics,
    ).drop(columns="fold")

    # Se unen las predicciones de todos los folds y se ordenan las columnas en
    # el formato que espera la etapa de auditoría.
    audit_predictions = pd.concat(
        audit_folds,
        ignore_index=True,
    ).loc[:, AUDIT_COLUMNS]

    # Se construye un resumen con la configuración de validación cruzada, las
    # métricas promedio, su desviación estándar y el detalle de cada fold.
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

    # Se devuelven tanto los resultados predictivos como la tabla lista para Aequitas.
    return result, audit_predictions
