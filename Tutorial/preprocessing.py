import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from fairlearn.preprocessing import (
    CorrelationRemover,
    PrototypeRepresentationLearner,
)


def Preprocesar(csv_path, preprocessing_method=None):
    """
    Carga y prepara el conjunto de datos COMPAS para entrenamiento y auditoría.
    
    ----------
    csv_path : Ruta al archivo ``compas-scores-two-years.csv``.
    preprocessing_method : Método de mitigación de Fairlearn. Puede ser ``None``, 
                           ``"CorrelationRemover"`` o ``"PrototypeRepresentationLearner"``.
                           También se puede proporcionar directamente una instancia compatible.

    Devuelve: El primer DataFrame contiene las características listas para entrenar y
            la variable objetivo. El segundo conserva el contexto sensible sin
            transformar para auditar posteriormente las predicciones.
    """
    # Nombre de la variable que el modelo intentará predecir:
    # 1 indica reincidencia dentro de dos años y 0 indica que no hubo reincidencia.
    target = "two_year_recid"

    # Variables seleccionadas para entrenar el modelo.
    features = [
        "sex",
        "age",
        "juv_fel_count",
        "juv_misd_count",
        "priors_count",
        "c_charge_degree",
        "race",
    ]

    # Atributos sensibles utilizados por los métodos de mitigación de Fairlearn.
    sensitive = ["race", "sex"]

    # Se carga el archivo CSV original en un DataFrame.
    df = pd.read_csv(csv_path)

    # Se aplican los mismos criterios de filtrado usados en el análisis público
    # de COMPAS: cercanía entre la evaluación y el arresto, casos válidos de
    # reincidencia, cargos conocidos y puntuaciones disponibles.
    df = df[
        df["days_b_screening_arrest"].between(-30, 30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()

    # Antes de codificar los datos se guarda un contexto de auditoría con los
    # atributos originales. Este DataFrame permitirá asociar cada predicción
    # con la persona y los grupos protegidos a los que pertenece.
    audit = df[["id", "race", "sex", "age_cat", target]].rename(
        columns={
            "id": "entity_id",
            target: "label_value",
        }
    ).reset_index(drop=True)

    # Se crea la tabla de trabajo con las características y la variable objetivo.
    data = df[features + [target]].copy()

    # Se identifican automáticamente las columnas categóricas.
    cat_cols = data.select_dtypes(
        include=["object", "category"],
    ).columns

    # Las columnas restantes se consideran numéricas.
    num_cols = data.columns.difference(cat_cols)

    # Se fuerza la conversión de las variables numéricas. Los valores que no
    # puedan convertirse se reemplazan temporalmente por valores ausentes.
    data[num_cols] = data[num_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Los valores numéricos ausentes se imputan con la mediana de su columna.
    data[num_cols] = data[num_cols].fillna(
        data[num_cols].median(),
    )

    # Los valores categóricos ausentes se marcan como "Unknown" y después cada
    # categoría se transforma en un número entero mediante OrdinalEncoder.
    data[cat_cols] = OrdinalEncoder().fit_transform(
        data[cat_cols].fillna("Unknown"),
    ).astype(int)

    # Se asegura que la variable objetivo sea binaria y de tipo entero.
    data[target] = data[target].astype(int)

    # Se separan las características de entrada y las etiquetas reales.
    X = data[features].copy()
    y = data[target].copy()

    # Se convierte la opción recibida en el objeto de preprocesamiento adecuado.
    method = _get_preprocessor(
        preprocessing_method,
        sensitive,
    )

    # MITIGACIÓN DE SESGO: 
    # ----------------------------------------------------------------------

    # CorrelationRemover elimina de las características la información
    # linealmente correlacionada con los atributos sensibles indicados.
    if isinstance(method, CorrelationRemover):
        transformed = method.fit_transform(X)

        # El resultado ya no incluye las columnas sensibles, por lo que se
        # reconstruye el DataFrame usando únicamente las columnas restantes.
        remaining_columns = [
            column for column in X.columns if column not in sensitive
        ]
        X = pd.DataFrame(
            transformed,
            columns=remaining_columns,
        )

    # PrototypeRepresentationLearner aprende una nueva representación de los
    # ejemplos intentando conservar utilidad predictiva y reducir diferencias
    # entre los grupos sensibles.
    elif isinstance(method, PrototypeRepresentationLearner):
        # Se crea un identificador interseccional combinando raza y sexo para
        # que el método conozca el grupo sensible de cada observación.
        groups = X[sensitive].astype(str).agg("_".join, axis=1)

        # Se aprende y aplica la representación basada en prototipos.
        transformed = method.fit_transform(
            X,
            y,
            sensitive_features=groups,
        )
        X = pd.DataFrame(transformed)

        # Las nuevas variables no conservan los nombres originales, por lo que
        # se asignan nombres descriptivos y consecutivos.
        X.columns = [
            f"prototype_{index}" for index in range(X.shape[1])
        ]

    # Se rechaza cualquier método que no sea uno de los tipos admitidos.
    elif method is not None:
        raise ValueError(
            "preprocessing_method debe ser None, CorrelationRemover "
            "o PrototypeRepresentationLearner"
        )
    
    X[target] = y.to_numpy()
    
    return X.reset_index(drop=True), audit


def _get_preprocessor(method, sensitive):
    """
    Convierte el nombre o la clase recibida en un preprocesador de Fairlearn.
    """
    # None indica que se entrenará un modelo base sin mitigación previa.
    if method in (None, "None"):
        return None

    # Se admite tanto el nombre del método como la propia clase.
    if method in ("CorrelationRemover", CorrelationRemover):
        # Las columnas sensibles se indican explícitamente para que Fairlearn
        # pueda retirar su correlación de las demás características.
        return CorrelationRemover(
            sensitive_feature_ids=sensitive,
        )

    # Se admite tanto el nombre del método como la propia clase.
    if method in (
        "PrototypeRepresentationLearner",
        PrototypeRepresentationLearner,
    ):
        # Se fija la semilla para que la transformación sea reproducible.
        return PrototypeRepresentationLearner(
            random_state=0,
        )

    # Si ya se recibió una instancia, se devuelve sin modificarla. La función
    # principal verificará después que sea de un tipo compatible.
    return method
