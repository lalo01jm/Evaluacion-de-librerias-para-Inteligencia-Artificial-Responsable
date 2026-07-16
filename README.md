
# Evaluación de librerías para Inteligencia Artificial Responsable

Repositorio educativo en español para estudiar sesgo algorítmico en modelos de
clasificación y explorar herramientas de auditoría y mitigación con
[Aequitas](https://github.com/dssg/aequitas) y
[Fairlearn](https://fairlearn.org/).

El proyecto utiliza el conjunto de datos COMPAS como caso de estudio. Su objetivo
es mostrar un flujo reproducible que separa la preparación de los datos, el
entrenamiento predictivo y la evaluación de equidad entre grupos.

## Objetivos

- Analizar variables y posibles fuentes de sesgo en el conjunto de datos COMPAS.
- Entrenar modelos de clasificación con predicciones fuera de muestra.
- Medir desempeño predictivo y disparidades entre grupos.
- Auditar resultados con Aequitas.
- Explorar métodos de preprocesamiento y mitigación de Fairlearn.
- Documentar las diferencias prácticas entre ambas librerías.

## Recorrido recomendado

Ejecuta los notebooks desde la raíz del repositorio y en este orden:

1. [`1.Preprocesamiento.ipynb`](1.Preprocesamiento.ipynb): carga, filtra y
   transforma los datos; también conserva el contexto sensible necesario para la
   auditoría.
2. [`2.Entrenamiento_Modelos.ipynb`](2.Entrenamiento_Modelos.ipynb): entrena los
   modelos mediante validación cruzada estratificada y genera predicciones fuera
   de muestra.
3. [`3.Auditoria (Aequitas).ipynb`](<3.Auditoria%20%28Aequitas%29.ipynb>): calcula
   métricas por grupo, disparidades y evaluaciones de equidad.

Como alternativa compacta, [`Tutorial/Tutorial.ipynb`](Tutorial/Tutorial.ipynb)
presenta el flujo completo utilizando las funciones reutilizables de la carpeta
`Tutorial/`.

Los notebooks `analisis_compas_random_forest.ipynb`, `Compas Analysis.ipynb` y
`Compas_Analysis_Python.ipynb` contienen exploraciones complementarias y material
de referencia; no son requisitos para ejecutar el flujo principal.

## Datos y resultados

- `data/raw/` contiene los archivos de entrada del caso COMPAS.
- `data/processed/` contiene datos transformados, contexto de auditoría y
  resultados generados por los notebooks.
- `data/archive/` conserva material auxiliar o histórico.

Los atributos sensibles analizados incluyen raza, sexo y categoría de edad. Las
métricas de equidad deben interpretarse junto con el tamaño de cada grupo, la
calidad de los datos, los grupos de referencia elegidos y el contexto social del
problema.

## Herramientas estudiadas

| Herramienta | Uso principal en el proyecto                                                        |
| ----------- | ----------------------------------------------------------------------------------- |
| Aequitas    | Auditoría posterior al entrenamiento, métricas por grupo y razones de disparidad. |
| Fairlearn   | Preprocesamiento, mitigación y evaluación de diferencias entre grupos.            |

## Reproducibilidad

El entrenamiento usa particiones estratificadas y semillas fijas cuando el
método lo permite. Aun así, las versiones de las dependencias pueden afectar los
resultados. Una siguiente mejora prevista es fijar versiones compatibles y
automatizar la ejecución de pruebas y notebooks.

## Limitaciones

- COMPAS es un caso controvertido y no representa todos los contextos de riesgo
  algorítmico.
- Una métrica de paridad no demuestra por sí sola que un sistema sea justo.
- La elección de atributos sensibles, grupos de referencia y umbrales afecta las
  conclusiones.
