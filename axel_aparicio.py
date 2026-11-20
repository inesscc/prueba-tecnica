# PRUEBA TÉCNICA INE
# Axel Aparicio
#
# Primero revisaré las dimensiones y la estructura
# de los datos para conocer las variables disponibles
# antes de seleccionar y entrenar un modelo.

# import getpass
# import pandas as pd
# import mysql.connector
#
# conexion = mysql.connector.connect(
#     host="sql.freedb.tech",
#     port=3306,
#     user="u_y8tlWZ",
#     password=getpass.getpass("Contraseña MySQL: "),
#     database="freedb_N4nnid0a"
# )
#
# cursor = conexion.cursor(dictionary=True)
#
# cursor.execute("SELECT * FROM train")
# train = pd.DataFrame(cursor.fetchall())
#
# cursor.execute("SELECT * FROM test")
# test = pd.DataFrame(cursor.fetchall())
#
# print("TRAIN:", train.shape)
# print("TEST:", test.shape)
# print("COLUMNAS:", train.columns.tolist())
# print(train.head())
#
# cursor.close()
# conexion.close()

# PREGUNTA 1: Clasificación de textos

import pandas as pd

# Carga de datos locales
train = pd.read_parquet("data/train_glosas.parquet")
test = pd.read_parquet("data/test_glosas.parquet")
noticias = pd.read_parquet("data/noticias_sample.parquet")

print("NOTICIAS:", noticias.shape)
print("COLUMNAS NOTICIAS:", noticias.columns.tolist())

print("TRAIN:", train.shape)
print("TEST:", test.shape)
print("COLUMNAS:", train.columns.tolist())
print(train.head())

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Se utiliza actividad, oficio y tareas porque las tres
# aportan información para identificar el código del empleo.
columnas = ["actividad", "oficio", "tareas"]

# Se Completan los textos faltantes con cadenas vacías.
# No utilizo el ID porque no aporta información del oficio.
X_train = train[columnas].fillna("").astype(str).agg(" ".join, axis=1)
y_train = train["cod_final"].astype(str)

X_test = test[columnas].fillna("").astype(str).agg(" ".join, axis=1)

# Selecciono TF-IDF para transformar textos en números
# y LinearSVC para clasificar. Es una solución sencilla
# y rápida, considerando el tiempo de la prueba.

modelo = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        strip_accents="unicode",
        max_features=50000
    )),
    ("clasificador", LinearSVC())
])

print("\nEntrenando clasificador...")
modelo.fit(X_train, y_train)

# Predicciones del conjunto de testeo
predicciones = modelo.predict(X_test)
test["prediccion"] = predicciones

print("\nPrimeras predicciones:")
print(test[["cod_final", "prediccion"]].head(10))

# Se evualuan solo registros que tengan etiqueta real
validos = test["cod_final"].notna()

if validos.any():
    reales = test.loc[validos, "cod_final"].astype(str)
    estimados = test.loc[validos, "prediccion"]

    print("\nAccuracy:",
          accuracy_score(reales, estimados))

    print("Macro F1:",
          f1_score(reales, estimados,
                   average="macro", zero_division=0))

    print("\nReporte de clasificacion:")
    print(classification_report(
        reales, estimados, zero_division=0
    ))
else:
    print("El conjunto test no contiene etiquetas para evaluar.")

# 1. Análisis de resultados
#
# El modelo obtuvo un Accuracy de 90,20%, lo que indica
# que clasificó correctamente la mayoría de los registros
# del conjunto de prueba.
#
# Sin embargo, el Macro F1 fue de 76,82%.
# Esto muestra que el desempeño no es igual para todas
# las categorías.
#
# Por ejemplo, los códigos 85 y 48 presentan buenos
# resultados, mientras que los códigos 09 y 27 tienen
# un F1 considerablemente menor.
#
# Considero que es un resultado inicial satisfactorio,
# especialmente porque se utilizó un modelo relativamente
# sencillo y no se realizó optimización.
#
# La principal limitación es que algunas categorías
# todavía presentan dificultades de clasificación.

# 2. Plan de mejora (sin implementar)
#
# Primero revisaría cuántos ejemplos de entrenamiento
# tiene cada código, ya que podría existir un desequilibrio
# entre las categorías.
#
# También analizaría los registros mal clasificados
# para identificar descripciones ambiguas, errores en
# las etiquetas o actividades similares entre sí.
#
# Luego evaluaría mejorar el procesamiento del texto
# y comparar otros clasificadores utilizando validación.
#
# Finalmente, comprobaría las mejoras con datos
# independientes antes de utilizar el modelo en producción.

# PREGUNTA 2: Modelo en producción

# 1. Análisis de resultados
#
# Al revisar los resultados, observo que el modelo pierde
# rendimiento con el paso de los meses.
#
# En testing tenía un Accuracy de 93,8%, pero en el mes 10
# baja a 83%. Algo similar ocurre con el Macro F1, que
# disminuye de 82,3% a 71%.
#
# Me fijé sobre todo en el Macro F1, porque le da el mismo
# peso a todas las categorías, también a las que tienen
# pocos registros. Que baje tanto me hace pensar que el
# modelo está fallando en algunos códigos puntuales.

# 2. Posibles causas
#
# Mi primera hipótesis es que las descripciones de los
# empleos hayan cambiado respecto de los datos originales.
#
# Por ejemplo, podrían aparecer nuevas ocupaciones,
# nombres diferentes para un mismo oficio o nuevas formas
# de describir las tareas.
#
# También podría haber cambios en los criterios utilizados
# para asignar determinados códigos.
#
# Igual, esto es solo una hipótesis: antes de afirmarlo
# habría que mirar los errores y los datos de cada período.
#
# Ojo también con que cada mes son solo 100 registros,
# así que parte de las diferencias entre meses podría ser
# simple ruido.

# 3. Estrategia propuesta
#
# Lo primero sería ver en qué códigos se están cometiendo
# más errores y comparar los textos nuevos con los del
# entrenamiento.
#
# Si se confirma que algo cambió, juntaría registros
# recientes bien etiquetados y probaría reentrenar.
# Antes de reemplazar el modelo actual lo compararía con
# la versión nueva, usando datos que no se hayan usado
# para entrenar.
#
# Y dejaría un control periódico de Accuracy y Macro F1
# para darme cuenta a tiempo si vuelve a caer.

# PREGUNTA 3: Resumen de noticias

import requests
import getpass

# Solicito la clave para no dejarla publicada en GitHub.
api_key = getpass.getpass("API key OpenRouter: ")

resumenes = []

for i, cuerpo in enumerate(noticias["cuerpo"].fillna(""), start=1):

    respuesta = requests.post(
        "https://openrouter.ai/api/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "google/gemini-2.5-flash-lite",
            "reasoning": {"enabled": False},
            "input": (
                "Resume esta noticia en español en un máximo de "
                "35 palabras. Termina con una oración completa. "
                "Conserva los hechos importantes y no inventes "
                "información:\n\n" + str(cuerpo)
            )
        },
        timeout=60
    )

    respuesta.raise_for_status()

    resumen = respuesta.json()["output"][0]["content"][0]["text"]
    resumen = " ".join(resumen.split()[:50])

    resumenes.append(resumen)
    print(f"Noticia {i}: {resumen}")

noticias["resumen"] = resumenes

print("Cantidad de resúmenes:", len(resumenes))



# PREGUNTA 4: Experiencia personal

# Problema de negocio
#
# En la UMCE hemos trabajado con modelos predictivos
# para identificar estudiantes que podrían presentar
# problemas de rendimiento o abandonar sus estudios.
#
# La idea es detectar estos casos a tiempo y no
# cuando el problema ya ocurrió.


# Estrategia utilizada
#
# Para esto utilizamos datos históricos de matrícula,
# notas y rendimiento académico.
#
# Primero revisamos y preparamos los datos, luego
# seleccionamos las variables y evaluamos los modelos
# para ver qué tan bien identifican a los estudiantes
# en riesgo.
#
# También participo en PRINU, una plataforma que
# estamos desarrollando para atender clientes
# mediante inteligencia artificial.


# Reflexión y aprendizaje
#
# He aprendido que no basta con que un modelo tenga
# buenos resultados, también debe ser útil para
# resolver el problema.
#
# Por ejemplo, de poco sirve tener un Accuracy alto
# si no logramos detectar a los estudiantes en riesgo.
#
# Creo que lo más importante es entender el problema,
# trabajar con datos confiables y obtener resultados
# que ayuden a tomar mejores decisiones.
