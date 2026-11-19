#PROBLEMA 1: CLASIFICANDO TEXTO
from pathlib import Path
from dotenv import load_dotenv
import re
# NOTA: SE UTILIZA .env para cargar las variables de entorno necesarias para la conexión a la base de datos MySQL y la API key de la aplicación.
load_dotenv(Path(__file__).with_name(".env"))
import os
import mysql.connector
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

#Conexión a la base de datos MySQL
with mysql.connector.connect(
    host=os.environ["DB_HOST"],
    port=int(os.environ["DB_PORT"]),
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    database=os.environ["DB_NAME"],
    connection_timeout=15
) as conexion:
    if conexion.is_connected():
        print("Conexión exitosa a MySQL")
    else:
        print("Error al conectar a MySQL")

    with conexion.cursor(dictionary=True) as cursor:
        # El límite del servidor se expresa en milisegundos y aplica a cada consulta.
        cursor.execute("SET SESSION max_execution_time = 60000")
        cursor.execute("SELECT * FROM train")
        registros_train = cursor.fetchall()

        cursor.execute("SELECT * FROM test")
        registros_test = cursor.fetchall()

#1. Descargar los datos de la tabla train

train = pd.DataFrame(registros_train)

print("Dimensiones del entrenamiento:")
print(train.shape)

print("\nPrimeros registros:")
print(train.head())

print("\nColumnas:")
print(train.columns.tolist())

#Revisión de calidad de los datos

print("\nTipos de datos:")
print(train.dtypes)

print("\nValores nulos:")
print(train.isnull().sum())

print("\nDistribución de códigos:")
print(train["cod_final"].value_counts())

print("\nCantidad de categorías:")
print(train["cod_final"].nunique())

#2. Entrenamiento del clasificador de texto

# Selección de columnas de texto y limpieza de valores nulos

columnas_texto = ["actividad", "oficio", "tareas"]

train[columnas_texto] = (
    train[columnas_texto]
    .fillna("")
    .astype(str)
)

#Concatenar columnas de texto 

train["texto"] = (
    train["actividad"] + " " +
    train["oficio"] + " " +
    train["tareas"]
)

#Normalización de texto: a minúsculas, eliminar caracteres especiales

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

train["texto"] = train["texto"].apply(limpiar_texto)

# Preparar la variable de entrenamiento

train = train.dropna(subset=["cod_final"]).copy()

train["cod_final"] = (
    train["cod_final"]
    .astype(str)
    .str.strip()
)

# Entrenamiento del clasificador de texto
# Para ello se utilizará TF-IDF para vectorizar los textos y una SVM como clasificador, adecuada
# para textos vectorizados

from sklearn.model_selection import train_test_split

X = train["texto"]
y = train["cod_final"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

#Construcción del modelo

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

modelo = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
    ),
    (
        "clasificador",
        LinearSVC(
            random_state=42,
            max_iter=5000
        )
    )
])

# Entrenamiento del modelo
modelo.fit(X_train, y_train)

print("Modelo entrenado correctamente")

# Evaluación del modelo en el conjunto de validación

y_pred = modelo.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

f1_macro = f1_score(
    y_val,
    y_pred,
    average="macro",
    zero_division=0
)

print("Accuracy:", round(accuracy, 4))
print("F1-macro:", round(f1_macro, 4))

print("\nReporte de clasificación:")
print(
    classification_report(
        y_val,
        y_pred,
        zero_division=0
    )
)

#3. Descargar los datos de la tabla test en la base de datos

test = pd.DataFrame(registros_test)

print("Dimensiones de test:")
print(test.shape)

print("\nPrimeros registros:")
print(test.head())

print("\nColumnas:")
print(test.columns.tolist())

# Preprocesamiento de la tabla test

test[columnas_texto] = (
    test[columnas_texto]
    .fillna("")
    .astype(str)
)

test["texto"] = (
    test["actividad"] + " " +
    test["oficio"] + " " +
    test["tareas"]
)

test["texto"] = test["texto"].apply(limpiar_texto)

# 4. Predicción de los códigos finales para la tabla test

modelo.fit(X, y)

print("Modelo final entrenado con todo train")

test["cod_predicho"] = modelo.predict(test["texto"])

print(
    test[
        ["id", "actividad", "oficio", "tareas", "cod_predicho"]
    ].head(10)
)

#Revisión de etiquetas en cod_final

if "cod_final" in test.columns:

    test_evaluable = test.dropna(
        subset=["cod_final"]
    ).copy()

    test_evaluable["cod_final"] = (
        test_evaluable["cod_final"]
        .astype(str)
        .str.strip()
    )

    if not test_evaluable.empty:

        acc_test = accuracy_score(
            test_evaluable["cod_final"],
            test_evaluable["cod_predicho"]
        )

        f1_test = f1_score(
            test_evaluable["cod_final"],
            test_evaluable["cod_predicho"],
            average="macro",
            zero_division=0
        )

        print("Accuracy test:", round(acc_test, 4))
        print("F1-macro test:", round(f1_test, 4))

    else:
        print("Test no tiene etiquetas válidas para evaluar")

else:
    print("Test no contiene cod_final: solo se generan predicciones")

os.makedirs("resultados", exist_ok=True)

test.to_csv(
    "resultados/predicciones_test.csv",
    index=False,
    encoding="utf-8-sig"
)
print("Archivos CSV generados")

# 5. Análisis del modelo

# Se implementó un clasificador supervisado utilizando TF-IDF y LinearSVC. 
# Como variables predictoras se utilizaron la actividad económica, el oficio y las tareas 
# desempeñadas por cada persona.
# 
# El modelo obtuvo un Accuracy de 0.882 y un F1-macro de 0.686 en el conjunto de evaluación.

# Fortalezas: La solución es sencilla de implementar y permite aprovechar simultáneamente 
# las tres descripciones textuales disponibles. Mediante el uso de un Pipeline, se garantiza
# un procesamiento consistente entre entrenamiento y predicción.

# Limitaciones: El modelo representa las descripciones principalmente según sus términos 
# y frecuencias, sin embargo, no se incluyó un componente semántico. Por eso podría 
# confundir ocupaciones similares o presentar dificultades ante descripciones ambiguas, 
# incompletas o poco frecuentes.

# Además, la revisión de la distribución de clases, de las métricas por categoría y de los registros
#  clasificados incorrectamente permitirá identificar con mayor precisión qué códigos 
# requieren atención.

#6. Plan de mejora

# Para mejorar el desempeño del modelo y su capacidad de predicción se propone lo siguiente:
# - Análisis los errores por categoría, viendo aquellos en que el modelo confunde con mayor frecuencia.
# - Revisar la distribución de clases para detectar categorías con pocos ejemplos.
# - Evaluar la calidad de las etiquetas para encontrar posibles inconsistencias en cod_final.
# - Mejorar la normalización de los textos, tratando abreviaturas, errores ortográficos y variantes de términos.
# - Probar otros  modelos, como Regresión Logística o embeddings de texto y evaluar su rendimiento.
# - Evaluar con nuevos registros para verificar la capacidad de generalización ante datos distintos.


# PROBLEMA 2: MODELO EN PRODUCCIÓN

# En los 10 meses en que el modelo ha estado en producción, se observa que el accuracy
# a disminuido 0.108 respecto al testing y el valor de F1-macro ha mostrado una caída similar,
# de 0.113 respecto al testing. Esto explica que el desempeño del modelo en producción
# ha disminuido respecto al conjunto de prueba.

# Las hipótesis que pueden explicar esta disminución pueden ser:
# a) Cambios en la distribución de los datos, respecto a los que recibía durante el entrenamiento, por ejemplo
# que hayan aparecido nuevas ocupaciones (nuevos códigos) o cambios en la forma de describir las tareas, lo que
# afecta la frecuencia de los datos respecto al entrenamiento y de esta forma, las métricas de accuracy y F1-macro
# también se ven afectadas, por estos nuevos datos ingresados.

# b) Cambios en la calidad de los datos, como errores en la transcripción de las 
# descripciones o inconsistencias en las etiquetas, lo que puede afectar negativamente el 
# desempeño del modelo en producción, por lo que es importante realizar continuamente el procesamiento.
# Esto incluye la limpieza de los datos, la verificación de la consistencia de las etiquetas y la actualización del modelo si es necesario.

# c) Cambios en la relación de los datos y las categorías, por ejemplo, que haya ocurrido una
# nueva forma de clasificación, dado un cambio en las responsabilidades o criterios para asignar los códigos,
# que son distintos a los que el modelo había aprendido durante el entrenamiento. Por ello, es necesario,
# re entrenar el modelo con los nuevos datos y ajustar su capacidad de generalización.

# La estrategia para mitigar este problema, en primer lugar se debería establecer un 
# monitoreo continuo del desempeño del modelo en producción, para detectar cualquier 
# deterioro en las métricas de accuracy y F1-macro. Luego, se realizaría un etiquetado de los datos frente a
# los cambios que surjan en el mercado laboral y actualizar el modelo cuando se sobrepase un umbral de deterioro
# en su capacidad de predicción, evaluando con datos recientes e independientes de los que fueron usados en el entrenamiento.

# PROBLEMA 3: PROCESANDO TEXTO

import os
import mysql.connector
import pandas as pd
import requests


#Conexión a la base de datos

with mysql.connector.connect(
    host=os.environ["DB_HOST"],
    port=int(os.environ.get("DB_PORT", "3306")),
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASSWORD"],
    database=os.environ["DB_NAME"],
    connection_timeout=15
) as conexion:
    print("Conexión exitosa a MySQL")

    #1. Descarga de la tabla noticias

    with conexion.cursor(dictionary=True) as cursor:
        cursor.execute("SELECT * FROM noticias")
        registros = cursor.fetchall()
        noticias = pd.DataFrame(registros)

print("Cantidad de noticias:", len(noticias))
print(noticias.head())

if noticias.empty:
    raise ValueError("La tabla noticias está vacía")

if not {"cuerpo", "titular"}.issubset(noticias.columns):
    raise ValueError("Faltan columnas cuerpo o titular")

#Configurar OpenRouter

API_KEY = os.environ["OPENROUTER_API_KEY"]

URL = "https://openrouter.ai/api/v1/responses"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

#2. Generar el resumen de la noticia, considerando las instrucciones dadas

def generar_resumen(cuerpo, titular):

    prompt = f"""
    Resume la siguiente noticia en español.

    Instrucciones:
    - Máximo 50 palabras.
    - Mantén los hechos más importantes.
    - No agregues información.
    - Utiliza lenguaje claro y objetivo.
    - Entrega únicamente el resumen.

    Titular: {titular}

    Noticia:
    {cuerpo}
    """

    payload = {
        "model": "google/gemini-2.5-flash-lite",
        "input": prompt,
        "reasoning": {
            "enabled": False
        },
        "max_output_tokens": 150,
        "temperature": 0
    }

    respuesta = requests.post(
        URL,
        headers=HEADERS,
        json=payload,
        timeout=90
    )

    respuesta.raise_for_status()

    datos = respuesta.json()

    # Extraer el texto según la estructura solicitada
    resumen = datos["output"][0]["content"][0]["text"]

    # Asegurar un máximo de 50 palabras
    palabras = resumen.split()
    resumen = " ".join(palabras[:50])

    return resumen


#Generar resúmenes de las 10 noticias

resumenes = []

for i, fila in noticias.iterrows():

    print(f"Procesando noticia {i + 1}/{len(noticias)}")

    try:
        resumen = generar_resumen(
            str(fila["cuerpo"] or ""),
            str(fila["titular"] or "")
        )

    except (
        requests.RequestException,
        KeyError,
        IndexError,
        TypeError,
        ValueError
    ) as error:
        print(f"Error en noticia {i + 1}: {error}")
        resumen = None

    resumenes.append(resumen)

#Crear la columna resumen

noticias["resumen"] = resumenes

#Verificar el límite de 50 palabras

noticias["cantidad_palabras"] = (
    noticias["resumen"]
    .fillna("")
    .apply(lambda texto: len(texto.split()))
)

print("\nRESULTADOS:")
print(
    noticias[
        ["titular", "resumen", "cantidad_palabras"]
    ].to_string(index=False)
)

print("\nResúmenes generados correctamente:",
      noticias["resumen"].notna().sum())

# Guardar los resultados en CSV

os.makedirs("resultados", exist_ok=True)

noticias.to_csv(
    "resultados/noticias_resumidas.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nArchivo generado: resultados/noticias_resumidas.csv")

# PROBLEMA 4: CONTAR SOBRE UN PROYECTO

# 1. Problema de negocio: Para desarrollar un prototipo de un sistema de información climática para
# la Subsecretaría de Ciencia, la situación planteada fue recopilar, procesar y poner al público
# la información sobre cambio climático generada en Chile desde distintas fuentes. El principal desafío fue
# integrar información de distintas fuentes, cada una con su propio formato y nivel de calidad, 
# y garantizar que los datos fueran accesibles y útiles para los usuarios finales.


# 2. Estrategia utilizada: La solución presentada se desarrolló en AWS, combinando procesos de ingeniería de datos, 
# un repositorio DSpace-CRIS y una arquitectura RAG. Para ello, en primer lugar se implementó un proceso ETL utilizando
# Python para recopilar desde documentos y fuentes de texto, filtrar y normalizar la información textual, identificar registros duplicados y enriquecer metadatos 
# mediante APIs externas. Luego, se implementó una arquitectura RAG con PostgreSQL, con un mecanismo de recuperación híbrica
# entre búsquedas semánticas y léxicas, mejorando la pertinencia de los resultados, entregada a un modelo de lenguaje natural
# para su posterior consulta y generación de respuestas en lenguaje natural, mediante un chatbot de consulta sobre los documentos.

# 3. Reflexión y aprendizaje: Este proyecto me permitió comprender la importancia de la calidad de los datos y la necesidad 
# de un enfoque estructurado para su procesamiento antes de utilizar modelos de lenguaje natural, pudiendo conocer
# además la relevancia de combinar estrategias de recuperación de información semántica y léxica para mejorar la pertinencia 
# de los resultados y presentar información relevante de manera comprensible para los usuarios finales.
# Como consejo a quienes comienzan y quieren integrar estas herramientas es realizar prototipos acotados al problema a resolver,
# validar la calidad de los datos y evaluar los resultados antes de escalar a una solución completa.
