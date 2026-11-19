#%%
import os
import mysql.connector
import pandas as pd
from dotenv import load_dotenv
import re
import unicodedata
import time
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
import requests
#%%
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
}

#%% 1 Clasificando texto 
def leer_tabla(nombre_tabla, config=DB_CONFIG):
    conexion = mysql.connector.connect(**config)
    try:
        df = pd.read_sql(f"SELECT * FROM {nombre_tabla}", conexion)
    finally:
        conexion.close()
    return df


def listar_tablas(config=DB_CONFIG):
    conexion = mysql.connector.connect(**config)
    try:
        cursor = conexion.cursor()
        cursor.execute("SHOW TABLES")
        tablas = [fila[0] for fila in cursor.fetchall()]
        cursor.close()
    finally:
        conexion.close()
    return tablas

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def preparar_texto(df):
    unido = df[CAMPOS_TEXTO].astype(str).agg(" ".join, axis=1)
    return unido.map(limpiar_texto)


def agregar_columnas_limpias(df):
    for c in CAMPOS_TEXTO:
        df[c + "_limpio"] = df[c].map(limpiar_texto)
    df["texto"] = preparar_texto(df)
    return df

train = leer_tabla("train")
print("\ntrain:", train.shape)
print("\nColumnas:", list(train.columns))
print("\nNulos:")
print(train.isna().sum())
print("\nSin datos o \"\":")
print((train.astype(str).apply(lambda s: s.str.strip()) == "").sum())
print("\nDuplicados:", train.duplicated().sum())

#%%

CAMPOS_TEXTO = ["actividad", "oficio", "tareas"]

train = agregar_columnas_limpias(train)
print(train[CAMPOS_TEXTO + ["texto"]].sample(5, random_state=0).to_string())

#%% El mismo oficio cae en distintos códigos

for oficio in ["profesor", "profesora", "tecnico paramedico", "secretaria",
               "guardia de seguridad", "vendedora"]:
    codigos = train.loc[train["oficio_limpio"] == oficio, "cod_final"].value_counts()
    print(f"{oficio:22s} n={codigos.sum():4d}  códigos distintos={codigos.size:3d}  "
          f"más frecuentes: {dict(codigos.head(4))}")

profesores = train[train["oficio_limpio"].isin(["profesor", "profesora"])]
for cod in ["85", "84"]:
    print(f"\nprofesor/a con código {cod}:",
          profesores.loc[profesores["cod_final"] == cod, "actividad"].head(4).tolist())

#%% Pregunta 2

def vectorizador():
    return FeatureUnion([
        ("palabras", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                       min_df=3, sublinear_tf=True)),
    ])


REPRESENTACIONES = {
    "concatenado": ["texto"],
    "solo actividad": ["actividad_limpio"],
    "solo oficio": ["oficio_limpio"],
    "solo tareas": ["tareas_limpio"],
    "por campo": ["actividad_limpio", "oficio_limpio", "tareas_limpio"],
}


def construir_modelo(columnas, class_weight=None):
    tfidf = ColumnTransformer([(c, vectorizador(), c) for c in columnas])
    return Pipeline([
        ("tfidf", tfidf),
        ("clf", LinearSVC(C=1.0, class_weight=class_weight, max_iter=5000)),
    ])


X_tr, X_val, y_tr, y_val = train_test_split(
    train, train["cod_final"],
    test_size=0.15, stratify=train["cod_final"], random_state=42,
)

filas = []
for representacion, columnas in REPRESENTACIONES.items():
    for pesos in [None, "balanced"]:
        t0 = time.time()
        p = construir_modelo(columnas, pesos).fit(X_tr, y_tr).predict(X_val)
        filas.append({
            "representacion": representacion,
            "pesos": pesos or "sin balancear",
            "accuracy": accuracy_score(y_val, p),
            "macro_f1": f1_score(y_val, p, average="macro"),
            "segundos": time.time() - t0,
        })

comparacion = pd.DataFrame(filas).sort_values("macro_f1", ascending=False)
print(comparacion.round(4).to_string(index=False))

#%% Pregunta 2: decisiones
"""
Limpié el texto de forma simple: minúsculas, sin tildes ni puntuación. Son
textos cortos, así que no quité stopwords ni lematicé; casi todas las palabras
aportan.

Usé TF-IDF de palabras y de caracteres. Los caracteres ayudan con faltas de
ortografía y variantes ("contruccion", "operaria"), comunes en texto digitado a
mano. Como clasificador, un SVM lineal: rinde bien con este tipo de datos y
entrena en segundos.

Para elegir separé un 15% de train y dejé test para el final. Concatenar los
tres campos fue lo peor (macro F1 0.74): mezcla el oficio con el empleador, y el
código depende del empleador. Me quedé con un vectorizador por campo y pesos
balanceados (0.79), porque hay códigos con 5 ejemplos y otros con más de 5.000.
Elegí por macro F1 porque la accuracy casi no cambia entre opciones.
"""

#%% Preguntas 3 y 4

mejor = comparacion.iloc[0]
columnas_mejor = REPRESENTACIONES[mejor["representacion"]]
pesos_mejor = None if mejor["pesos"] == "sin balancear" else "balanced"
print("Elegido en validación:", mejor["representacion"], "/", mejor["pesos"])

test = agregar_columnas_limpias(leer_tabla("test"))

modelo = construir_modelo(columnas_mejor, pesos_mejor).fit(train, train["cod_final"])
test["prediccion"] = modelo.predict(test)

print(f"\nTest  accuracy: {accuracy_score(test['cod_final'], test['prediccion']):.4f}")
print(f"Test  macro F1: {f1_score(test['cod_final'], test['prediccion'], average='macro'):.4f}")

#%% Insumos pregunta 5

reporte = pd.DataFrame(classification_report(
    test["cod_final"], test["prediccion"], output_dict=True, zero_division=0)).T
por_clase = reporte.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")

print("Peores 10 clases por F1:")
print(por_clase.sort_values("f1-score").head(10).round(3).to_string())

print("\nMejores 10 clases por F1 (con al menos 20 casos en test):")
print(por_clase[por_clase["support"] >= 20].sort_values("f1-score", ascending=False)
      .head(10).round(3).to_string())

errores = test[test["cod_final"] != test["prediccion"]]
print(f"\nErrores: {len(errores)} de {len(test)}")
print("Pares real -> predicho más frecuentes:")
print(errores.groupby(["cod_final", "prediccion"]).size()
      .sort_values(ascending=False).head(10).to_string())

solo_actividad = test["oficio"].astype(str).str.strip() == ""
for nombre, mascara in [("solo actividad", solo_actividad), ("tres campos", ~solo_actividad)]:
    sub = test[mascara]
    print(f"\n{nombre:15s} n={len(sub):5d}  accuracy={accuracy_score(sub['cod_final'], sub['prediccion']):.4f}")

# Términos con más peso por código en el modelo final (solo n-gramas de palabra)
nombres = pd.Series(modelo.named_steps["tfidf"].get_feature_names_out())
coeficientes = modelo.named_steps["clf"].coef_
clases = list(modelo.named_steps["clf"].classes_)
es_palabra = nombres.str.contains("__palabras__").values

print("\nTérminos con más peso por código (campo: término):")
for cod in ["84", "85", "86", "48", "80", "41", "43"]:
    pesos = pd.Series(coeficientes[clases.index(cod)][es_palabra], index=nombres[es_palabra])
    top = pesos.nlargest(8).index.str.replace("_limpio__palabras__", ": ", regex=False)
    print(f"  {cod}: {', '.join(top)}")

#%% Pregunta 5
"""
En test el modelo llega a 0.92 de accuracy y 0.80 de macro F1, casi igual que
en validación, así que no veo señales de sobreajuste en la elección.

Lo que más me ayudó a entender el problema es que el código describe al
empleador, no a la persona. Una profesora de liceo va al 85, pero si trabaja
para el departamento de educación municipal va al 84. "Guardia de seguridad"
aparece en 47 códigos. Por eso actividad es el campo clave: solo, llega a 0.92
de accuracy; oficio, a 0.61. Y casi todos los términos con más peso del modelo
vienen de actividad.

Funciona muy bien en los códigos grandes con vocabulario propio: educación,
salud, agricultura y transporte superan 0.95 de F1.

Le cuesta en tres cosas. Los códigos con pocos ejemplos: cuatro tienen F1 de 0,
y eso es lo que baja la macro F1. Los códigos vecinos, como 41 y 43
(construcción), o salud y educación municipal, que terminan en 84; algunos de
estos podrían ser ambiguos incluso para una persona. Y los registros que solo
traen actividad, con 0.70 de accuracy contra 0.92, aunque son apenas 46.
"""

#%% Pregunta 6
"""
Iría de lo más barato a lo más costoso.

Primero revisaría los errores con alguien que conozca el clasificador, sobre
todo salud y educación municipal y el par 41/43. Si la regla es ambigua, hay que
precisarla antes de tocar el modelo.

Después pediría etiquetar más casos de los códigos chicos, eligiendo los que el
modelo clasifica con poca confianza. Al azar casi todo cae en los códigos
grandes.

Con eso ajustaría los hiperparámetros con validación cruzada, que no hice para 
evitar estar optimizando.

Solo entonces probaría embeddings o un BERT en español, comparado contra este
modelo. Puede ayudar con sinónimos, pero cuesta más, y lo dejaría solo si mejora
la macro F1.

En producción, mandaría a revisión manual los casos de baja confianza. No
mejora el modelo, pero sí lo que se entrega, y da las etiquetas que más sirven
para reentrenar.
"""

#%% 2 Modelo en producción
"""
Con muestras de 100 registros, la caída entre el testing y el mes 6 (de 0.938 a
0.913) equivale a 2 o 3 errores más por cada 100 casos y puede deberse al azar
del muestreo, aunque que baje en todas las mediciones obliga a vigilarla. Además,
0.923 y 0.913 no son valores posibles con exactamente 100 registros, así que
convendría revisar cómo se calcularon. En el mes 10, en cambio, son unos 11
errores más, lo que ya no se explica por azar y sugiere que algo cambió.

Tengo dos hipótesis. La primera es que, como el etiquetado es manual, haya
cambiado el criterio de los codificadores, ya sea por personas nuevas o por una
actualización del manual, de modo que el modelo se evalúa contra una regla
distinta a la que aprendió. La segunda es que haya llegado un grupo de textos
poco representado en el entrenamiento, como trabajos de temporada, oficios nuevos
o registros que solo traen la actividad, y bastaría con que fuera 1 de cada 5 o
10 registros para explicar la caída. La macro F1 siempre está bajo la accuracy
porque las clases están desbalanceadas, y con 100 registros un solo error en un
código poco frecuente la mueve bastante, por lo que no la usaría sola para
decidir.

Antes de reentrenar intentaría saber cuál de las dos cosas pasó, pidiendo que se
vuelvan a etiquetar casos antiguos para ver si cambian las etiquetas y revisando
si los errores del mes 10 se concentran en ciertos códigos o tipos de texto. Si
cambió el criterio, reentrenaría con las etiquetas nuevas y dejaría documentada
la regla; si cambió la población, agregaría ejemplos de ese grupo al
entrenamiento. Para adelante, etiquetaría todos los meses con un muestreo que
incluya los códigos poco frecuentes y definiría una alerta cuando la métrica
salga del rango esperado para el tamaño de la muestra.
""" 
#%% 3 Procesando texto

OPENROUTER_URL = "https://openrouter.ai/api/v1/responses"
MODELO_LLM = "google/gemini-2.5-flash-lite"
MAX_PALABRAS = 50

noticias = leer_tabla("noticias")
print("noticias:", noticias.shape, list(noticias.columns))


def llamar_llm(prompt, api_key, temperatura=0):
    respuesta = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": MODELO_LLM,
            "input": prompt,
            "reasoning": {"effort": "none"},  # razonamiento desactivado (costos)
            "temperature": temperatura,
        },
        timeout=60,
    )
    respuesta.raise_for_status()
    return respuesta.json()["output"][0]["content"][0]["text"].strip()


def resumen_valido(texto):
    # No vacío, dentro del límite, un solo párrafo y sin prefijos tipo "Resumen:".
    return (
        0 < len(texto.split()) <= MAX_PALABRAS
        and "\n" not in texto
        and not texto.startswith("resumen")
    )


def resumir(cuerpo, titular, api_key=os.getenv("GEMINI_CREDENCIAL"), intentos=2):
    # El titular sirve de ancla para el hecho central; se pide 40 palabras para
    # dejar margen, porque los LLM no cuentan palabras con exactitud.
    prompt = (
        "Resume la noticia en español, en un solo párrafo de máximo 40 palabras.\n"
        "- Usa el titular como guía de cuál es el hecho central, pero no lo copies.\n"
        "- Prioriza qué pasó, quién, cuándo y dónde; omite detalles secundarios.\n"
        "- Usa solo información que esté en el cuerpo, sin agregar ni opinar.\n"
        "Responde solo con el resumen, sin introducciones.\n\n"
        f"Titular: {titular}\n\nNoticia: {cuerpo}"
    )
    for i in range(intentos):
        # Primer intento determinista; en el reintento se sube algo la temperatura,
        # porque con 0 el modelo devolvería la misma respuesta inválida.
        # Se pasa a minúscula para que quede consistente con los textos de la base.
        texto = llamar_llm(prompt, api_key, temperatura=0 if i == 0 else 0.3).lower()
        if resumen_valido(texto):
            return texto
    # Último recurso: si tras los reintentos no cumple, se recorta a 50 palabras.
    return " ".join(texto.split()[:MAX_PALABRAS])


noticias["resumen"] = [resumir(c, t) for c, t in zip(noticias["cuerpo"], noticias["titular"])]
noticias["n_palabras"] = noticias["resumen"].str.split().str.len()

pd.set_option("display.max_colwidth", None)
print(noticias[["titular", "resumen", "n_palabras"]].to_string())

#%% 4 Cuéntanos sobre un proyecto 
"""
En un proyecto de salud teníamos las reglas de negocio y el conocimiento
clínico repartidos entre documentos, conversaciones por distintos medios y
decisiones que quedaban en reuniones, y cada área los entendía a su manera.
Armamos un RAG para consolidar todo eso y que se usara bien en tres niveles:
entender la parte administrativa, interpretar correctamente los datos y
detectar relaciones que deberían levantar una alerta.

Para resolverlo juntamos los documentos, los hilos de conversación y las
transcripciones de reuniones, y a partir de ahí diseñamos ontologías por
ámbito, con las entidades, sus relaciones y las reglas que las conectan. Esas
ontologías alimentan el motor RAG, así que la búsqueda no depende solo de que
las palabras se parezcan sino de cómo se relacionan los conceptos. Encima
trabajan varios agentes, cada uno a cargo de un nivel, y uno que coordina y
decide a cuál derivar cada consulta.

Lo que más aprendí es que el modelo importa menos que el trabajo previo de
documentar y alfabetizar al equipo. Antes de construir hay que dejar definidos
los parámetros: qué significa cada término, qué plazos aplican y qué
herramienta se usa para cada dato. Si eso no está acordado con quienes conocen
el negocio, el sistema encuentra el texto correcto pero responde con criterios
distintos. A quien empiece le diría que parta por ahí, y que trate lo que se
decide en reuniones y conversaciones como una fuente tan válida como un
documento formal, porque muchas veces ahí está la regla que de verdad se
aplica.
"""

# %%
