# Prueba técnica - Científico/a de datos
# Pedro Aravena
#
# Estructura del archivo: cada pregunta va en un bloque "# %%" (se puede correr
# por celdas en VS Code / Positron o completo con `python pedro_aravena.py`).
# La API key de OpenRouter se lee desde la variable de entorno OPENROUTER_API_KEY
# o desde .venv/key.txt (no versionado), para no dejarla en el repositorio.

# %% Librerías y conexión ---------------------------------------------------
import os
import re
import time

import mysql.connector
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

DB = dict(
    host="sql.freedb.tech",
    port=3306,
    user="u_y8tlWZ",
    password="wSkxw3xbhsQg",
    database="freedb_N4nnid0a",
)


def conectar(intentos: int = 30, espera: int = 10):
    """El usuario de la base es compartido y el servidor limita las conexiones
    simultáneas (error 1203 max_user_connections). Reintento hasta tener cupo."""
    for i in range(intentos):
        try:
            return mysql.connector.connect(**DB, connection_timeout=30)
        except mysql.connector.Error as e:
            print(f"Intento {i + 1}: {e.msg} -> espero {espera}s")
            time.sleep(espera)
    raise RuntimeError("No se pudo conectar a la base")


# Rutas relativas a la carpeta del script y no a la carpeta desde donde se
# ejecuta (VS Code puede correrlo desde la raíz del workspace). En modo
# interactivo por celdas __file__ no existe: busco la carpeta que contiene
# este archivo partiendo de la carpeta actual.
try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = next(
        (d for d in (os.getcwd(), os.path.join(os.getcwd(), "prueba-tecnica"))
         if os.path.exists(os.path.join(d, "pedro_aravena.py"))),
        os.getcwd(),
    )
print("Carpeta base:", BASE)
DATA_DIR = os.path.join(BASE, "..", "data")  # fuera del repo
os.makedirs(DATA_DIR, exist_ok=True)


def leer_tabla(tabla: str, bloque: int = 2000) -> pd.DataFrame:
    """Descarga una tabla desde MySQL a un DataFrame.

    El servidor (freedb, gratuito) corta las consultas que superan su tiempo
    máximo de ejecución: un SELECT * de la tabla completa falló con el error
    3024. Por eso intento subir el límite de la sesión y además descargo por
    bloques con LIMIT/OFFSET. Cada tabla se guarda en CSV local la primera vez,
    para no depender de nuevo del servidor.
    """
    ruta = os.path.join(DATA_DIR, f"{tabla}.csv")
    if os.path.exists(ruta):
        return pd.read_csv(ruta, dtype={"cod_final": str})
    con = conectar()
    cur = con.cursor()
    try:
        cur.execute("SET SESSION max_execution_time = 0")
    except Exception as e:
        print("No se pudo cambiar max_execution_time:", e)
    partes, offset = [], 0
    try:
        while True:
            cur.execute(f"SELECT * FROM {tabla} LIMIT {bloque} OFFSET {offset}")
            filas = cur.fetchall()
            if not filas:
                break
            cols = [c[0] for c in cur.description]
            partes.append(pd.DataFrame(filas, columns=cols))
            offset += bloque
    finally:
        con.close()
    df = pd.concat(partes, ignore_index=True)
    # Sin ORDER BY el orden no está garantizado entre bloques; elimino
    # posibles duplicados por id como resguardo.
    df = df.drop_duplicates(subset="id") if "id" in df.columns else df
    df.to_csv(ruta, index=False)
    return df


# %% =========================================================================
# CLASIFICANDO TEXTO
# ============================================================================

# %% 1) Descargar train -----------------------------------------------------
train = leer_tabla("train")
train.head()            
train.shape          
train.info()            
train["cod_final"].value_counts()   
train.sample(10)        
print(train.head())
print(train.isna().sum())

# cod_final lo trato como etiqueta categórica (string). Si viene como número
# se pierde el cero a la izquierda ("01" -> 1), así que lo normalizo a 2 dígitos.
train["cod_final"] = train["cod_final"].astype(str).str.strip().str.zfill(2)

dist = train["cod_final"].value_counts()
print(f"Clases: {dist.size}")
print(dist.head(10))
print(dist.tail(10))


# %% 2) Entrenar el clasificador -------------------------------------------
# Preprocesamiento: concateno actividad + oficio + tareas en un solo texto.
# Las tres columnas aportan información complementaria: la actividad apunta
# a la rama económica (que es lo que parece codificar cod_final, ej. 84 =
# administración pública, 41 = construcción), mientras oficio y tareas
# ayudan cuando la actividad viene vacía o es ambigua ("corporación municipal").
# Antepongo un prefijo a cada campo para que el modelo pueda distinguir de
# qué columna viene cada palabra sin tener que armar tres vectorizadores.

def limpiar(s) -> str:
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    s = s.lower()
    s = re.sub(r"[^\wáéíóúñü ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def armar_texto(df: pd.DataFrame) -> pd.Series:
    return (
        "act " + df["actividad"].map(limpiar)
        + " ofi " + df["oficio"].map(limpiar)
        + " tar " + df["tareas"].map(limpiar)
    )


# Modelo: TF-IDF de palabras (1-2 gramas) + TF-IDF de caracteres (3-5 gramas)
# alimentando un SVM lineal.
# - Los textos son cortos, escritos a mano por encuestadores, con abreviaturas
#   y faltas de ortografía ("paramedico", "rrhh", "proteccion"). Los n-gramas
#   de caracteres hacen al modelo robusto a eso sin tener que corregir texto.
# - SVM lineal sobre TF-IDF es un baseline clásico, rápido y difícil de
#   superar en clasificación de textos cortos con muchas clases. Con el tiempo
#   disponible prefiero esto a un modelo de embeddings o un fine-tuning.
# - class_weight="balanced" para que las clases chicas no queden aplastadas
#   por las grandes (me importa el macro F1, no solo la accuracy).
# - No hago búsqueda de hiperparámetros: el enunciado pide no optimizar.
modelo = Pipeline([
    ("tfidf", FeatureUnion([
        ("palabras", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])),
    ("clf", LinearSVC(C=1.0, class_weight="balanced")),
])

X_train = armar_texto(train)
y_train = train["cod_final"]

t0 = time.time()
modelo.fit(X_train, y_train)
print(f"Entrenamiento: {time.time() - t0:.1f} s")


# %% 3) Descargar test ------------------------------------------------------
test = leer_tabla("test")
test["cod_final"] = test["cod_final"].astype(str).str.strip().str.zfill(2)
print(test.shape)

# Clases que aparecen en test y no en train: el modelo nunca va a acertarlas.
no_vistas = set(test["cod_final"]) - set(train["cod_final"])
print("Clases en test no vistas en train:", no_vistas)


# %% 4) Predicciones --------------------------------------------------------
X_test = armar_texto(test)
test["pred"] = modelo.predict(X_test)


# %% 5) Análisis de resultados ---------------------------------------------
acc = accuracy_score(test["cod_final"], test["pred"])
f1_macro = f1_score(test["cod_final"], test["pred"], average="macro")
f1_w = f1_score(test["cod_final"], test["pred"], average="weighted")
print(f"Accuracy: {acc:.3f} | Macro F1: {f1_macro:.3f} | Weighted F1: {f1_w:.3f}")

# Baseline trivial: predecir siempre la clase más frecuente. Sirve para
# dimensionar cuánto aporta el modelo.
print(f"Baseline clase mayoritaria: {(test['cod_final'] == dist.index[0]).mean():.3f}")

reporte = pd.DataFrame(
    classification_report(test["cod_final"], test["pred"], output_dict=True, zero_division=0)
).T
reporte = reporte.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
reporte["n_train"] = reporte.index.map(dist).fillna(0).astype(int)
reporte = reporte.sort_values("f1-score")
print("\nClases con peor F1:")
print(reporte.head(10).round(2))
print("\nClases con mejor F1:")
print(reporte.tail(5).round(2))

# Relación entre tamaño de la clase en train y desempeño
print("\nCorrelación (spearman) n_train vs F1:",
      round(reporte[["n_train", "f1-score"]].corr(method="spearman").iloc[0, 1], 2))

# Pares de confusión más frecuentes
errores = test[test["cod_final"] != test["pred"]]
print("\nConfusiones más frecuentes (real -> predicho):")
print(errores.groupby(["cod_final", "pred"]).size().sort_values(ascending=False).head(10))

# Algunos ejemplos de errores para mirarlos a ojo
pd.set_option("display.max_colwidth", 80)
print(errores[["actividad", "oficio", "tareas", "cod_final", "pred"]].sample(min(10, len(errores)), random_state=1))

# --- Comentarios sobre los resultados ---
# Test: 3.927 registros, 79 clases.
#   Accuracy 0.910 | Macro F1 0.789 | Weighted F1 0.910
#   Baseline (siempre clase 48): 0.156
#
# Fortalezas:
# - 0.91 de accuracy contra 0.16 del baseline, sin tocar hiperparámetros y
#   entrenando en ~17 s. Es barato de reentrenar y se puede ver qué palabras
#   pesan en cada código.
# - Las clases grandes (48, 01, 84, 85) salen bien, y son la mayoría de los casos.
#
# Espacios de mejora:
# - La diferencia entre accuracy y macro F1 dice que el problema son las clases
#   chicas. Algunas con 5-15 casos en train (63, 70, 59) no se aciertan nunca.
#   Igual no es solo tamaño (spearman n_train vs F1 = 0.37), hay clases chicas
#   con F1 = 1.
# - El F1 por clase hay que leerlo con cuidado: muchas clases tienen menos de 10
#   casos en test y un error mueve mucho el número. La 63 tiene soporte 0, o
#   sea son puros falsos positivos.
# - Las confusiones son entre clases parecidas: 41/42/43 (construcción) y
#   84/85/86 (adm. pública, educación, salud). Ahí manda una regla de
#   codificación más que el texto (ej. profesor de corporación municipal -> 84),
#   y el modelo no la conoce. Varios errores son textos ambiguos que también
#   le costarían a un codificador.


# %% 6) Plan de mejora (sin ejecutar) ---------------------------------------
# 1. Revisar las confusiones con los codificadores y el manual. Si es una regla
#    (empleador público pesa más que la tarea), agregarla como variable o regla
#    previa al modelo. Es lo más barato.
# 2. Validación cruzada estratificada, para no sacar conclusiones con clases
#    de 2 casos en test.
# 3. Clases chicas: más ejemplos etiquetados o sobremuestreo. Probar un modelo
#    jerárquico (sección CIIU y después división), ya que los errores quedan
#    casi siempre dentro de la misma sección.
# 4. Probar un vectorizador por campo y comparar con embeddings o un BERT en
#    español (BETO). Solo si la mejora justifica el costo.
# 5. En operación: mandar a revisión manual los casos de baja confianza.


# %% =========================================================================
# MODELO EN PRODUCCIÓN
# ============================================================================
# Primero ver qué es ruido. Con 100 casos al mes, el error estándar de la
# accuracy es ~sqrt(0.92*0.08/100) = 0.027. Los meses 3 y 6 caen menos que eso
# respecto al testing, así que no se puede decir que el modelo empeoró. El mes
# 10 cae ~0.11 (unos 4 errores estándar), eso sí es una caída real. El macro F1
# con 100 casos y 79 clases es todavía más inestable.
#
# Hipótesis: drift. El modelo aprendió con una foto fija y los datos cambiaron:
# - Cambia la composición del empleo (estacionalidad, oficios nuevos como
#   reparto por apps). Si suben clases que el modelo predice peor, baja la
#   accuracy.
# - Cambia cómo se escribe el texto (nuevos encuestadores, cambio de
#   cuestionario o de sistema de captura).
# - Cambia la etiqueta (actualización del manual o del clasificador): mismo
#   texto, otro código.
# Que caiga poco hasta el mes 6 y mucho al 10 me hace pensar en cambios que se
# acumulan o en algo puntual entre esos meses.
#
# Estrategia:
# 1. Diagnosticar antes de reentrenar: comparar mes 10 vs train en distribución
#    de clases, palabras fuera del vocabulario, largo de textos y confianza del
#    modelo. Y preguntar a terreno si cambió algo.
# 2. Monitorear esas métricas todos los meses sobre todos los registros (no
#    necesitan etiqueta), como alerta temprana.
# 3. Reportar métricas con intervalos, acumular meses y etiquetar más casos de
#    baja confianza o de clases raras.
# 4. Reentrenar periódicamente con datos recientes y cambiar el modelo solo si
#    le gana al actual en datos nuevos.
# 5. Mientras, mandar los casos dudosos a codificación manual.


# %% =========================================================================
# PROCESANDO TEXTO (resúmenes con LLM)
# ============================================================================

# %% 1) Descargar noticias --------------------------------------------------
noticias = leer_tabla("noticias")
print(noticias.shape)

# %% 2) Generar resúmenes ---------------------------------------------------
def leer_key() -> str:
    """La key no va en el código. Se toma de la variable de entorno
    OPENROUTER_API_KEY o, si no está, de .venv/key.txt (no versionado).
    Busco el archivo en la carpeta del script y en la carpeta actual, para
    que funcione tanto al correr el archivo como por celdas o selección."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    base = globals().get("BASE", os.getcwd())
    candidatas = [
        os.path.join(base, ".venv", "key.txt"),
        os.path.join(os.getcwd(), ".venv", "key.txt"),
        os.path.join(os.getcwd(), "prueba-tecnica", ".venv", "key.txt"),
    ]
    for ruta in candidatas:
        if os.path.exists(ruta):
            with open(ruta) as f:
                return f.read().strip()
    raise RuntimeError("Falta la API key: defina OPENROUTER_API_KEY o cree .venv/key.txt")


OPENROUTER_KEY = leer_key()
URL = "https://openrouter.ai/api/v1/responses"

PROMPT = (
    "Resume la siguiente noticia en español en un máximo de 40 palabras. "
    "Responde solo con el resumen, sin título ni comentarios.\n\n"
    "Titular: {titular}\n\nNoticia: {cuerpo}"
)
# Pido 40 y no 50 porque los modelos tienden a pasarse un poco del límite
# que se les indica; igual valido después y recorto si excede 50.


def resumir(cuerpo: str, titular: str, reintentos: int = 6) -> str:
    payload = {
        "model": "google/gemini-2.5-flash-lite",
        "input": PROMPT.format(titular=titular, cuerpo=cuerpo),
        "reasoning": {"effort": "none"},  # razonamiento desactivado
        "temperature": 0,                 # resultados reproducibles
        "max_output_tokens": 200,
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"}
    for intento in range(reintentos):
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            salida = r.json()["output"]
            # Busco el bloque de tipo "message" por si la API devuelve otros
            # elementos antes (ej. reasoning); en el caso normal es output[0].
            msg = next(o for o in salida if o.get("type") == "message")
            return msg["content"][0]["text"].strip()
        except Exception as e:
            # El error más común fue 429 (límite de solicitudes de la key
            # compartida): espero cada vez más antes de reintentar.
            print(f"Error (intento {intento + 1}): {e}")
            time.sleep(min(60, 3 * 2 ** intento))
    return None


def max_palabras(texto: str, n: int = 50) -> str:
    if texto is None:
        return None
    palabras = texto.split()
    return texto if len(palabras) <= n else " ".join(palabras[:n])


resumen = []
for c, t in zip(noticias["cuerpo"], noticias["titular"]):
    resumen.append(max_palabras(resumir(c, t)))
    time.sleep(1)  # pausa corta: sin ella OpenRouter respondió 429 (too many requests)

noticias["resumen"] = resumen
noticias["n_palabras"] = noticias["resumen"].str.split().str.len()
print(noticias["n_palabras"].describe())
for t, r in zip(noticias["titular"], noticias["resumen"]):
    print(f"- {t}\n  {r}\n")
noticias.to_csv(os.path.join(DATA_DIR, "noticias_resumen.csv"), index=False)

# --- Resultado obtenido (10 noticias, todas bajo 50 palabras) ---
# Palabras por resumen: mín 26, máx 35, promedio 32.
# Ninguno tuvo que recortarse. Pedir 40 dejó margen suficiente.
#
# 1. escenas de geometría variable (35 palabras)
#    Sánchez aplica la "geometría variable" de Zapatero para gobernar,
#    buscando apoyos diversos. Aunque Ciudadanos y otros socios parecen
#    conformes, las tensiones internas, la crisis económica o la gestión de
#    la pandemia podrían desbaratar sus planes.
#
# 2. iglesias se reunirá con junqueras sin cristal y sin límite de tiempo (33 palabras)
#    Iglesias se reunirá con Junqueras en la cárcel de Lledoners sin cristal
#    y sin límite de tiempo. Abordarán la situación de los presos catalanes y
#    buscarán el apoyo de ERC a los presupuestos.
#
# 3. bankia paga otros 219 millones al estado vía dividendo, el mayor de su historia (31 palabras)
#    Bankia paga 219 millones de euros al Estado en dividendos, el mayor
#    importe histórico. Con esto, la devolución total de ayudas públicas por
#    esta vía asciende a 961 millones de euros.
#
# 4. sánchez tiene ya una mayoría en el congreso para derogar la reforma laboral y la "ley mordaza " (26 palabras)
#    El Congreso tiene mayoría de izquierdas para derogar la reforma laboral
#    y la "ley mordaza". Sánchez podrá tramitar la ley de eutanasia y otras
#    propuestas progresistas.
#
# 5. un presidente irresponsable (35 palabras)
#    El rey emérito Juan Carlos I regulariza casi 4.5 millones de euros por
#    viajes pagados por una fundación. El presidente Sánchez critica su
#    conducta, mientras el autor defiende el gesto como un intento de
#    enmienda.
#
# 6. castro embarga el palacete y otra finca a matas para cubrir su fianza en nóos (30 palabras)
#    El juez Castro embarga el palacete y otra finca de Jaume Matas para
#    cubrir su fianza de 3,5 millones en el caso Nóos. Otros acusados también
#    ven sus bienes embargados.
#
# 7. el jefe del ejecutivo, dispuesto a reunirse con pablo iglesias si se lo pide (35 palabras)
#    Rajoy se muestra dispuesto a reunirse con Pablo Iglesias si este lo
#    solicita, destacando que habla con todos. Atribuye la falta de encuentro
#    a sus diferentes ámbitos de trabajo y cuestiona la viabilidad de
#    Podemos.
#
# 8. el último choque con puigdemont y las diferencias entre defensas la cuenta atrás para el juicio al procés (34 palabras)
#    El juicio del procés se acerca, con diferencias entre las defensas y
#    tensiones independentistas. El gobierno busca apoyo para los
#    presupuestos, mientras Puigdemont lanza un nuevo partido. La política y
#    la justicia se entrelazan.
#
# 9. interior jubila al policía que perdió la visión tras los disturbios del procés» (31 palabras)
#    Un policía de 42 años, herido gravemente en los disturbios del procés en
#    Barcelona, ha sido jubilado por el Ministerio del Interior debido a la
#    pérdida de visión y otras secuelas.
#
# 10. otegi sobre bateragune "ni nos domaron, ni nos doblaron, ni nos van a domesticar" (30 palabras)
#    El Tribunal Supremo reabrirá el juicio del caso Bateragune. Arnaldo
#    Otegi reacciona con firmeza: "Ni nos domaron, ni nos doblaron, ni nos
#    van a domesticar". EH Bildu analizará la decisión.

# %% =========================================================================
# CUÉNTANOS SOBRE UN PROYECTO
# ============================================================================
# Observatorio de Legislación de Género (https://legislaciongenero.cl).
# Partió como cooperación técnica entre CAF y la Cámara de Diputadas y
# Diputados; cuando terminó lo seguí por mi cuenta. Está en producción y se
# actualiza diariamente con datos del Senado.
#
# Problema:
# De los ~18.700 proyectos de ley presentados desde 1990, identificar cuáles
# tienen que ver con género, y clasificar los que van ingresando cada día. El
# problema de fondo es que no hay una definición operativa de "ley de género"
# ni datos etiquetados.
#
# Estrategia:
# 1. Primero reglas: un léxico por ejes temáticos sobre el título y el texto.
#    Simple y explicable, pero con mucho ruido ("paridad" agarraba paridad de
#    precios de combustibles, "cuidado" leyes de mascotas). Salieron 2.170.
# 2. Después un LLM (Claude Sonnet) con el criterio escrito en el prompt:
#    cuatro grupos (tema central, componente de género, relación débil,
#    fuera) y salida JSON con decisión, grupo, confianza y un motivo que
#    parte por "Objeto:" (qué regula el proyecto). Alrededor puse controles:
#    - 50 casos de control con respuesta conocida como test de regresión. Si
#      una versión del criterio baja de 90%, no corre. Sonnet sacó 49/50 y
#      Haiku 47/50.
#    - Los casos límite (relación débil o confianza media/baja) van a una cola
#      que reviso yo antes de publicar. Lo que valido vuelve al modelo como
#      precedente (buscado por similitud de título).
#    - Cada registro guarda quién decidió (regla, modelo o persona) y con qué
#      versión del criterio.
#    Quedaron ~1.400 proyectos. En la última revisión coincidí con el modelo
#    en 113 de 139 casos difíciles (81%).
# 3. Ahora estoy armando un clasificador de ML usando las decisiones ya
#    validadas como datos etiquetados.
#
# Aprendizajes:
# - El criterio pesa más que el modelo. La coincidencia pasó de 24% a 81%
#   casi solo por ir corrigiendo las reglas a partir de los desacuerdos.
# - El LLM también se deja llevar por las palabras (incluía una nacionalidad
#   por gracia porque nombraba madres e hijas). Se arregló obligándolo a decir
#   primero qué regula el proyecto.
# - Importa cómo se equivoca, no solo cuánto. Dos modelos con aciertos
#   parecidos: uno fallaba con confianza alta y otro con media. Con una cola
#   que filtra por confianza solo sirve el segundo.
# - Revisar el texto de entrada: una mala conversión de Word antiguos (tildes
#   como letras cirílicas) sesgaba hacia excluir y no se veía en las métricas.
# - Consejo: partir simple para entender dónde está la ambigüedad, poner a una
#   persona en los casos dudosos y medir siempre contra etiquetas que no hizo
#   el modelo.
#
# Otra experiencia relevante, aunque no de texto: un visualizador territorial
# (piloto en Viña del Mar y Quilpué) que cruza microdatos del Censo 2024 a
# nivel de manzana con datos de seguridad (CEAD) y catastro del SII, con un
# pipeline reproducible en R.
