
# ==================================================
# PREGUNTA 1: CLASIFICANDO TEXTO
# ==================================================

# Objetivo: entrenar un modelo de clasificacion
# para predecir la variable cod_final a partir
# de las caracteristicas del empleo.

# datos prueba
setwd("C:/prueba_tecnica")

install.packages("arrow")
library(arrow)

# datos
train <- arrow::read_parquet("train_glosas.parquet")
test <- arrow::read_parquet("test_glosas.parquet")

# rev estruct
str(train)

# dimensiones
dim(train)
dim(test)

# nom columnas
names(train)

# datos faltantes
colSums(is.na(train))

# cuenta categoria
length(unique(train$cod_final))

# repticion codigos
sort(table(train$cod_final), decreasing = TRUE)

# Reemplazo de datos faltantes
train$actividad[is.na(train$actividad)] <- ""
train$oficio[is.na(train$oficio)] <- ""
train$tareas[is.na(train$tareas)] <- ""

test$actividad[is.na(test$actividad)] <- ""
test$oficio[is.na(test$oficio)] <- ""
test$tareas[is.na(test$tareas)] <- ""

# Uniendo textos
train$texto <- paste(train$actividad, train$oficio, train$tareas)

test$texto <- paste(test$actividad, test$oficio, test$tareas)

# resultados
head(train$texto)

# Converti variable objetivo a factor
train$cod_final <- as.factor(train$cod_final)

# Rev categorias
nlevels(train$cod_final)

# Rev distribucion
summary(train$cod_final)

# paquetes
install.packages("quanteda")
install.packages("quanteda.textmodels")

# librerias necesarias
library(quanteda)
library(quanteda.textmodels)

# Creacion de tokens
tokens_train <- tokens(train$texto,
                       remove_punct = TRUE,
                       remove_numbers = TRUE)

# Convertir minusculas
tokens_train <- tokens_tolower(tokens_train)

# Creando matriz de palabras
matriz_train <- dfm(tokens_train)

# Eliminar pblrs poco frecuentes
matriz_train <- dfm_trim(matriz_train, min_docfreq = 5)

# Rev dimensiones
dim(matriz_train)

# modelo Naive Bayes
modelo <- textmodel_nb(
  matriz_train,
  y = train$cod_final
)

# textos de prueba
tokens_test <- tokens(test$texto,
                      remove_punct = TRUE,
                      remove_numbers = TRUE)

tokens_test <- tokens_tolower(tokens_test)

#matriz de prueba
matriz_test <- dfm(tokens_test)

# mismas palabras del entrenamiento
matriz_test <- dfm_match(
  matriz_test,
  features = featnames(matriz_train)
)

# forecast
predicciones <- predict(modelo, newdata = matriz_test)

# resultados
head(predicciones)


# Comparar resultados
resultado <- data.frame(
  real = as.character(test$cod_final),
  predicho = as.character(predicciones)
)

# Calculo exactitud
accuracy <- mean(resultado$real == resultado$predicho)

accuracy


# F1 por categoria
codigos <- unique(c(as.character(test$cod_final),
                    as.character(predicciones)))

f1 <- c()

for (codigo in codigos) {
  
  vp <- sum(resultado$real == codigo & resultado$predicho == codigo)
  fp <- sum(resultado$real != codigo & resultado$predicho == codigo)
  fn <- sum(resultado$real == codigo & resultado$predicho != codigo)
  
  precision <- ifelse(vp + fp == 0, 0, vp / (vp + fp))
  recall <- ifelse(vp + fn == 0, 0, vp / (vp + fn))
  
  f1_codigo <- ifelse(precision + recall == 0, 0,
                      2 * precision * recall / (precision + recall))
  
  f1 <- c(f1, f1_codigo)
}

# Promedio de F1
mean(f1)


### Analisis de resultados ###

# Utilice un modelo Naive Bayes para clasificar los textos
# segun la variable cod_final.
# Para esto se uni las variables actividad, oficio y tareas.
#
# El modelo obtuvo una precision del 77,11%, lo que considero
# un resultado inicial bueno como primera aproximacion.
#
# Una fortaleza del modelo es que es sencillo y se aplica rapido.
# Sin embargo, existen 79 categorias y algunas tienen muy pocos
# registros, lo que puede afectar las predicciones.
#
# Como mejora, se podria realizar una mayor limpieza de los textos,
# revisar los errores por categoria y probar otros posibles modelos
# para comparar los resultados.


### Analisis de resultados ###

# Utilice un modelo Naive Bayes para clasificar los textos,
# considerando las variables actividad, oficio y tareas.
#
# El modelo obtuvo un accuracy de 77,11% y un F1-score
# promedio de 50,90%.
#
# Los resultados mostraron que el modelo logra clasificar
# correctamente gran  parte de los registros.
# Sin embargo, el F1-score es menor, lo que indica que
# existen diferencias en el rendimiento entre categorias.
#
# Una posible explicacion es que algunas categorias tienen
# muchos registros y otras tienen muy pocos.
#
# Como mejora, se podria realizar una  limpieza de
# los textos, revisar los errores por categoria y probar
# otros modelos para comparar sus resultados.



# ==================================================
# FINAL PREGUNTA 1
# ==================================================



# ==================================================
# PREGUNTA 2: MODELO EN PRODUCCION
# ==================================================

### Analisis de resultados ###
#
# al revisar los resultados, observe que el modelo
# presenta una disminucion en su rendimiento con el tiempo.
#
# inicialmente tenia un accuracy de 93,8% y un Macro F1
# de 82,3%. Sin embargo, en el mes 10 estos valores
# disminuyeron a 83,0% y 71,0%, respectivamente.
#
# una posible explicacion es que los datos utilizados
# en produccion han cambiado respecto a los datos
# con los cuales se entreno el modelo.
#
# por ejemplo, podria aparecer nuevas palabras,
# actividades laborales o formas de describir los empleos
# que el modelo no reconocera correctamente.
#
# esto permite explicar por que el modelo comienza
# a cometer mas errores a medida que pasa el tiempo.
#
# Como mejorar
#
# Primero revisaria si los registros que fueron clasificados
# incorrectamente para identificar potenciales patrones.
#
# compararia los datos actuales con los utilizados
# durante el entrenamiento, para verificar si existen
# cambios en las caracteristicas de los textos.
#
# tambien podria incorporar nuevos registros etiquetados
# al conjunto de entrenamiento, especialmente de las
# categorias donde el modelo presenta mayores errores.
#
# ademas, entrenaria nuevamente el modelo
# y compararia sus resultados con el modelo anterior.
#
# finalmente, realizaria un seguimiento mensual
# del accuracy y Macro F1 para detectar nuevas
# disminuciones en el rendimiento.

# ==================================================
# FINAL PREGUNTA 2
# ==================================================


# ==================================================
# PREGUNTA 3: PROCESANDO TEXTO
# ==================================================

# lib
library(arrow)
library(httr)
library(jsonlite)

# Directorio de trabajo
setwd("C:/prueba_tecnica")

# noticias
noticias <- arrow::read_parquet("noticias_sample.parquet")

# datos
dim(noticias)
names(noticias)

# API
api_key <- readline("Ingrese API key: ")

url <- "https://openrouter.ai/api/v1/responses"

# Funcion para noticias
resumir <- function(texto) {

  mensaje <- paste(
    "Resume la siguiente noticia en español.",
    "El resumen debe tener maximo 40 palabras.",
    "Entrega solamente el resumen:",
    texto
  )

  respuesta <- httr::POST(
    url,
    httr::add_headers(
      Authorization = paste("Bearer", api_key),
      `Content-Type` = "application/json"
    ),
    body = list(
      model = "google/gemini-2.5-flash-lite",
      reasoning = list(enabled = FALSE),
      input = mensaje
    ),
    encode = "json"
  )

  httr::stop_for_status(respuesta)

  datos <- httr::content(
    respuesta,
    as = "parsed",
    type = "application/json"
  )

  resumen <- datos$output[[1]]$content[[1]]$text

  return(resumen)
}

# Funcion para contar palabras
contar_palabras <- function(texto) {
  if (is.na(texto) || trimws(texto) == "") {
    return(0)
  }

  return(length(strsplit(trimws(texto), "\\s+")[[1]]))
}

# resumenes
noticias$resumen <- rep(NA_character_, nrow(noticias))

for (i in 1:nrow(noticias)) {

  resumen <- resumir(noticias$cuerpo[i])

  intentos <- 1

  # Repetir si supera las 50 palabras
  while (contar_palabras(resumen) > 50 && intentos < 3) {

    resumen <- resumir(
      paste(
        "Resume en maximo 35 palabras la siguiente noticia:",
        noticias$cuerpo[i]
      )
    )

    intentos <- intentos + 1
  }

  noticias$resumen[i] <- resumen
}

# resultados
print(noticias[, c("titular", "resumen")])

# Contando palabras
palabras <- sapply(noticias$resumen, contar_palabras)

print(palabras)

# Verificando limite de 50 palabras
all(palabras <= 50)

#  resultados solo si cumplen el limite
if (all(palabras <= 50)) {

  write.csv(
    noticias,
    "noticias_resumidas.csv",
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )

  print("Resumenes guardados correctamente")

} else {

  print("Algunos resumenes superan las 50 palabras")
}

# ==================================================
# ANALISIS RESULTADOS
# ==================================================

# Se utilizo el modelo Gemini 2.5 Flash Lite
# para generar resumenes de las noticias.
#
# Para esto se envio el contenido de cada noticia
# mediante la API de OpenRouter.
#
# Se solicito un resumen breve y se desactivo
# el razonamiento para reducir costos.
#
# Los resultados se guardaron en una nueva columna
# llamada resumen.
#
# Finalmente, se reviso la cantidad de palabras
# para verificar que los resumenes cumplieran
# con el maximo de 50 palabras solicitado.

# ==================================================
# FINAL PREGUNTA 3
# ==================================================


# ==================================================
# PREGUNTA 4: COMENTARIOS SOBRE UN PROYECTO
# ==================================================


# Problema
#
# Un ejemplo de proyecto es la clasificacion automatica
# de descripciones de empleos en distintas categorias.
#
# El objetivo es reducir el trabajo manual de revisar
# textos y asignar un codigo a cada registro.



# Estrategia
#
# Para abordar este problema se puede utilizar un
# modelo de clasificacion de texto como Naive Bayes.
#
# Primero se revisan y preparan los datos, luego
# se unen las variables de texto y se transforman
# en una matriz de palabras.
#
# Posteriormente se entrena el modelo y se evalua
# su rendimiento utilizando accuracy y F1-score.
#


# Aprendizaje
#
# Uno de los principales aprendizajes es que la
# calidad y cantidad de datos influye en los resultados.
#
# Tambien es importante revisar la distribucion de
# las categorias, ya que algunas pueden tener
# pocos registros y ser mas dificiles de clasificar.
#
# Como recomendacion, comenzaria con un modelo
# sencillo, evaluaria sus resultados y posteriormente
# probaria mejoras de manera gradual.

# ==================================================
# FINAL PREGUNTA 4
# ==================================================














