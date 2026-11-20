# Pregunta 4 – Cuéntanos sobre un proyecto: Notas de Campo: clasificación y priorización editorial de noticias ambientales con reglas + LLM

## Contexto

- **Proyecto:** Notas de Campo, la línea editorial asistida por IA que uso para generar contenido de LinkedIn e Instagram para ICTA (mi consultora) y para mi marca personal.
- **Mi rol:** diseñé la solución completa: arquitectura, taxonomía, reglas, criterios de evaluación y flujo de revisión. El código lo construí en conjunto con asistentes de IA, iterando y corrigiendo.
- **Stack:**
  - TypeScript sobre Cloudflare Workers + D1 (SQLite), desplegado y en uso continuo;
  - Workers AI (Llama 3.3 70B) para evaluar;
  - la API de Claude para redactar borradores.

## 1. Problema de negocio

- **Situación:** para alimentar las redes de ICTA y las mías sigo unas 25 fuentes de noticias ambientales (RSS y sitios web). Revisarlas a mano me tomaba varias horas a la semana, y mi criterio para elegir qué publicar no era constante: cambiaba según el día.
- **Tarea a automatizar:**
  1. clasificar cada artículo en una taxonomía de **26 categorías** (agua, sequía, incendios, SEIA, minería…);
  2. priorizarlo;
  3. sugerir una **ruta editorial**: señal, post técnico, opinión, caso aplicado, boletín o revisión manual.
- **Restricción:** *nada se publica solo.* El sistema clasifica, evalúa y propone; yo apruebo.
- **Éxito:** menos horas de revisión y más contenido publicado de forma regular.

## 2. Estrategia utilizada

**Dos capas, de lo barato y explicable a lo costoso:**

1. **Clasificación por reglas:** palabras clave con sinónimos y **exclusiones** para evitar falsos positivos, más un puntaje y una ruta sugerida. Es explicable, sin costo y filtra el volumen. En una corrida de 265 artículos, 128 quedaron para revisión manual.
2. **Evaluación con LLM en 13 dimensiones**, por ejemplo relevancia territorial, novedad, necesidad de verificación y riesgo reputacional:
   - el modelo responde un **JSON validado contra un esquema** (enteros de 0 a 5);
   - el **total se calcula en código**, no lo decide el modelo;
   - si la salida no valida, el artículo **cae a revisión manual** y queda registrado.

   Después, el LLM propone ángulos y Claude redacta borradores editables.

**Decisiones clave:**
- **Reglas antes que LLM,** como línea base y filtro.
- **Un modelo por tarea:** uno barato para evaluar, que es una tarea estructurada, y uno más fuerte en español para redactar.
- **Tope de costo diario** de evaluaciones.
- **Revisión humana obligatoria** en un panel de aprobación antes de publicar.

## 3. Reflexión y aprendizaje

**Tres problemas reales me obligaron a cambiar el diseño:**
- **El LLM devolvía JSON mal formado o con valores fuera de rango.** Lo resolví forzando un esquema, validando en código y enviando lo inválido a revisión manual.
- **La evaluación podía disparar costos o agotar la cuota.** Lo resolví con un tope diario.
- **Con solo título y resumen, los borradores salían genéricos.** Tuve que entregarle a Claude el texto completo de forma transitoria.

La lección: **la salida de un LLM es un dato no confiable.** Hay que validarla, acotarla y tener una ruta de escape hacia una persona. Es la misma lógica que veo para codificar respuestas abiertas: automatizar lo seguro y derivar lo dudoso.

**Lo que haría distinto:** juzgué el éxito por el tiempo ahorrado y el contenido publicado, pero nunca etiqueté una muestra para medir la calidad de la clasificación. Sin eso no puedo cuantificar cuánto aporta el LLM sobre las reglas.

**Consejo para quien empieza:**
1. definir la métrica y una muestra etiquetada antes de elegir el modelo;
2. construir una línea base simple;
3. diseñar desde el inicio qué pasa cuando el modelo falla o duda.

_Generado automáticamente el 20-11-2026 desde pregunta4_proyecto.ipynb_