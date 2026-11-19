# Aspectos generales

Esta prueba ha sido diseñada con el objetivo de evaluar sus conocimientos para el cargo de científico/a de datos. La prueba tendrá una duración máxima de **90 minutos**. El tiempo comenzará a correr desde el momento en que terminemos de revisar las instrucciones. Las respuestas (código y desarrollo escrito) deben entregarse en un archivo con extensión **.R, .ipynb o .py**. El nombre del archivo debe seguir la siguiente estructura *nombre_apellido.extension*. Para faciltar la revisión es deseable que indique en su archivo cuál es la pregunta que está respondiendo.        

**Importante**: El archivo debe ser entregado como un *pull request* a la rama master dentro de [este](https://github.com/inesscc/prueba-tecnica) repositorio. Para ello, genere una rama con su nombre y apellido, de la siguiente manera **nombre_apellido**. En caso de que trabaje en python, agregue un archivo **requirements.txt** con las dependencias. Si tiene dificultades para trabajar con el repositorio, podrá entregarlo vía correo en formato .zip, asumiendo un descuento.     

**El uso de herramientas de inteligencia artificial no está prohibido. Si decide trabajar con IA, hágalo de manera reflexiva. Las respuestas de la IA contienen patrones que son fácilmente detectables y que no siempre se adecúan a las especificidades de la tarea que se está desarrollando. Nos interesa conocer su razonamiento sobre los problemas propuestos, de modo que sus comentarios recibirán mucha atención al momento de la evaluación. En varias secciones de la prueba el código valdrá tanto como las justificaciones de sus decisiones**    sdfsdfsdf  



## Problemas

### Paso 0 (1 minuto)

Clone el siguiente repositorio.

https://github.com/inesscc/prueba-tecnica.git

git@github.com:inesscc/prueba-tecnica.git

En él encontrará las instrucciones para la prueba y algunos archivos que tendrá que utilizar para responder las preguntas. 


### Clasificando texto (44 minutos)

Usted cuenta con un conjunto de datos sobre características del empleo de las personas. Cada uno de estos empleos está asociado a un código. A continuación, se muestra un ejemplo de algunos datos:     

| id | actividad | oficio | tareas | cod_final |
|---|---|---|---|---|
| 0 | producción uva exportación | temporera agrícola | mantenimiento de parronales | 01 |
| 1 | corporación municipal | profesor mecánico industrial | preparar clases, enseñar | 84 |
| 4 | contratista de rrhh para obras de construcción | trabajador de construcción | veredas | 41 |
| 5 | municipalidad de puerto natales | técnico paramedico | toma de signos vitales, entrega de horas a pacientes, tratamientos inyectables etc. | 84 |
| 7 | guardias seguridad y proteccion | administrador de contrato | administrar dirigir contratar personal | 80 |


**IMPORTANE: El objetivo es construir un clasificador de texto para aprender a predecir la variables *cod_final*. Usted cuenta con poco tiempo. No destine esfuerzos a la optimización del resultado**

Para ello usted cuenta con una partición *train* y otra *test*. Los datos ya han sido preparados y se encuentran en las tablas **train** y **test** en una base de datos **mysql**. Los datos de la conexión son los siguientes:

- host="sql.freedb.tech"
- port=3306
- user="u_y8tlWZ"
- password="wSkxw3xbhsQg"
- database="freedb_N4nnid0a"

En python el paquete *mysql-connector-python* facilita la interacción con la base de datos, sin embargo, puede utilizar el paquete que usted prefiera.  


Realice las siguientes tareas:

1) Descargue la información de la tabla train desde la base de datos
2) Entrene un clasificador de texto. Incorpore el procesamiento de datos y el algoritmo que le parezcan más idóneos para la tarea.     
3) Descargue la información de la tabla test desde la base de datos
4) Haga predicciones sobre el set de testeo
5) Analice los resultados, indicando tanto las fortalezas como los espacios de mejora del modelo.  
6) Proponga (sin ejecutarlo) un plan para mejorar el desempeño de su modelo.   

### Modelo en producción (20 minutos)

Su modelo ha sido puesto en producción. Cuenta con muestras mensuales de 100 registros cada una, que han sido etiquetadas manualmente para algunos periodos posteriores a la puesta en producción. Eso le permite obtener algunas métricas. A continuación, se muestran los resultados del desempeño del modelo a lo largo del tiempo. 

- **Testing**: 
    * Acc: 0.938 
    * Macro F1: 0.823 

- **Mes 3**:  
    * Acc: 0.923 
    * Macro F1: 0.812 
 
- **Mes 6** 
    * Acc: 0.913 
    * Macro F1: 0.796 
 
- **Mes 10** 
    * Acc: 0.83
    * Macro F1: 0.71 

**Sugiera alguna hipótesis para explicar el comportamiento de los datos y elabore una estrategia (sin implementarla) para mitigar el problema. Justique con sus propias palabras cada uno de sus argumentos**  


### Procesando texto (10 minutos)

Usted tiene que procesar datos de texto provenientes de noticias en medios de prensa. Para cada noticia, usted debe generar un resumen que no supere las 50 palabras. Los datos se encuentran en la tabla *noticias* de la base de datos. Corresponde a un dataset con 10 filas. A continuación, se muestra un ejemplo de una fila:     

| cuerpo | titular |
|---|---|
| el presidente del gobierno, pedro sánchez, comparecerá este martes en el palacio de la moncloa tras la reunión del consejo de ministros para informar del plan de recuperación, transformación y resiliencia de la economía tras la pandemia de coronavirus. este plan se analizará en una reunión posterior a la del consejo que celebrará la comisión creada al efecto y de la que forman parte todos los miembros del ejecutivo. tras informar en su comparecencia ante los medios, sánchez presentará el plan este miércoles ante el pleno del congreso, y será aprobado en una próxima reunión del consejo de ministros para ser enviado a bruselas antes del 30 de abril. | pedro sánchez comparece para explicar su plan de recuperación |

Usted cuenta con una API key para utilizar un servicio llamado OpenRouter, que permite hacer inferencias de grandes modelos de lenguaje. Este [link](https://openrouter.ai/docs/quickstart) dirije a la documentación oficial, donde encontrará un ejemplo de cómo usar el endpoint *responses* a través de *python*. 

Considere las siguientes condiciones:
- La API key se encuentra en [este archivo](https://1drv.ms/t/c/3ba457a6b989f5e0/IQDYUbsfa5KwRabKDXrJB9GGAY6PvDqj2Vhr9u0H_xcoNRg?e=3TGQPe)  
- Debe utilizar el modelo **google/gemini-2.5-flash-lite** para resolver la tarea.
- Debe desactivar el razonamiento, para minimizar costos
- El resultado está contenido en el *text* de un elemento *output* 

```   
 response.json()["output"][0]["content"][0]["text"]
```
**Genere un código que ejecute las siguientes tareas**:

1) Descargue la tabla *noticias* de la base de datos
2) Genere una columna o lista *resumen* que contenga el resumen de cada texto.    


### Cuéntanos sobre un proyecto (15 minutos)


Nos interesa conocer su experiencia trabajando con clasificadores de texto, arquitectura RAG u otro uso de LLM para el procesamiento de texto.  

En su respuesta refiérase, al menos, a los siguientes aspectos:
- Problema de negocio que estaba tratando de resolver (¿Cuál es la tarea que se quiere automatizar?) 
- Estrategia utilizada para resolverlo (¿Cuál es el enfoque, modelo o arquitectura que se utilizó?) 
- Reflexión y aprendizaje (¿Qué consejo darías a alguien que está comenzado con un problema similar?)




**¡Mucho éxito!**


