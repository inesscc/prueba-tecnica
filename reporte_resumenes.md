# Reporte automático – Resúmenes de noticias

- Modelo: `google/gemini-2.5-flash-lite` (razonamiento desactivado)
- Noticias procesadas: 10 | palabras por resumen: media 41.0, máx 46
- Puntaje medio del evaluador (1-5): fidelidad 4.9, cobertura 4.0, claridad 5.0
- Fidelidad léxica media: 0.78 | compresión media: 0.09

## Sugerencias

1. Hay cifras en los resúmenes que no están en la noticia (noticias [4]). Revisarlas a mano y reforzar en el prompt: 'no agregues cifras ni fechas que no aparezcan'.
2. Algunos resúmenes usan bastante vocabulario ausente del original (fidelidad léxica < 0,8). No siempre es un error (puede ser parafraseo), pero conviene revisarlos primero.
3. El evaluador detecta resúmenes que omiten información central. Se puede pedir explícitamente 'qué, quién, dónde, cuándo' o generar primero una lista de hechos clave y luego el resumen.
4. Para producción: validar con una muestra de resúmenes evaluados por personas y medir el acuerdo con el juez LLM antes de confiar en él; usar un modelo juez distinto del generador.

## Detalle por noticia

| # | Titular | Palabras | Fid. léxica | Núm. inventados | Fid. | Cob. | Clar. | Problema | Sugerencia |
|---|---|---|---|---|---|---|---|---|---|
| 0 | escenas de geometría variable | 43 | 0.55 | - | 4 | 4 | 5 | Omite la comparación con la legislatura de Zapatero y el rol de UP. | Incluir brevemente la comparación con Zapatero y la incomodidad de UP para mayor contexto. |
| 1 | iglesias se reunirá con junqueras sin cristal | 41 | 0.91 | - | 5 | 5 | 5 | ninguno | ninguna |
| 2 | bankia paga otros 219 millones al estado vía  | 38 | 0.89 | - | 5 | 4 | 5 | Omite el contexto de la devolución total de ayudas públicas. | Incluir brevemente que este dividendo contribuye a la devolución de ayudas públicas. |
| 3 | sánchez tiene ya una mayoría en el congreso p | 37 | 0.83 | - | 5 | 4 | 5 | Omite la mención de otros grupos parlamentarios que suman la mayoría. | Incluir que la mayoría se forma con PSOE, Unidas Podemos, ERC y Compromís. |
| 4 | un presidente irresponsable | 40 | 0.93 | 4,5 | 5 | 4 | 5 | Omite la crítica al presidente Sánchez y la comparación con otros líderes. | Incluir la crítica a Sánchez y la comparación con otros líderes para una cobertura más completa. |
| 5 | castro embarga el palacete y otra finca a mat | 46 | 0.92 | - | 5 | 4 | 5 | Omite otros embargos y fianzas mencionados en la noticia. | Incluir brevemente que el embargo a Matas es parte de un proceso más amplio de embargos a varios acusados. |
| 6 | el jefe del ejecutivo, dispuesto a reunirse c | 39 | 0.52 | - | 5 | 3 | 5 | Omite la opinión de Rajoy sobre Podemos y Ciudadanos. | Incluir brevemente la visión de Rajoy sobre Podemos y Ciudadanos para mayor contexto. |
| 7 | el último choque con puigdemont y las diferen | 46 | 0.8 | - | 5 | 4 | 5 | Omite detalles sobre las diferencias entre defensas y la postura de Vox. | Incluir brevemente las divergencias entre las defensas y la acusación de Vox para mayor exhaustividad. |
| 8 | interior jubila al policía que perdió la visi | 41 | 0.67 | - | 5 | 4 | 5 | Omite detalles sobre la gravedad de las lesiones y el impacto en su vida personal. | Incluir que las secuelas afectan su vida diaria y le impiden realizar actividades básicas. |
| 9 | otegi sobre bateragune "ni nos domaron, ni no | 39 | 0.77 | - | 5 | 4 | 5 | omite la mención de los otros acusados y el motivo de la anulación. | Incluir brevemente los nombres de los otros acusados y el motivo de la anulación por el TEDH. |

## Resúmenes

**0. escenas de geometría variable**  
Sánchez utiliza la "geometría variable" para gobernar, buscando alianzas coyunturales con nacionalistas, separatistas y ciudadanos. Aunque UP muestra incomodidad, Cs apoya al gobierno. Otros socios negociarán su apoyo, mientras el gobierno confía en su estrategia, a pesar de posibles imprevistos económicos o judiciales.

**1. iglesias se reunirá con junqueras sin cristal y sin límite de tiempo**  
Pablo Iglesias se reunirá con Oriol Junqueras este viernes en la cárcel de Lledoners (Barcelona). La reunión, sin cristal y sin límite de tiempo, abordará la situación de los políticos catalanes presos y buscará el apoyo de ERC a los presupuestos.

**2. bankia paga otros 219 millones al estado vía dividendo, el mayor de su historia**  
Bankia pagó 219 millones de euros al Estado el 11 de abril como dividendo, el mayor de su historia. Esto eleva a 961 millones la devolución total por esta vía. El Estado posee el 61,4% de la entidad.

**3. sánchez tiene ya una mayoría en el congreso para derogar la reforma laboral y la "ley mordaza "**  
El nuevo Congreso tiene mayoría de izquierdas para derogar la reforma laboral y la "ley mordaza". Pedro Sánchez podrá tramitar la ley de eutanasia y otras propuestas progresistas, dependiendo de su voluntad política y el calendario legislativo.

**4. un presidente irresponsable**  
Don Juan Carlos ha regularizado casi 4,5 millones de euros ante Hacienda por viajes en jet privado pagados por una fundación. Anteriormente, abonó 680.000 euros por dádivas de un empresario. El presidente del gobierno, Pedro Sánchez, rechaza sus conductas incívicas.

**5. castro embarga el palacete y otra finca a matas para cubrir su fianza en nóos**  
El juez José Castro embargó el palacete y otra finca de Jaume Matas en Palma para cubrir su fianza de 3,5 millones de euros en el caso Nóos. El embargo se realizó hoy, al concluir el plazo para que los acusados informaran de sus bienes embargables.

**6. el jefe del ejecutivo, dispuesto a reunirse con pablo iglesias si se lo pide**  
Mariano Rajoy, presidente del gobierno, se mostró dispuesto a reunirse con Pablo Iglesias, líder de Podemos, si este lo solicita. Rajoy afirmó que habla con todos y atribuyó su falta de encuentro previo a sus diferentes ámbitos de trabajo.

**7. el último choque con puigdemont y las diferencias entre defensas la cuenta atrás para el juicio al procés**  
El juicio del procés comenzará la semana del 5 de febrero en el Tribunal Supremo. Doce acusados y sus defensas preparan sus estrategias, mientras las diferencias entre partidos independentistas y el gobierno marcan la cuenta atrás. El juicio se desarrollará cerca de las elecciones de mayo.

**8. interior jubila al policía que perdió la visión tras los disturbios del procés»**  
El Ministerio del Interior ha jubilado a un agente de la UIP de 42 años que perdió la visión tras ser herido por un objeto en Barcelona durante los disturbios del procés. Las lesiones le impiden continuar en el servicio policial.

**9. otegi sobre bateragune "ni nos domaron, ni nos doblaron, ni nos van a domesticar"**  
Arnaldo Otegi reaccionó al nuevo juicio ordenado por el Tribunal Supremo por pertenencia a organización terrorista, tuiteando "¡ni nos domaron, ni nos doblaron, ni nos van a domesticar!". El Supremo anuló la sentencia de 2012 por vulneración del TEDH.
