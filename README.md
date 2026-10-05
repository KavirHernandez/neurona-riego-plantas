# Actividad Neurona Artificial para el Riego de Plantas

## Descripción

Este proyecto implementa una neurona artificial sencilla en Python para decidir si una planta necesita riego.

La neurona utiliza dos variables de entrada:

- Humedad del suelo (%)
- Temperatura ambiente (°C)

La salida es binaria:

- `1` = regar
- `0` = no regar

El proyecto utiliza NumPy, normalización de datos y una función de activación sigmoide.

## Requisitos

- Python
- `uv`
- NumPy

## Instalación

Clonar o descargar el repositorio y entrar en la carpeta del proyecto.

Luego ejecutar:

```bash
uv sync
```

## Ejecución

Para ejecutar el programa:

```bash
uv run main.py
```

El programa muestra:

1. Datos de entrenamiento.
2. Datos normalizados.
3. Resultados del entrenamiento base.
4. Pesos y sesgo aprendidos.
5. Probabilidades y decisiones para los datos de entrenamiento.
6. Error cuadrático medio final.
7. Predicciones para cinco condiciones nuevas.
8. Ocho experimentos con diferentes épocas y tasas de aprendizaje.
9. Comparación de los umbrales 0.4, 0.5 y 0.6.

## Datos de entrenamiento

Se utilizaron los 10 casos indicados en la actividad:

| Caso | Humedad (%) | Temperatura (°C) | Esperada |
|---:|---:|---:|---:|
| 1 | 80 | 18 | 0 |
| 2 | 70 | 22 | 0 |
| 3 | 65 | 28 | 0 |
| 4 | 55 | 25 | 0 |
| 5 | 50 | 32 | 0 |
| 6 | 40 | 30 | 1 |
| 7 | 35 | 25 | 1 |
| 8 | 30 | 32 | 1 |
| 9 | 20 | 35 | 1 |
| 10 | 10 | 38 | 1 |

## Normalización

Para el entrenamiento se utilizan los datos normalizados:

- Humedad: `humedad / 100`
- Temperatura: `temperatura / 50`

Se utiliza la misma escala para los datos nuevos.

La normalización permite trabajar con valores de magnitud más comparable durante el entrenamiento.

## Configuración base

La configuración principal utilizada fue:

- Épocas: `10000`
- Tasa de aprendizaje: `0.5`
- Umbral de decisión: `0.5`

### Resultados

- Peso de humedad: `-33.27729484064782`
- Peso de temperatura: `0.3467428040501618`
- Sesgo: `14.748383350549062`
- Error cuadrático medio: `0.005377904234075987`
- Aciertos: `10/10`

El peso negativo de la humedad indica que una mayor humedad tiende a disminuir la probabilidad de riego. El peso positivo de la temperatura indica que una mayor temperatura tiende a aumentar dicha probabilidad.

## Pruebas con condiciones nuevas

| Humedad | Temperatura | Probabilidad | Decisión |
|---:|---:|---:|---:|
| 75% | 30°C | 0.0000 | 0 |
| 45% | 34°C | 0.5023 | 1 |
| 25% | 22°C | 0.9986 | 1 |
| 50% | 25°C | 0.1523 | 0 |
| 30% | 40°C | 0.9936 | 1 |

La condición de `45%` de humedad y `34°C` es especialmente cercana al umbral, con una probabilidad de `0.5023`.

## Experimentos

| Experimento | Épocas | Tasa de aprendizaje | Error final | Aciertos | P(45%, 34°C) |
|---|---:|---:|---:|---:|---:|
| Base | 10000 | 0.5 | 0.005378 | 10/10 | 0.5023 |
| Pocas épocas | 100 | 0.5 | 0.089365 | 10/10 | 0.5597 |
| Épocas intermedias | 1000 | 0.5 | 0.030556 | 10/10 | 0.5688 |
| Muchas épocas | 20000 | 0.5 | 0.002800 | 10/10 | 0.4877 |
| Tasa pequeña | 10000 | 0.01 | 0.067082 | 10/10 | 0.5751 |
| Tasa moderada | 10000 | 0.1 | 0.019562 | 10/10 | 0.5470 |
| Tasa alta | 10000 | 1.0 | 0.002800 | 10/10 | 0.4877 |
| Tasa muy alta | 10000 | 2.0 | 0.001397 | 10/10 | 0.4731 |

### Interpretación de los experimentos

Todos los experimentos lograron `10/10` aciertos en los datos de entrenamiento. Sin embargo, el error final fue diferente.

Con 100 épocas el error fue `0.089365`, mientras que con 10000 épocas bajó a `0.005378`. Esto muestra que más entrenamiento permitió un mejor ajuste.

Al aumentar de 10000 a 20000 épocas, el error bajó a `0.002800`, aunque la mejora fue menor que la observada al pasar de pocas épocas a muchas.

La tasa de aprendizaje de `0.01` produjo un error mayor (`0.067082`), por lo que el aprendizaje fue más lento en este experimento.

Las tasas de `1.0` y `2.0` funcionaron correctamente con este conjunto de datos. La tasa `2.0` obtuvo el menor error (`0.001397`). En estos experimentos no se observó inestabilidad.

## Comparación de umbrales

Se probaron los umbrales `0.4`, `0.5` y `0.6` sin volver a entrenar la neurona.

| Umbral | Aciertos |
|---:|---:|
| 0.4 | 10/10 |
| 0.5 | 10/10 |
| 0.6 | 10/10 |

Los tres umbrales clasificaron correctamente los 10 ejemplos de entrenamiento.

Sin embargo, el umbral sí afecta las condiciones nuevas. Por ejemplo, para `45%` de humedad y `34°C`, la probabilidad es `0.5023`:

- Umbral 0.4 → `1`
- Umbral 0.5 → `1`
- Umbral 0.6 → `0`

Cambiar el umbral no modifica los pesos ni el sesgo porque no se realiza un nuevo entrenamiento. Solamente cambia la regla utilizada para convertir la probabilidad en una decisión binaria.

## Análisis
Análisis
### 1.	¿Por qué fue necesario normalizar la humedad y la temperatura?
RTA: La humedad está expresada entre aproximadamente 10 y 80, mientras que la temperatura está entre 18 y 38. Al dividir la humedad entre 100 y la temperatura entre 50, ambas variables quedan en escalas más pequeñas y comparables.
Esto ayuda a que el entrenamiento de la neurona sea más estable y facilita que los pesos se ajusten adecuadamente.

### 2.	¿En qué operaciones se utilizó X_normalizado y para qué se conservó X?
RTA: X_normalizado se utiliza durante el entrenamiento para calcular la suma ponderada:
z = X_normalizado @ pesos + sesgo
También se utiliza para calcular el gradiente de los pesos. Se conserva X porque contiene los valores originales de humedad y temperatura, que son más fáciles de interpretar cuando mostramos los resultados, por ejemplo: Humedad=40% / Temperatura=30°C
Esto coincide con la indicación del taller de utilizar X_normalizado para los cálculos de la neurona y conservar X para interpretar los resultados
### 3.	¿Qué ocurrió al utilizar solamente 100 épocas?
RTA: Con 100 épocas obtuvimos:
•	Error: 0.089365 
•	Aciertos: 10/10 
•	Probabilidad para 45% y 34 °C: 0.5597 
Aunque consiguió clasificar correctamente los 10 casos, el error fue mucho mayor que con 10.000 épocas (0.005378). Por eso se puede decir que 100 épocas fueron suficientes para clasificar este pequeño conjunto de datos, pero el modelo todavía no había alcanzado un ajuste tan bueno como después de más entrenamiento.
### 4.	¿Más épocas siempre produjeron una mejora importante?
RTA: No necesariamente. Cuando pasa de 10.000 a 20.000 épocas, el error bajó: 0.005378 - 0.002800 Por lo tanto, sí hubo una mejora, pero no fue tan grande como la diferencia entre 100 y 10.000 épocas. Eso muestra que aumentar las épocas puede mejorar el ajuste, pero llega un punto en que las mejoras son cada vez menores.
### 5.	¿Qué efecto tuvo una tasa de aprendizaje demasiado pequeña?
RTA: Con una tasa de aprendizaje de 0.01 obtuvimos:
•	Error: 0.067082 
•	Aciertos: 10/10 
•	Probabilidad 45%/34°C: 0.5751 
El modelo consiguió clasificar correctamente los ejemplos, pero presentó un error bastante mayor que la configuración base.
Por lo tanto, en este experimento la tasa 0.01 hizo que el aprendizaje fuera más lento.

### 6.	¿Qué efecto tuvo una tasa de aprendizaje alta o muy alta?
RTA: En los resultados:
Tasa	Error
0.5	0.005378
1.0	0.002800
2.0	0.001397
En este caso concreto, aumentar la tasa de aprendizaje no produjo inestabilidad. Incluso la tasa 2.0 obtuvo el menor error. Por eso, según los resultados, debemos decir que una tasa alta funcionó bien para este conjunto de datos, aunque esto no significa que una tasa muy alta siempre vaya a funcionar bien en otros problemas.

### 7.	¿Qué representa el signo del peso correspondiente a la humedad?
RTA: El peso de la humedad fue: -33.2773 Es negativo.
Esto significa que, manteniendo las demás condiciones constantes, una mayor humedad tiende a disminuir la salida de la neurona. Tiene sentido con los datos: cuando el suelo está más húmedo, la neurona tiende a decidir no regar (que es 0).
### 8.	¿Qué representa el signo del peso correspondiente a la temperatura?
RTA: El peso de la temperatura fue: +0.3467 Es positivo.
Esto significa que una mayor temperatura tiende a aumentar la salida de la neurona y, por tanto, aumenta la probabilidad de que la decisión sea regar (que es 1).
### 9.	¿Por qué una probabilidad debe convertirse en 0 o 1 mediante un umbral?
RTA: La sigmoide produce un valor entre 0 y 1, que podemos interpretar como una probabilidad o nivel de confianza. Pero el problema necesita una decisión binaria:
•	0 = no regar 
•	1 = regar 
Por eso utilizamos un umbral. Con el umbral de 0.5:
(probabilidades >= 0.5).astype(int)
Si la probabilidad es 0.5 o superior, se decide regar; si es menor, no se riega.

### 10.	¿Qué limitaciones tiene esta neurona para representar el riego de una planta real?
RTA: La neurona utilizada en el taller es un modelo muy sencillo y solamente utiliza dos variables: humedad del suelo y temperatura ambiente. En un sistema real sería necesario considerar otros factores, como el tipo de planta, humedad ambiental, lluvia, iluminación, hora del día, tipo de suelo y características del sistema de riego. Además, los datos de entrenamiento solo son 10, por lo que no son suficientes para representar todas las situaciones que pueden ocurrir en una planta real. Por ello, este modelo sirve como demostración educativa de una neurona artificial, pero no debería utilizarse directamente para controlar un sistema real de riego sin más datos, pruebas y mecanismos de seguridad


## Estructura del proyecto

```text
neurona-riego-plantas/
│
├── main.py
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```


## Ejecución rápida

```bash
uv sync
uv run main.py
```

## Autor
Johan Kavir Hernandez Galeano
