import numpy as np


# Datos de entrenamiento
# Cada fila contiene:
# [humedad del suelo, temperatura ambiental]

X = np.array([
    [80, 18],
    [70, 22],
    [65, 28],
    [55, 25],
    [50, 32],
    [40, 30],
    [35, 25],
    [30, 32],
    [20, 35],
    [10, 38]
], dtype=float)


# Respuestas esperadas
# 0 = no regar
# 1 = regar

y = np.array([
    [0],
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1],
    [1]
], dtype=float)


# Aca inicia la normalizacion
# Escala máxima esperada para cada entrada
# Humedad: 0 a 100
# Temperatura: 0 a 50

escala = np.array([100, 50])


# Normalizamos cada columna
X_normalizado = X / escala


print("\nDatos normalizados:")
print(X_normalizado)


print("Datos de entrenamiento:")
print(X)

print("\nRespuestas esperadas:")
print(y)

# Función de activación sigmoide
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# Función para entrenar la neurona
def entrenar_neurona(tasa_aprendizaje, epocas):

    # Pesos iniciales
    pesos = np.zeros((2, 1))

    # Sesgo inicial
    sesgo = 0.0

    # Entrenamiento
    for epoca in range(epocas):

        # Suma ponderada
        z = X_normalizado @ pesos + sesgo

        # Probabilidad mediante la sigmoide
        probabilidad = sigmoide(z)

        # Error
        error = probabilidad - y

        # Gradiente respecto a z
        gradiente_z = error * probabilidad * (1 - probabilidad)

        # Gradiente de los pesos
        gradiente_pesos = X_normalizado.T @ gradiente_z

        # Gradiente del sesgo
        gradiente_sesgo = np.sum(gradiente_z)

        # Actualización de pesos y sesgo
        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

    # Probabilidades finales
    z_final = X_normalizado @ pesos + sesgo
    probabilidades = sigmoide(z_final)

    # Error cuadrático medio
    error_final = np.mean((probabilidades - y) ** 2)

    # Respuestas binarias
    respuestas = (probabilidades >= 0.5).astype(int)

    # Cantidad de respuestas correctas
    aciertos = np.sum(respuestas == y)

    return pesos, sesgo, probabilidades, error_final, aciertos

# Entrenamiento base
tasa_aprendizaje = 0.5
epocas = 10000

pesos, sesgo, probabilidades, error_final, aciertos = entrenar_neurona(
tasa_aprendizaje,epocas)

# Resultados del entrenamiento

print("\n--- RESULTADOS DEL ENTRENAMIENTO ---")

print("\nPeso de la humedad:")
print(pesos[0][0])

print("\nPeso de la temperatura:")
print(pesos[1][0])

print("\nSesgo:")
print(sesgo)


# Calculamos nuevamente las probabilidades finales
z_final = X_normalizado @ pesos + sesgo
probabilidades = sigmoide(z_final)

# Convertimos las probabilidades en 0 o 1
respuestas = (probabilidades >= 0.5).astype(int)


print("\nProbabilidades finales:")
for i in range(len(X)):
    print(
        f"Caso {i + 1}: "
        f"Humedad={X[i][0]:.0f}% | "
        f"Temperatura={X[i][1]:.0f}°C | "
        f"Probabilidad={probabilidades[i][0]:.4f} | "
        f"Respuesta={respuestas[i][0]} | "
        f"Esperada={int(y[i][0])}"
    )


# Error cuadrático medio final
error_final = np.mean((probabilidades - y) ** 2)

print("\nError cuadrático medio final:")
print(error_final)


# Pruebas con condiciones nuevas

nuevos_datos = np.array([
    [75, 30],
    [45, 34],
    [25, 22],
    [50, 25],
    [30, 40]
], dtype=float)


# Normalizamos utilizando la misma escala
nuevos_datos_normalizados = nuevos_datos / escala


# Calculamos la suma ponderada
z_nuevos = nuevos_datos_normalizados @ pesos + sesgo


# Calculamos las probabilidades
probabilidades_nuevas = sigmoide(z_nuevos)


# Convertimos las probabilidades a 0 o 1
respuestas_nuevas = (probabilidades_nuevas >= 0.5).astype(int)


print("\n--- PRUEBAS CON CONDICIONES NUEVAS ---")

for i in range(len(nuevos_datos)):
    print(
        f"Humedad={nuevos_datos[i][0]:.0f}% | "
        f"Temperatura={nuevos_datos[i][1]:.0f}°C | "
        f"Probabilidad={probabilidades_nuevas[i][0]:.4f} | "
        f"Decisión={respuestas_nuevas[i][0]}"
    )

# Experimentos solicitados en el taller

experimentos = [
    ("Base", 10000, 0.5),
    ("Pocas épocas", 100, 0.5),
    ("Épocas intermedias", 1000, 0.5),
    ("Muchas épocas", 20000, 0.5),
    ("Tasa pequeña", 10000, 0.01),
    ("Tasa moderada", 10000, 0.1),
    ("Tasa alta", 10000, 1.0),
    ("Tasa muy alta", 10000, 2.0)
]

print("\n--- EXPERIMENTOS ---")

for nombre, epocas_exp, tasa_exp in experimentos:

    # Entrenamos la neurona con los parámetros del experimento
    pesos_exp, sesgo_exp, probabilidades_exp, error_exp, aciertos_exp = entrenar_neurona(
        tasa_exp,
        epocas_exp
    )

    # Probabilidad para el caso de 45% de humedad y 34°C
    dato_experimento = np.array([[45, 34]], dtype=float)
    dato_experimento_normalizado = dato_experimento / escala

    z_experimento = dato_experimento_normalizado @ pesos_exp + sesgo_exp
    probabilidad_45_34 = sigmoide(z_experimento)[0][0]

    print(f"\nExperimento: {nombre}")
    print(f"Épocas: {epocas_exp}")
    print(f"Tasa de aprendizaje: {tasa_exp}")
    print(f"Error final: {error_exp:.6f}")
    print(f"Aciertos: {int(aciertos_exp)}/10")
    print(f"Probabilidad para 45% y 34°C: {probabilidad_45_34:.4f}")


# Comparación de diferentes umbrales

umbrales = [0.4, 0.5, 0.6]

print("\n--- COMPARACIÓN DE UMBRALES ---")

for umbral in umbrales:

    respuestas_umbral = (probabilidades >= umbral).astype(int)

    aciertos_umbral = np.sum(respuestas_umbral == y)

    print(f"\nUmbral: {umbral}")
    print(f"Aciertos: {int(aciertos_umbral)}/10")

    for i in range(len(X)):
        print(
            f"Caso {i + 1}: "
            f"Probabilidad={probabilidades[i][0]:.4f} | "
            f"Respuesta={respuestas_umbral[i][0]} | "
            f"Esperada={int(y[i][0])}"
        )
