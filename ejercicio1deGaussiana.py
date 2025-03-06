import numpy as np
import matplotlib.pyplot as plt

def gauss_elimination(A, b):
    n = len(b)
    for i in range(n):
        # Pivoteo parcial
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
        
        # Eliminación hacia adelante
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    # Sustitución regresiva
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

# Definición del sistema de ecuaciones (Ejercicio 1)
A = np.array([[3, 2, -1, 4], [5, -3, 2, -1], [-1, 4, -2, 3], [2, -1, 3, 5]], dtype=float)
b = np.array([10, 5, -3, 8], dtype=float)

# Copia de los valores originales de A y b (para cálculo de error)
A_original = A.copy()
b_original = b.copy()

# Resolución del sistema
sol = gauss_elimination(A, b)

# Calcular el residuo (error absoluto)
residuo = np.dot(A_original, sol) - b_original
error_absoluto = np.abs(residuo)

# Imprimir la solución
print("Solución del sistema (Ejercicio 1):")
print(sol)

# Graficar el error absoluto
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(error_absoluto) + 1), error_absoluto, color='red', alpha=0.7)
plt.xlabel("Ecuación")
plt.ylabel("Error absoluto")
plt.title("Error en cada ecuación del sistema")
plt.show()
