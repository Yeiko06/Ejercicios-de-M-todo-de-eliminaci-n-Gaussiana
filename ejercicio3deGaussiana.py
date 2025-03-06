import numpy as np

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

# Definición del sistema de ecuaciones (Ejercicio 3)
A = np.array([[1, 2, -3, 4, -1, 1], [-2, 3, 5, -1, 2, -1], [4, -1, 2, 6, -3, 1],
               [-3, 5, -1, 2, 4, -1], [2, -4, 6, -5, 1, 3], [-5, 1, 4, -1, 2, -6]], dtype=float)
b = np.array([7, -2, 10, 3, -8, 5], dtype=float)

# Resolución del sistema
sol = gauss_elimination(A, b)

# Imprimir la solución
print("Solución del sistema (Ejercicio 3):")
print(sol)
