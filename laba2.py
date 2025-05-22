import numpy as np
import matplotlib.pyplot as plt
from sympy import isprime # для проверки на простые числа

def read_matrix_from_file(filename, N): # Чтение матрицы из файла
    with open(filename, 'r') as f:
        matrix = []
        for _ in range(N):
            row = list(map(float, f.readline().split()))
            matrix.append(row)
    return np.array(matrix, dtype=float)

def split_matrix(A): # Разделение матрицы на подматрицы B, C, D, E
    n = A.shape[0] // 2
    B = A[:n, :n]
    C = A[n:, :n]
    D = A[n:, n:]
    E = A[:n, n:]
    return B, C, D, E

def count_primes_in_odd_columns(B): # Подсчет простых чисел в нечетных столбцах
    count = 0
    for j in range(0, B.shape[1], 2):  # Нечетные столбцы (0-based)
        for i in range(B.shape[0]):
            num = int(round(B[i, j]))
            if num >= 2 and isprime(num):
                count += 1
    return count

def sum_in_even_rows(B): # Сумма чисел в четных строках
    total = 0
    for i in range(1, B.shape[0], 2):  # Четные строки (1, 3, ...)
        total += np.sum(B[i, :])
    return total

def main():
    K = float(input("Введите K: "))
    N = int(input("Введите N (четное): "))
    if N % 2 != 0:
        print("N должно быть четным!")
        return
    try: # Чтение матрицы из файла
        A = read_matrix_from_file('matrix.txt', N)
    except FileNotFoundError:
        print("Файл matrix.txt не найден!")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return
    print("\nМатрица A:")
    print(A)

    n = N // 2 # Разделение матрицы
    B, C, D, E = split_matrix(A)

    primes_count = count_primes_in_odd_columns(B) # Проверка условий
    sum_even = sum_in_even_rows(B)

    F = A.copy() # Создание матрицы F

    if primes_count > sum_even:
        F[:n, :n], F[:n, n:] = F[:n, n:], F[:n, :n].copy() # Симметричный обмен B и E
        print("\nУсловие: простых в нечетных столбцах B > суммы в четных строках B")
        print("Выполнен симметричный обмен B и E")
    else:
        F[n:, :n], F[:n, n:] = F[:n, n:], F[n:, :n].copy() # Несимметричный обмен C и E
        print("\nУсловие: простых в нечетных столбцах B <= суммы в четных строках B")
        print("Выполнен несимметричный обмен C и E")

    print("\nМатрица F после преобразований:")
    print(F)

    det_A = np.linalg.det(A) # Вычисление выражений
    sum_diag_F = np.trace(F)

    print(f"\nОпределитель A: {det_A}")
    print(f"Сумма диагональных элементов F: {sum_diag_F}")

    if det_A > sum_diag_F:
        print("\nУсловие: det(A) > sum(diag(F))")
        A_inv = np.linalg.inv(A)
        result = A_inv @ A.T - K * np.linalg.inv(F)
        print("Результат: A⁻¹·Aᵀ - K·F⁻¹")
    else:
        print("\nУсловие: det(A) <= sum(diag(F))")
        A_inv = np.linalg.inv(A)
        G = np.tril(A)  # Нижняя треугольная матрица
        result = (A_inv + G - np.linalg.inv(F)) * K
        print("Результат: (A⁻¹ + G - F⁻¹)·K")

    print("\nРезультат вычислений:")
    print(result)

    plt.figure(figsize=(15, 5)) # Построение графиков

    plt.subplot(1, 3, 1) # Тепловая карта матрицы F
    plt.imshow(F, cmap='coolwarm')
    plt.colorbar()
    plt.title('Тепловая карта F')

    plt.subplot(1, 3, 2) # График значений первой строки F
    plt.plot(F[0], 'r-', marker='o')
    plt.title('Первая строка F')
    plt.grid(True)

    plt.subplot(1, 3, 3) # Диаграмма средних значений столбцов
    plt.bar(range(N), np.mean(F, axis=0))
    plt.title('Средние по столбцам F')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
