import time
import random
import threading
from threading import Thread, Lock

n = 500
threads_finished = 0
total_threads = 0
lock = Lock()

A = [[random.randint(0, 9) for _ in range(n)] for _ in range(n)]
B = [[random.randint(0, 9) for _ in range(n)] for _ in range(n)]
"""s
def print_matrix(matrix, name, size_to_print=15):
    print(f"\nМатрица {name} (первые {size_to_print}x{size_to_print} элементов):")
    for i in range(min(size_to_print, len(matrix))):
        row = matrix[i][:size_to_print]
        print(row)

print_matrix(A, "A")
print_matrix(B, "B")
"""

def multiply_standard(A, B, n):
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def multiply_by_columns(A, B, n):
    C = [[0] * n for _ in range(n)]
    for j in range(n):
        for i in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

print("\nВыполняется стандартное умножение")
start_time = time.time()
C_standard = multiply_standard(A, B, n)
end_time = time.time()
time_standard = end_time - start_time
print(f"Время выполнения стандартного умножения: {time_standard:.4f} секунд")

print("\nВыполняется умножение по столбцам")
start_time = time.time()
C_columns = multiply_by_columns(A, B, n)
end_time = time.time()
time_columns = end_time - start_time
print(f"Время выполнения умножения по столбцам: {time_columns:.4f} секунд")

"""
# Выводим результаты (только первые 10x10 элементов)
print_matrix(C_standard, "C (стандартное умножение)")
print_matrix(C_columns, "C (умножение по столбцам)")
"""

# Функция для вычисления подматрицы одним потоком
def compute_submatrix(A, B, C, start_row, end_row, start_col, end_col, thread_id):
    global threads_finished
    print(f"Поток {thread_id}: вычисление подматрицы [{start_row}:{end_row}, {start_col}:{end_col}]")
    for i in range(start_row, end_row):
        for j in range(start_col, end_col):
            total = 0
            for k in range(len(A)):
                total += A[i][k] * B[k][j]
            C[i][j] = total
    with lock:
        threads_finished += 1
        print(f"Поток {thread_id} завершил работу. Завершено потоков: {threads_finished}/{total_threads}")

# Функция параллельного умножения с заданным числом потоков
def multiply_parallel(A, B, n, num_threads):
    global threads_finished, total_threads
    threads_finished = 0
    total_threads = num_threads
    C = [[0] * n for _ in range(n)]
    threads = []
    rows_per_thread = n // num_threads
    extra_rows = n % num_threads
    thread_id = 0
    current_row = 0
    for i in range(num_threads):
        block_rows = rows_per_thread + (1 if i < extra_rows else 0)
        end_row = current_row + block_rows
        t = Thread(target=compute_submatrix, args=(A, B, C, current_row, end_row, 0, n, thread_id))
        threads.append(t)
        thread_id += 1
        current_row = end_row
    start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end_time = time.time()
    return C, end_time - start_time
"""
# Функция для сравнения матриц 
def matrices_equal(C1, C2, n):
    for i in range(n):
        for j in range(n):
            if C1[i][j] != C2[i][j]:
                print(f"Несоответствие в [{i}][{j}]: C1={C1[i][j]}, C2={C2[i][j]}")
                return False
    return True
"""

if __name__ == "__main__":
    thread_counts = [1, 2, 4, 8]
    for num_threads in thread_counts:
        print(f"\nЗапуск с {num_threads} потоками...")
        C_parallel, time_parallel = multiply_parallel(A, B, n, num_threads)
        print(f"Время выполнения с {num_threads} потоками: {time_parallel:.4f} секунд")
        """
        # Вывод фрагмента результата
        print_matrix(C_parallel, f"C (параллельно, {num_threads} потоков)")

        # Проверка корректности 
        correct = matrices_equal(C_standard, C_parallel, n)
        print(f"Результат корректен: {correct}")
        """
