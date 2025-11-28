// lab3.cpp
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

const int    N = 12288;      // Размер системы (делится на 1,2,4,8,16)
const double EPS = 1e-5;       // Требуемая точность
const int    MAX_ITERS = 10000;      // Максимальное число итераций

// Скалярное произведение: (u, v) = sum u_i * v_i
double dot_product(const std::vector<double>& u, const std::vector<double>& v) {
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += u[i] * v[i];
    }
    return sum;
}

// Норма ||u||_2 = sqrt(sum u_i^2)
double norm2(const std::vector<double>& u) {
    return std::sqrt(dot_product(u, u));
}

// Умножение матрицы A на вектор x: y = A * x
// Матрица A: диагональ = 2.0, все остальные элементы = 1.0
// То есть A_ii = 2, A_ij = 1 для i != j
void matvec_naive(const std::vector<double>& x, std::vector<double>& y) {
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) {
            double a_ij = (j == i) ? 2.0 : 1.0;
            s += a_ij * x[j];
        }
        y[i] = s;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: lab3 <variant: 1|2> [num_threads]\n";
        return 1;
    }

    int variant = std::atoi(argv[1]);
    if (variant != 1 && variant != 2) {
        std::cout << "Variant must be 1 or 2\n";
        return 1;
    }

    // Если передано число потоков — задаём его
    int user_threads = 0;
    if (argc >= 3) {
        user_threads = std::atoi(argv[2]);
        if (user_threads > 0) {
            omp_set_num_threads(user_threads);
        }
    }

    // Чтобы потом узнать фактическое число потоков
    int used_threads = 1;
#pragma omp parallel
    {
#pragma omp single
        used_threads = omp_get_num_threads();
    }

    // Векторы
    std::vector<double> x(N, 0.0);       // начальное приближение x0 = 0
    std::vector<double> b(N, 0.0);       // правая часть
    std::vector<double> r(N, 0.0);       // остаток r^n
    std::vector<double> z(N, 0.0);       // направление поиска z^n
    std::vector<double> r_new(N, 0.0);   // новый остаток r^{n+1}
    std::vector<double> Az(N, 0.0);      // A * z^n
    std::vector<double> Ax(N, 0.0);      // A * x^n

    // Формирование правой части b в зависимости от варианта
    if (variant == 1) {
        // Модельная задача с заданным решением: b_i = N + 1, x* = (1,1,...,1)
        for (int i = 0; i < N; ++i) {
            b[i] = static_cast<double>(N + 1);
        }
    }
    else if (variant == 2) {
        // Модельная задача с произвольным решением:
        // u_i = sin(2*pi*i/N), b = A*u
        const double PI = 3.14159265358979323846;
        std::vector<double> u(N, 0.0);
        for (int i = 0; i < N; ++i) {
            u[i] = std::sin(2.0 * PI * static_cast<double>(i) / static_cast<double>(N));
        }
        matvec_naive(u, b);
    }

    // Синхронизация и старт таймера
#pragma omp barrier
    double t0 = omp_get_wtime();

    // r^0 = b - A x^0, z^0 = r^0
    matvec_naive(x, Ax); // x0 = 0, так что можно было бы пропустить, но для аналогии с Python версией оставим
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        r[i] = b[i] - Ax[i];
        z[i] = r[i];
    }

    double bnorm = norm2(b);
    if (bnorm == 0.0) {
        bnorm = 1.0;
    }

    double rel = norm2(r) / bnorm;
    int    iters = 0;

    while (rel > EPS && iters < MAX_ITERS) {
        // Az = A * z^n
        matvec_naive(z, Az);

        double rr = dot_product(r, r);      // (r^n, r^n)
        double denom = dot_product(Az, z);     // (A z^n, z^n)

        if (denom == 0.0) {
            // Матрица выродилась для текущего направления, выходим
            break;
        }

        double alpha = rr / denom;

        // x^{n+1} = x^n + alpha * z^n
        // r^{n+1} = r^n - alpha * A z^n
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            x[i] += alpha * z[i];
            r_new[i] = r[i] - alpha * Az[i];
        }

        double rr_new = dot_product(r_new, r_new); // (r^{n+1}, r^{n+1})
        double beta = rr_new / rr;

        // z^{n+1} = r^{n+1} + beta * z^n
        // r^n <- r^{n+1}
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            z[i] = r_new[i] + beta * z[i];
            r[i] = r_new[i];
        }

        ++iters;
        rel = std::sqrt(rr_new) / bnorm;
    }

#pragma omp barrier
    double t1 = omp_get_wtime();

    // Вывод результатов (аналогично Python-версии)
    std::cout << "variant = " << variant << "\n";
    std::cout << "threads = " << used_threads << "\n";
    std::cout << "iterations = " << iters << "\n";
    std::cout << "relative_residual = " << std::fixed;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(10);
    std::cout << rel << "\n";
    std::cout.setf(std::ios::fmtflags(0), std::ios::floatfield);
    std::cout << "time_sec = " << (t1 - t0) << "\n";

    return 0;
}

