#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

const int    N = 12288;
const double EPS = 0.00001;
const int    MAX_ITERS = 10000;

double dot_product(const std::vector<double>& u, const std::vector<double>& v) {
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += u[i] * v[i];
    }
    return sum;
}

double norm2(const std::vector<double>& u) {
    return std::sqrt(dot_product(u, u));
}

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
    int user_threads = 0;
    if (argc >= 3) {
        user_threads = std::atoi(argv[2]);
        if (user_threads > 0) {
            omp_set_num_threads(user_threads);
        }
    }
    int used_threads = 1;
#pragma omp parallel
    {
#pragma omp single
        used_threads = omp_get_num_threads();
    }
    const double PI = 3.14159265358979323846;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) {
        x[i] = std::sin(4.0 * PI * static_cast<double>(i) / static_cast<double>(N));
    }
    std::vector<double> b(N, 0.0);
    std::vector<double> r(N, 0.0);
    std::vector<double> z(N, 0.0);
    std::vector<double> r_new(N, 0.0);
    std::vector<double> Az(N, 0.0);
    std::vector<double> Ax(N, 0.0);
    if (variant == 1) {
        for (int i = 0; i < N; ++i) {
            b[i] = static_cast<double>(N + 1);
        }
    }
    else if (variant == 2) {
        std::vector<double> u(N, 0.0);
        for (int i = 0; i < N; ++i) {
            u[i] = std::sin(2.0 * PI * static_cast<double>(i) / static_cast<double>(N));
        }
        matvec_naive(u, b);
    }
    double t0 = omp_get_wtime();
    matvec_naive(x, Ax);
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
        matvec_naive(z, Az);
        double rr = dot_product(r, r);   
        double denom = dot_product(Az, z);  
        if (denom == 0.0) {
            break;
        }
        double alpha = rr / denom;          
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            x[i] += alpha * z[i];       
            r_new[i] = r[i] - alpha * Az[i]; 
        }
        double rr_new = dot_product(r_new, r_new); 
        double beta = rr_new / rr;               
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            z[i] = r_new[i] + beta * z[i]; 
            r[i] = r_new[i];               
        }
        ++iters;
        rel = std::sqrt(rr_new) / bnorm;   
    }
    double t1 = omp_get_wtime();
    std::cout << "variant = " << variant << "\n";
    std::cout << "threads = " << used_threads << "\n";
    std::cout << "iterations = " << iters << "\n";
    std::cout.setf(std::ios::fixed);
    std::cout.precision(10);
    std::cout << "relative_residual = " << rel << "\n";
    std::cout.setf(std::ios::fmtflags(0), std::ios::floatfield);
    std::cout << "time_sec = " << (t1 - t0) << "\n";
    return 0;
}
