#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>
using namespace std;

const double A = 0, B = 5;
double f(double x)  { return exp(-x) * sin(4 * x); }
double df(double x) { return exp(-x) * (4 * cos(4 * x) - sin(4 * x)); }

struct Spline {
    int n; double h;
    vector<double> y, M;
    Spline(int n_) : n(n_), y(n_ + 1), M(n_ + 1) {
        h = (B - A) / n;
        #pragma omp parallel for
        for (int i = 0; i <= n; i++) y[i] = f(A + i * h);

        vector<double> lo(n + 1, h / 6), di(n + 1, 2 * h / 3), up(n + 1, h / 6), r(n + 1);
        di[0] = di[n] = h / 3;
        r[0] = (y[1] - y[0]) / h - df(A);
        r[n] = df(B) - (y[n] - y[n - 1]) / h;
        for (int i = 1; i < n; i++) r[i] = (y[i + 1] - 2 * y[i] + y[i - 1]) / h;

        for (int i = 1; i <= n; i++) {
            double w = lo[i] / di[i - 1];
            di[i] -= w * up[i - 1];
            r[i]  -= w * r[i - 1];
        }
        M[n] = r[n] / di[n];
        for (int i = n - 1; i >= 0; i--) M[i] = (r[i] - up[i] * M[i + 1]) / di[i];
    }
    double operator()(double x) const {             // значение сплайна в точке x
        int i = min(n - 1, max(0, (int)((x - A) / h)));
        double xl = A + i * h, xr = xl + h;
        return M[i] * pow(xr - x, 3) / (6 * h) + M[i + 1] * pow(x - xl, 3) / (6 * h)
             + (y[i] - M[i] * h * h / 6) * (xr - x) / h
             + (y[i + 1] - M[i + 1] * h * h / 6) * (x - xl) / h;
    }
};

double maxErr(const Spline& s, long K) {
    double e = 0;
    #pragma omp parallel for reduction(max:e)
    for (long k = 0; k <= K; k++) {
        double x = A + (B - A) * k / K;
        e = max(e, fabs(s(x) - f(x)));
    }
    return e;
}

int main() {
    printf("%6s %12s %8s\n", "n", "max|S-f|", "отн.");
    double prev = 0;
    for (int n = 10; n <= 640; n *= 2) {
        Spline s(n);
        double e = maxErr(s, 200000);
        printf("%6d %12.3e %8.2f\n", n, e, prev ? prev / e : 0.0);
        prev = e;
    }

    long K = 30000000;
    Spline s(1000);
    printf("\nУскорение (K=%ld точек)\n%3s %9s %7s %7s\n", K, "p", "T, с", "S", "E");
    double T1 = 0;
    int g = omp_get_max_threads();
    for (int p = 1; p <= g; p *= 2) {
        omp_set_num_threads(p);
        double t0 = omp_get_wtime();
        double e = maxErr(s, K);
        double T = omp_get_wtime() - t0;
        if (p == 1) T1 = T;
        printf("%3d %9.4f %7.2f %7.2f   (err=%.3e)\n", p, T, T1 / T, T1 / T / p, e);
    }
}