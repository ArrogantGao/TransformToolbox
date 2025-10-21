// type-1 and type-2 nudft in 3D
// following the definition in https://finufft.readthedocs.io/en/latest/c.html#id2

#ifndef NUDFT_HPP
#define NUDFT_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <iostream>

using namespace std;

namespace transtb {

    template<typename T>
    void nudft3d1_direct(const int M, T* x, T* y, T* z, complex<T>* c, const int iflag, const int N1, const int N2, const int N3, complex<T>* f){
        for (int n = 0; n < N3; n++){
            // cout << "n = " << n << "kz = " << (n - N3 / 2) << endl;
                for (int m = 0; m < N2; m++){
                    for (int l = 0; l < N1; l++){
                    double kx = (l - N1 / 2);
                    double ky = (m - N2 / 2);
                    double kz = (n - N3 / 2);

                    for (int i = 0; i < M; i++){
                        auto xi = x[i];
                        auto yi = y[i];
                        auto zi = z[i];
                        auto ci = c[i];
                        f[n * N2 * N1 + m * N1 + l] += ci * exp(complex<T>(0, iflag * (kx * xi + ky * yi + kz * zi)));
                    }
                }
            }
        }
    }

    template <typename T>
    inline void nudft3d1(const int M,
                                const T* __restrict x,
                                const T* __restrict y,
                                const T* __restrict z,
                                const std::complex<T>* __restrict c,
                                const int iflag,
                                const int N1, const int N2, const int N3,
                                std::complex<T>* __restrict f) {

        using complex_t = std::complex<T>;
        const T iflag_sign = (iflag > 0) ? 1 : -1;

        std::vector<complex_t> x_cache(N1);
        std::vector<complex_t> y_cache(N2);
        std::vector<complex_t> z_cache(N3);

        const T kx_min = - N1 / 2;
        const T ky_min = - N2 / 2;
        const T kz_min = - N3 / 2;

        // #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; ++i) {
            const T xi = x[i];
            const T yi = y[i];
            const T zi = z[i];
            const complex_t ci = c[i];

            const complex_t exp_x_step = std::exp(complex_t(0, iflag_sign * xi));
            const complex_t exp_y_step = std::exp(complex_t(0, iflag_sign * yi));
            const complex_t exp_z_step = std::exp(complex_t(0, iflag_sign * zi));

            x_cache[0] = std::exp(complex_t(0, iflag_sign * xi * kx_min)) * ci;
            for (int l = 1; l < N1; ++l)
                x_cache[l] = x_cache[l - 1] * exp_x_step;

            y_cache[0] = std::exp(complex_t(0, iflag_sign * yi * ky_min));
            for (int m = 1; m < N2; ++m)
                y_cache[m] = y_cache[m - 1] * exp_y_step;

            z_cache[0] = std::exp(complex_t(0, iflag_sign * zi * kz_min));
            for (int n = 1; n < N3; ++n)
                z_cache[n] = z_cache[n - 1] * exp_z_step;

            for (int n = 0; n < N3; ++n) {
                const complex_t z_val = z_cache[n];
                for (int m = 0; m < N2; ++m) {
                    const complex_t zy_val = z_val * y_cache[m];
                    complex_t* __restrict fptr = f + (n * N2 + m) * N1;
                    #pragma omp simd
                    for (int l = 0; l < N1; ++l)
                        fptr[l] += zy_val * x_cache[l];
                }
            }
        }
    }

    template<typename T>
    void nudft3d1_iterate(const int M, T* x, T* y, T* z, complex<T>* c, const int iflag, const int N1, const int N2, const int N3, complex<T>* f){
        T iflag_sign = iflag > 0 ? 1 : -1;

        for (int i = 0; i < M; i++){
            auto xi = x[i];
            auto yi = y[i];
            auto zi = z[i];
            auto ci = c[i];

            auto exp_x0 = exp(complex<T>(0, iflag_sign * xi));
            auto exp_y0 = exp(complex<T>(0, iflag_sign * yi));
            auto exp_z0 = exp(complex<T>(0, iflag_sign * zi));
            
            double kx_min = - N1 / 2;
            double ky_min = - N2 / 2;
            double kz_min = - N3 / 2;

            auto exp_xmin= exp(complex<T>(0, iflag_sign * xi * (kx_min))) * ci;
            auto exp_ymin = exp(complex<T>(0, iflag_sign * yi * (ky_min)));
            auto exp_zmin = exp(complex<T>(0, iflag_sign * zi * (kz_min)));

            auto exp_z = exp_zmin;
            for (int n = 0; n < N3; n++){
                auto exp_y = exp_ymin;
                for (int m = 0; m < N2; m++){
                    auto exp_x = exp_xmin;
                    for (int l = 0; l < N1; l++){
                        f[n * N2 * N1 + m * N1 + l] += exp_z * exp_y * exp_x;
                        exp_x *= exp_x0;
                    }
                    exp_y *= exp_y0;
                }
                exp_z *= exp_z0;
            }
        }
    }

    // slice the output array as 2D planes when computing
    template<typename T>
    void nudft3d1_s2(const int M, T* x, T* y, T* z, complex<T>* c, const int iflag, const int N1, const int N2, const int N3, complex<T>* f){
        std::vector<complex<T>> x_cache(N1);
        std::vector<complex<T>> y_cache(N2);

        T iflag_sign = iflag > 0 ? 1 : -1;

        for (int n = 0; n < N3; n++){
            for (int i = 0; i < M; i++){
                auto xi = x[i];
                auto yi = y[i];
                auto zi = z[i];
                auto ci = c[i];

                auto exp_x0 = exp(complex<T>(0, iflag_sign * xi));
                auto exp_y0 = exp(complex<T>(0, iflag_sign * yi));

                double kx_min = - N1 / 2;
                double ky_min = - N2 / 2;
                double kz_min = - N3 / 2;

                x_cache[0] = exp(complex<T>(0, iflag_sign * xi * (kx_min))) * ci;
                y_cache[0] = exp(complex<T>(0, iflag_sign * yi * (ky_min)));

                for (int l = 1; l < N1; l++){
                    x_cache[l] = x_cache[l - 1] * exp_x0;
                }

                for (int m = 1; m < N2; m++){
                    y_cache[m] = y_cache[m - 1] * exp_y0;
                }

                auto exp_zn = exp(complex<T>(0, iflag_sign * zi * (kz_min + n)));

                complex<T> temp_y, temp_xy;
                for (int m = 0; m < N2; m++){
                    temp_y = y_cache[m];
                    for (int l = 0; l < N1; l++){
                        temp_xy = temp_y * x_cache[l];
                        f[n * N2 * N1 + m * N1 + l] += temp_xy * exp_zn;
                    }
                }
            }
        }
    }

    template<typename T>
    void nudft3d1_s1(const int M, T* x, T* y, T* z, complex<T>* c, const int iflag, const int N1, const int N2, const int N3, complex<T>* f){
        std::vector<complex<T>> z_cache(N3);

        T iflag_sign = iflag > 0 ? 1 : -1;

        for (int l = 0; l < 2 * N1; l++){
            for (int m = 0; m < 2 * N2; m++){
                for (int i = 0; i < M; i++){
                    auto xi = x[i];
                    auto yi = y[i];
                    auto zi = z[i];
                    auto ci = c[i];

                    auto exp_z0 = exp(complex<T>(0, iflag_sign * zi / 2));

                    for (int n = -N3; n < N3; n++){
                        z_cache[n + N3] = pow(exp_z0, n);
                    }

                    complex<T> temp_x, temp_xy;
                    temp_x = ci * exp(complex<T>(0, iflag_sign * xi * (l - N1) / 2));
                    temp_xy = temp_x * exp(complex<T>(0, iflag_sign * yi * (m - N2) / 2));
                    for (int n = 0; n < 2 * N3; n++){
                        f[l * 4 * N2 * N3 + m * 2 * N3 + n] += temp_xy * z_cache[n];
                    }
                }
            }
        }
    }
}

#endif