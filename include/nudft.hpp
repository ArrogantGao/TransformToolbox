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

    template<typename T>
    void nudft3d1(const int M, T* x, T* y, T* z, complex<T>* c, const int iflag, const int N1, const int N2, const int N3, complex<T>* f){
        std::vector<complex<T>> x_cache(N1);
        std::vector<complex<T>> y_cache(N2);
        std::vector<complex<T>> z_cache(N3);

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

            x_cache[0] = exp(complex<T>(0, iflag_sign * xi * (kx_min))) * ci;
            y_cache[0] = exp(complex<T>(0, iflag_sign * yi * (ky_min)));
            z_cache[0] = exp(complex<T>(0, iflag_sign * zi * (kz_min)));

            for (int l = 1; l < N1; l++){
                x_cache[l] = x_cache[l - 1] * exp_x0;
            }

            for (int m = 1; m < N2; m++){
                y_cache[m] = y_cache[m - 1] * exp_y0;
            }

            for (int n = 1; n < N3; n++){
                z_cache[n] = z_cache[n - 1] * exp_z0;
            }

            complex<T> temp_z, temp_zy;
            for (int n = 0; n < N3; n++){
                temp_z = z_cache[n];
                for (int m = 0; m < N2; m++){
                    temp_zy = temp_z * y_cache[m];
                    for (int l = 0; l < N1; l++){
                        f[n * N2 * N1 + m * N1 + l] += temp_zy * x_cache[l];
                    }
                }
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