#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "transtb.hpp"
#include "doctest.h"
#include <complex>
#include <vector>
#include <cmath>
#include <finufft.h>
#include <iostream>
#include <random>

TEST_CASE("Test nudft3d1") {
    int M = 1;

    auto Ns = {6, 7, 8};
    auto flags = {1, -1};

    for (int N1 : Ns){
        for (int N2 : Ns){
            for (int N3 : Ns){
                for (int iflag : flags){
                    std::vector<double> x(M);
                    std::vector<double> y(M);
                    std::vector<double> z(M);
                    std::vector<std::complex<double>> c(M);

                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<> dis(-1, 1);
                    
                    for (int i = 0; i < M; i++){
                        x[i] = dis(gen);
                        y[i] = dis(gen);
                        z[i] = dis(gen);
                        c[i] = std::complex<double>(dis(gen), dis(gen));
                    }

                    std::vector<std::complex<double>> f_s3(N1 * N2 * N3), f_s2(N1 * N2 * N3), f_s1(N1 * N2 * N3), f_ref(N1 * N2 * N3), f_direct(N1 * N2 * N3);

                    finufft3d1(M, x.data(), y.data(), z.data(), c.data(), iflag, 1e-6, N1, N2, N3, f_ref.data(), nullptr);

                    transtb::nudft3d1_direct(M, x.data(), y.data(), z.data(), c.data(), iflag, N1, N2, N3, f_direct.data());

                    transtb::nudft3d1_s2(M, x.data(), y.data(), z.data(), c.data(), iflag, N1, N2, N3, f_s2.data());
                    
                    transtb::nudft3d1(M, x.data(), y.data(), z.data(), c.data(), iflag, N1, N2, N3, f_s3.data());

                    for (int i = 0; i < N1 * N2 * N3; i++){
                        CHECK(abs(f_s3[i] - f_ref[i]) < 1e-4);
                        CHECK(abs(f_s2[i] - f_ref[i]) < 1e-4);
                        CHECK(abs(f_ref[i] - f_direct[i]) < 1e-4);
                    }
                }
            }
        }
    }
}