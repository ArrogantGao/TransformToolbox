#include "transtb.hpp"
#include <complex>
#include <vector>
#include <cmath>
#include <finufft.h>
#include <iostream>
#include <random>
#include <omp.h>
#include <chrono>
#include <fstream>

using namespace std;

void nufft(vector<complex<double>>& coeffs, vector<double>& x, vector<double>& y, vector<double>& z, vector<complex<double>>& c, const finufft_plan& p) {
    const int N = x.size();
    finufft_setpts(p, N, x.data(), y.data(), z.data(), 0, NULL, NULL, NULL);
    finufft_execute(p, c.data(), coeffs.data());
}

int main(){
    omp_set_num_threads(1);

    auto Ms = {1, 8, 64, 512, 4096, 32768};
    auto Ns = {8, 16, 32, 64};

    int samples = 20;

    ofstream file("../data/nudft_vs_nufft_time.csv");
    file << "M,N,finufft,nudft_s3,nudft_s2" << endl;

    for (int N : Ns){
        for (int M : Ms){
            cout << "--------------------------------" << endl;
            cout << "M = " << M << ", N = " << N << endl;
            vector<double> x(M);
            vector<double> y(M);
            vector<double> z(M);
            vector<complex<double>> c(M);

            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dis(-1, 1);

            for (int i = 0; i < M; i++){
                x[i] = dis(gen);
                y[i] = dis(gen);
                z[i] = dis(gen);
                c[i] = complex<double>(dis(gen), dis(gen));
            }

            vector<complex<double>> coeffs(N * N * N);

            int64_t nmodes[3] = {N, N, N};
            finufft_plan p;
            finufft_makeplan(1, 3, nmodes, 1, 1, 1e-4, &p, NULL);

            auto start_finufft = chrono::high_resolution_clock::now();
            for (int i = 0; i < samples; i++){
                nufft(coeffs, x, y, z, c, p);
            }
            auto end_finufft = chrono::high_resolution_clock::now();
            auto duration_finufft = chrono::duration_cast<chrono::nanoseconds>(end_finufft - start_finufft);
            cout << "Time taken by finufft: " << duration_finufft.count() / 1000 / samples << " microseconds" << endl;
            
            auto start_nudft_s3 = chrono::high_resolution_clock::now();
            for (int i = 0; i < samples; i++){
                transtb::nudft3d1(M, x.data(), y.data(), z.data(), c.data(), 1, N, N, N, coeffs.data());
            }
            auto end_nudft_s3 = chrono::high_resolution_clock::now();
            auto duration_nudft_s3 = chrono::duration_cast<chrono::nanoseconds>(end_nudft_s3 - start_nudft_s3);
            cout << "Time taken by nudft_s3: " << duration_nudft_s3.count() / 1000 / samples << " microseconds" << endl;

            auto start_nudft_s2 = chrono::high_resolution_clock::now();
            for (int i = 0; i < samples; i++){
                transtb::nudft3d1_s2(M, x.data(), y.data(), z.data(), c.data(), 1, N, N, N, coeffs.data());
            }
            auto end_nudft_s2 = chrono::high_resolution_clock::now();
            auto duration_nudft_s2 = chrono::duration_cast<chrono::nanoseconds>(end_nudft_s2 - start_nudft_s2);
            cout << "Time taken by nudft_s2: " << duration_nudft_s2.count() / 1000 / samples << " microseconds" << endl;

            // write the coeffs to a file
            file << M << "," << N << "," << duration_finufft.count() / samples << "," << duration_nudft_s3.count() / samples << "," << duration_nudft_s2.count() / samples << endl;
        }
    }

    file.close();

    return 0;
}