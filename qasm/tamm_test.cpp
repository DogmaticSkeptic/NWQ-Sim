#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include <tamm/tamm.hpp>
#include <itensor/all.h>

using Cplx = std::complex<double>;

int main(int argc, char* argv[]) {
    // Initialize TAMM
    tamm::initialize(argc, argv);
    tamm::ProcGroup pg = tamm::ProcGroup::create_world_coll();

    // Generate 20 evenly spaced problem sizes between 1 and 5000 inclusive
    std::vector<size_t> sizes;
    sizes.reserve(20);
    const size_t N_min = 1;
    const size_t N_max = 4000;
    const int num_N = 10;
    const double stepN = double(N_max - N_min) / double(num_N - 1);
    for(int idx = 0; idx < num_N; ++idx) {
        double val = double(N_min) + idx * stepN;
        sizes.push_back(size_t(std::round(val)));
    }
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    // Generate 4 evenly spaced tile sizes between 1 and 100 inclusive
    std::vector<size_t> tile_sizes;
    tile_sizes.reserve(4);
    const size_t T_min = 60;
    const size_t T_max = 160;
    const int num_T = 4;
    const double stepT = double(T_max - T_min) / double(num_T - 1);
    for(int idx = 0; idx < num_T; ++idx) {
        double val = double(T_min) + idx * stepT;
        tile_sizes.push_back(size_t(std::round(val)));
    }
    std::sort(tile_sizes.begin(), tile_sizes.end());
    tile_sizes.erase(std::unique(tile_sizes.begin(), tile_sizes.end()), tile_sizes.end());

    std::unordered_map<size_t,double> itensor_times;
    for(size_t N : sizes) {
        double duration = std::numeric_limits<double>::quiet_NaN();
        try {
            // Use new Index(dim, name) constructor to avoid deprecation warnings
            itensor::Index I(int(N), "I");
            itensor::Index K(int(N), "K");
            itensor::Index J(int(N), "J");
            itensor::ITensor A(I, K), B(K, J);
            // Fill A and B with 1.0
            for(int ii = 1; ii <= int(N); ++ii) {
                for(int kk = 1; kk <= int(N); ++kk) {
                    A.set(I(ii), K(kk), 1.0);
                }
            }
            for(int kk = 1; kk <= int(N); ++kk) {
                for(int jj = 1; jj <= int(N); ++jj) {
                    B.set(K(kk), J(jj), 1.0);
                }
            }
            // Time the contraction C = A * B
            auto t0 = std::chrono::high_resolution_clock::now();
            itensor::ITensor C = A * B;
            auto t1 = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
            if(pg.rank().value() == 0) {
                std::cout << "ITensor: N=" << N << ", time=" << duration << " s\n";
            }
        }
        catch(std::exception const& e) {
            if(pg.rank().value() == 0) {
                std::cerr << "ITensor contraction for N=" << N
                          << " threw exception: " << e.what() << "\n";
            }
            duration = std::numeric_limits<double>::quiet_NaN();
        }
        itensor_times[N] = duration;
    }

    // Define TAMM distribution settings
    std::vector<std::pair<tamm::DistributionKind,std::string>> dist_settings = {
        {tamm::DistributionKind::nw, "nw"},
        {tamm::DistributionKind::dense, "dense"}
    };

    std::ofstream csv("tensor_times.csv");
    csv << "distribution,tile,N,cpu_time,gpu_time,itensor_time\n";
    if(pg.rank().value() == 0) {
        std::cout << "distribution,tile,N,cpu_time,gpu_time,itensor_time\n";
    }

    // Loop over TAMM distribution kinds
    for(const auto& [dist_kind, dist_name] : dist_settings) {
        tamm::ExecutionContext ec{pg, dist_kind, tamm::MemoryManagerKind::ga};
        if(!ec.has_gpu() && pg.rank().value() == 0) {
            std::cerr << "Warning (" << dist_name
                      << "): no GPU detected; gpu_time entries will be zero\n";
        }
        tamm::ExecutionHW cpu_hw = tamm::ExecutionHW::CPU;
        tamm::ExecutionHW gpu_hw = ec.exhw();
        bool use_dense = (dist_kind == tamm::DistributionKind::dense);

        for(size_t tile_val : tile_sizes) {
            for(size_t N : sizes) {
                tamm::Tile tile = static_cast<tamm::Tile>(std::min<size_t>(tile_val, N));
                tamm::TiledIndexSpace tis{tamm::IndexSpace{tamm::range(N)}, tile};
                auto [i,k,j] = tis.labels<3>("all");

                tamm::Tensor<Cplx> A{i,k};
                tamm::Tensor<Cplx> B{k,j};
                tamm::Tensor<Cplx> C{i,j};
                if(use_dense) {
                    A.set_dense();
                    B.set_dense();
                    C.set_dense();
                }

                tamm::Scheduler sch{ec};
                sch.allocate(A,B,C).execute();
                sch(A() = Cplx{1.0, 0.0})(B() = Cplx{1.0, 0.0})(C() = Cplx{0.0, 0.0}).execute();

                // TAMM CPU timing
                auto t0 = std::chrono::high_resolution_clock::now();
                sch(C(i,j) += A(i,k) * B(k,j)).execute(cpu_hw, false);
                auto t1 = std::chrono::high_resolution_clock::now();
                double cpu_time = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

                sch(C() = Cplx{0.0, 0.0}).execute();

                // TAMM GPU timing
                double gpu_time = 0.0;
                if(ec.has_gpu()) {
                    auto t2 = std::chrono::high_resolution_clock::now();
                    sch(C(i,j) += A(i,k) * B(k,j)).execute(gpu_hw, false);
                    auto t3 = std::chrono::high_resolution_clock::now();
                    gpu_time = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();
                }

                // Lookup ITensor time
                double it_time = std::numeric_limits<double>::quiet_NaN();
                auto it_it = itensor_times.find(N);
                if(it_it != itensor_times.end()) {
                    it_time = it_it->second;
                }

                csv << dist_name << "," << tile_val << "," << N << ","
                    << cpu_time << "," << gpu_time << ",";
                if(std::isnan(it_time)) {
                    csv << "NA\n";
                } else {
                    csv << it_time << "\n";
                }

                if(pg.rank().value() == 0) {
                    std::cout << dist_name << "," << tile_val << "," << N << ","
                              << cpu_time << "," << gpu_time << ",";
                    if(std::isnan(it_time)) {
                        std::cout << "NA\n";
                    } else {
                        std::cout << it_time << "\n";
                    }
                }

                sch.deallocate(A,B,C).execute();
            }
        }
    }

    csv.close();
    tamm::finalize();
    return 0;
}

