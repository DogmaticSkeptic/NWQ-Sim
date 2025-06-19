#include <chrono>
#include <iostream>
#include <tamm/tamm.hpp>

using namespace tamm;

int main(int argc, char* argv[]) {
    tamm::initialize(argc, argv);
    ProcGroup pg = ProcGroup::create_world_coll();
    ExecutionContext ec{pg, DistributionKind::nw, MemoryManagerKind::ga};
    ExecutionHW ex_hw = ec.exhw();
    if(!ec.has_gpu()) {
        if(pg.rank().value() == 0) std::cerr << "GPU not available" << std::endl;
        tamm::finalize();
        return 1;
    }
    size_t N = 1024;
    Tile tile_size = 64;
    TiledIndexSpace tis{IndexSpace{range(N)}, tile_size};
    auto [i, k, j] = tis.labels<3>("all");
    Tensor<double> A{i, k};
    Tensor<double> B{k, j};
    Tensor<double> C{i, j};
    Scheduler sch{ec};
    sch.allocate(A, B, C).execute();
    sch(A() = 1.0)(B() = 1.0)(C() = 0.0).execute();
    const auto t0 = std::chrono::high_resolution_clock::now();
    sch(C(i, j) += A(i, k) * B(k, j)).execute(ex_hw, false);
    const auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
    if(pg.rank().value() == 0) {
        std::cout << "GPU tensor contraction time (N=" << N << ", tile=" << tile_size
                  << "): " << elapsed << " seconds" << std::endl;
    }
    sch.deallocate(A, B, C).execute();
    tamm::finalize();
    return 0;
}

