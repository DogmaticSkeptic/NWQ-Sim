#include <chrono>
#include <complex>
#include <iostream>
#include <vector>
#include <tamm/tamm.hpp>

int main(int argc, char* argv[]) {
    using Cplx = std::complex<double>;
    tamm::initialize(argc, argv);
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();
    int nranks = world_pg.size().value();
    std::cout << "Number of ranks: " << nranks << "\n";

    std::vector<size_t> tasks;
    for(int i = 1; i <= 20; i++) {
        tasks.push_back(static_cast<size_t>(i) * 64);
    }
    int ntasks = static_cast<int>(tasks.size());

    int subranks = std::max(1, nranks / ntasks);
    tamm::ProcGroup task_pg = tamm::ProcGroup::create_subgroups(world_pg, subranks);
    tamm::ExecutionContext ec_par{task_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch_par{ec_par};
    tamm::AtomicCounterGA ac{world_pg, 1};
    ac.allocate(0);

    int64_t next = -1;
    if(task_pg.rank().value() == 0) {
        next = ac.fetch_add(0, 1);
    }
    task_pg.broadcast(&next, 0);

    auto t0_par = std::chrono::high_resolution_clock::now();
    while(next < ntasks) {
        int64_t task_id = next;
        size_t N = tasks[static_cast<size_t>(task_id)];
        tamm::Tile bt = static_cast<tamm::Tile>(std::min(N, size_t(64)));
        tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(N)}, bt};
        tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
        auto [l, b, r] = bond.labels<3>("all");
        auto [p1, p2] = phys.labels<2>("all");

        tamm::Tensor<Cplx> A({l, p1, b});
        tamm::Tensor<Cplx> B({b, p2, r});
        tamm::Tensor<Cplx> C({l, p1, p2, r});
        A.set_dense();
        B.set_dense();
        C.set_dense();

        sch_par.allocate(A, B, C).execute();
        sch_par(A() = Cplx{1.0, 0.0})(B() = Cplx{1.0, 0.0})(C() = Cplx{0.0, 0.0}).execute();

        auto t0_task_par = std::chrono::high_resolution_clock::now();
        sch_par(C(l, p1, p2, r) = A(l, p1, b) * B(b, p2, r)).execute(ec_par.exhw(), false);
        auto t1_task_par = std::chrono::high_resolution_clock::now();
        double time_task_par =
            std::chrono::duration_cast<std::chrono::duration<double>>(t1_task_par - t0_task_par).count();
        if(task_pg.rank().value() == 0) {
            std::cout << "Parallel task " << task_id << " N=" << N
                      << " time=" << time_task_par << " s\n";
        }

        sch_par.deallocate(A, B, C).execute();

        if(task_pg.rank().value() == 0) {
            next = ac.fetch_add(0, 1);
        }
        task_pg.broadcast(&next, 0);
    }
    auto t1_par = std::chrono::high_resolution_clock::now();
    double time_par =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1_par - t0_par).count();
    if(world_pg.rank().value() == 0) {
        std::cout << "Parallel total time = " << time_par << " s\n";
    }
    ac.deallocate();

    tamm::ExecutionContext ec_seq{world_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch_seq{ec_seq};

    auto t0_seq = std::chrono::high_resolution_clock::now();
    for(int idx = 0; idx < ntasks; ++idx) {
        size_t N = tasks[static_cast<size_t>(idx)];
        tamm::Tile bt = static_cast<tamm::Tile>(std::min(N, size_t(64)));
        tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(N)}, bt};
        tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
        auto [l, b, r] = bond.labels<3>("all");
        auto [p1, p2] = phys.labels<2>("all");

        tamm::Tensor<Cplx> A({l, p1, b});
        tamm::Tensor<Cplx> B({b, p2, r});
        tamm::Tensor<Cplx> C({l, p1, p2, r});
        A.set_dense();
        B.set_dense();
        C.set_dense();

        sch_seq.allocate(A, B, C).execute();
        sch_seq(A() = Cplx{1.0, 0.0})(B() = Cplx{1.0, 0.0})(C() = Cplx{0.0, 0.0}).execute();

        auto t0_task_seq = std::chrono::high_resolution_clock::now();
        sch_seq(C(l, p1, p2, r) = A(l, p1, b) * B(b, p2, r)).execute(ec_seq.exhw(), false);
        auto t1_task_seq = std::chrono::high_resolution_clock::now();
        double time_task_seq =
            std::chrono::duration_cast<std::chrono::duration<double>>(t1_task_seq - t0_task_seq).count();
        if(world_pg.rank().value() == 0) {
            std::cout << "Sequential task " << idx << " N=" << N
                      << " time=" << time_task_seq << " s\n";
        }

        sch_seq.deallocate(A, B, C).execute();
    }
    auto t1_seq = std::chrono::high_resolution_clock::now();
    double time_seq =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1_seq - t0_seq).count();
    if(world_pg.rank().value() == 0) {
        std::cout << "Sequential total time = " << time_seq << " s\n";
    }

    tamm::finalize();
    return 0;
}

