#include <chrono>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>
#include <tamm/tamm.hpp>

int main(int argc, char* argv[]) {
    using Cplx = std::complex<double>;
    tamm::initialize(argc, argv);
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();
    int nranks = world_pg.size().value();

    // Prepare 20 merge tasks
    std::vector<size_t> tasks;
    for(int i = 1; i <= 20; ++i) {
        tasks.push_back(static_cast<size_t>(i) * 128);
    }
    int ntasks = static_cast<int>(tasks.size());

    // -------------------------------
    // Parallel section with subgroups
    // -------------------------------
    int subranks = std::max(1, nranks / ntasks);
    tamm::ProcGroup task_pg =
        tamm::ProcGroup::create_subgroups(world_pg, subranks);
    tamm::ExecutionContext ec_par{
        task_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch_par{ec_par};

    tamm::AtomicCounterGA ac{world_pg, 1};
    ac.allocate(0);

    int64_t next = -1;
    if(task_pg.rank().value() == 0) {
        next = ac.fetch_add(0, 1);
    }
    task_pg.broadcast(&next, 0);

    while(next < ntasks) {
        int64_t task_id = next;
        size_t N = tasks[static_cast<size_t>(task_id)];
        int world_rank = world_pg.rank().value();
        int sub_rank   = task_pg.rank().value();

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
        sch_par(A() = Cplx{1.0, 0.0})
               (B() = Cplx{1.0, 0.0})
               (C() = Cplx{0.0, 0.0})
               .execute();

        // Timestamp: start
        auto t_start = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(
            t_start.time_since_epoch()).count();
        std::cout << "PAR TASK " << task_id
                  << " START world=" << world_rank
                  << " sub="        << sub_rank
                  << " t="          << std::fixed
                  << std::setprecision(6) << secs
                  << "\n";

        sch_par(C(l, p1, p2, r) = A(l, p1, b) * B(b, p2, r))
               .execute(ec_par.exhw(), false);

        // Timestamp: end
        auto t_end = std::chrono::high_resolution_clock::now();
        double secs2 = std::chrono::duration<double>(
            t_end.time_since_epoch()).count();
        std::cout << "PAR TASK " << task_id
                  << " END   world=" << world_rank
                  << " sub="        << sub_rank
                  << " t="          << std::fixed
                  << std::setprecision(6) << secs2
                  << "\n";

        sch_par.deallocate(A, B, C).execute();

        if(sub_rank == 0) {
            next = ac.fetch_add(0, 1);
        }
        task_pg.broadcast(&next, 0);
    }

    ac.deallocate();

    // -------------------------
    // Sequential single-context
    // -------------------------
    tamm::ExecutionContext ec_seq{
        world_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch_seq{ec_seq};

    for(int idx = 0; idx < ntasks; ++idx) {
        size_t N = tasks[static_cast<size_t>(idx)];
        int world_rank = world_pg.rank().value();

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
        sch_seq(A() = Cplx{1.0, 0.0})
               (B() = Cplx{1.0, 0.0})
               (C() = Cplx{0.0, 0.0})
               .execute();

        // Timestamp: start
        auto t_start = std::chrono::high_resolution_clock::now();
        if(world_rank == 0) {
            double secs = std::chrono::duration<double>(
                t_start.time_since_epoch()).count();
            std::cout << "SEQ TASK " << idx
                      << " START world=" << world_rank
                      << " t="          << std::fixed
                      << std::setprecision(6) << secs
                      << "\n";
        }

        sch_seq(C(l, p1, p2, r) = A(l, p1, b) * B(b, p2, r))
               .execute(ec_seq.exhw(), false);

        // Timestamp: end
        auto t_end = std::chrono::high_resolution_clock::now();
        if(world_rank == 0) {
            double secs = std::chrono::duration<double>(
                t_end.time_since_epoch()).count();
            std::cout << "SEQ TASK " << idx
                      << " END   world=" << world_rank
                      << " t="          << std::fixed
                      << std::setprecision(6) << secs
                      << "\n";
        }

        sch_seq.deallocate(A, B, C).execute();
    }

    tamm::finalize();
    return 0;
}

