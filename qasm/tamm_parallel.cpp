#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <tamm/tamm.hpp>
#include <itensor/all.h>

int main(int argc, char* argv[]) {
    using Cplx = std::complex<double>;
    tamm::initialize(argc, argv);
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();

    int min_sub   = std::stoi(argv[1]);
    int max_sub   = std::stoi(argv[2]);
    int step_sub  = std::stoi(argv[3]);
    int ntasks    = std::stoi(argv[4]);
    size_t N      = static_cast<size_t>(std::stoi(argv[5]));
    std::string filename = argv[6];

    std::ofstream ofs(filename);
    ofs << "#subranks parallel_time[s] tamm_sequential_time[s] itensor_sequential_time[s]\n";

    for(int subranks = min_sub; subranks <= max_sub; subranks += step_sub) {
        std::vector<size_t> tasks(ntasks, N);

        tamm::ProcGroup task_pg =
            tamm::ProcGroup::create_subgroups(world_pg, subranks);
        tamm::ExecutionContext ec_par{
            task_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga
        };
        tamm::Scheduler sch_par{ec_par};

        tamm::AtomicCounterGA ac{world_pg,1};
        ac.allocate(0);
        int64_t next = -1;
        if(task_pg.rank().value() == 0) next = ac.fetch_add(0,1);
        task_pg.broadcast(&next,0);

        auto t0 = std::chrono::high_resolution_clock::now();
        while(next < ntasks) {
            size_t M = tasks[static_cast<size_t>(next)];
            tamm::Tile bt = static_cast<tamm::Tile>(std::min(M, size_t(64)));
            tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(M)}, bt};
            tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
            auto [l,b,r] = bond.labels<3>("all");
            auto [p1,p2] = phys.labels<2>("all");

            tamm::Tensor<Cplx> A({l,p1,b}), B({b,p2,r}), C({l,p1,p2,r});
            A.set_dense(); B.set_dense(); C.set_dense();
            sch_par.allocate(A,B,C).execute();
            sch_par(A()=Cplx{1.0,0.0})(B()=Cplx{1.0,0.0})(C()=Cplx{0.0,0.0}).execute();
            sch_par(C(l,p1,p2,r)=A(l,p1,b)*B(b,p2,r)).execute(ec_par.exhw(),false);
            sch_par.deallocate(A,B,C).execute();

            if(task_pg.rank().value() == 0) next = ac.fetch_add(0,1);
            task_pg.broadcast(&next,0);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double tpar = std::chrono::duration<double>(t1 - t0).count();
        ac.deallocate();

        tamm::ExecutionContext ec_seq{
            world_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga
        };
        tamm::Scheduler sch_seq{ec_seq};

        auto t2 = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < ntasks; ++i) {
            size_t M = tasks[static_cast<size_t>(i)];
            tamm::Tile bt = static_cast<tamm::Tile>(std::min(M, size_t(64)));
            tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(M)}, bt};
            tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
            auto [l,b,r] = bond.labels<3>("all");
            auto [p1,p2] = phys.labels<2>("all");

            tamm::Tensor<Cplx> A({l,p1,b}), B({b,p2,r}), C({l,p1,p2,r});
            A.set_dense(); B.set_dense(); C.set_dense();
            sch_seq.allocate(A,B,C).execute();
            sch_seq(A()=Cplx{1.0,0.0})(B()=Cplx{1.0,0.0})(C()=Cplx{0.0,0.0}).execute();
            sch_seq(C(l,p1,p2,r)=A(l,p1,b)*B(b,p2,r)).execute(ec_seq.exhw(),false);
            sch_seq.deallocate(A,B,C).execute();
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double tseq = std::chrono::duration<double>(t3 - t2).count();

        auto t_it0 = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < ntasks; ++i) {
            size_t M  = tasks[static_cast<size_t>(i)];
            size_t bt = std::min(M, size_t(64));
            itensor::Index l("l", M);
            itensor::Index p1("p1", 2);
            itensor::Index p2("p2", 2);
            itensor::Index b("b", bt);
            itensor::Index r("r", M);

            itensor::ITensor A_it(l,p1,b);
            itensor::ITensor B_it(b,p2,r);
            itensor::ITensor C_it(l,p1,p2,r);

            A_it.fill(1.0);
            B_it.fill(1.0);
            C_it = A_it * B_it;
        }
        auto t_it1 = std::chrono::high_resolution_clock::now();
        double titensor = std::chrono::duration<double>(t_it1 - t_it0).count();

        if(world_pg.rank().value() == 0) {
            ofs << subranks << " "
                << std::fixed << std::setprecision(6) << tpar      << " "
                << std::fixed << std::setprecision(6) << tseq     << " "
                << std::fixed << std::setprecision(6) << titensor << "\n";
        }
    }

    tamm::finalize();
    return 0;
}

