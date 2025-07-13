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
    using Clock = std::chrono::high_resolution_clock;
    tamm::initialize(argc, argv);
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();

    int min_sub   = std::stoi(argv[1]);
    int max_sub   = std::stoi(argv[2]);
    int step_sub  = std::stoi(argv[3]);
    int ntasks    = std::stoi(argv[4]);
    size_t N      = static_cast<size_t>(std::stoi(argv[5]));
    std::string filename = argv[6];

    std::vector<size_t> tasks(ntasks, N);

    tamm::ExecutionContext ec_seq{
        world_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga
    };
    tamm::Scheduler sch_seq{ec_seq};

    double seq_exec_time = 0.0;
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
        auto t0 = Clock::now();
        sch_seq(C(l,p1,p2,r)=A(l,p1,b)*B(b,p2,r)).execute(tamm::ExecutionHW::CPU,false);
        auto t1 = Clock::now();
        seq_exec_time += std::chrono::duration<double>(t1 - t0).count();
        sch_seq.deallocate(A,B,C).execute();
    }

    double itensor_time = 0.0;
    for(int i = 0; i < ntasks; ++i) {
        size_t M  = tasks[static_cast<size_t>(i)];
        size_t bt = std::min(M, size_t(64));
        itensor::Index l(M,"l");
        itensor::Index p1(2,"p1");
        itensor::Index p2(2,"p2");
        itensor::Index b(bt,"b");
        itensor::Index r(M,"r");
        itensor::ITensor A_it(l,p1,b), B_it(b,p2,r), C_it(l,p1,p2,r);
        A_it.fill(1.0);
        B_it.fill(1.0);
        auto t0 = Clock::now();
        C_it = A_it * B_it;
        auto t1 = Clock::now();
        itensor_time += std::chrono::duration<double>(t1 - t0).count();
    }

    std::ofstream ofs(filename);
    ofs << "#subranks parallel_time[s] tamm_sequential_time[s] itensor_sequential_time[s]\n";

    for(int subranks = min_sub; subranks <= max_sub; subranks += step_sub) {
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

        double par_exec_time = 0.0;
        int count = 0;
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
            auto t0 = Clock::now();
            sch_par(C(l,p1,p2,r)=A(l,p1,b)*B(b,p2,r)).execute(ec_par.exhw(),false);
            auto t1 = Clock::now();
            par_exec_time += std::chrono::duration<double>(t1 - t0).count();
            sch_par.deallocate(A,B,C).execute();
            ++count;
            if(task_pg.rank().value() == 0) next = ac.fetch_add(0,1);
            task_pg.broadcast(&next,0);
        }
        ac.deallocate();

        if(world_pg.rank().value() == 0) {
            ofs << subranks << " "
                << std::fixed << std::setprecision(6) << par_exec_time << " "
                << std::fixed << std::setprecision(6) << seq_exec_time << " "
                << std::fixed << std::setprecision(6) << itensor_time << "\n";
        }
    }

    tamm::finalize();
    return 0;
}

