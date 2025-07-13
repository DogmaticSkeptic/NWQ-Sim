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

    double alloc_time = 0.0;
    double init_time = 0.0;
    double exec_time = 0.0;
    double dealloc_time = 0.0;

    auto t_seq_start = Clock::now();
    for(int i = 0; i < ntasks; ++i) {
        size_t M = tasks[static_cast<size_t>(i)];
        tamm::Tile bt = static_cast<tamm::Tile>(164);
        tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(M)}, bt};
        tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
        auto [l,b,r] = bond.labels<3>("all");
        auto [p1,p2] = phys.labels<2>("all");

        tamm::Tensor<Cplx> A({l,p1,b}), B({b,p2,r}), C({l,p1,p2,r});
        A.set_dense(); B.set_dense(); C.set_dense();

        auto t0 = Clock::now();
        sch_seq.allocate(A,B,C).execute();
        auto t1 = Clock::now();
        alloc_time += std::chrono::duration<double>(t1 - t0).count();

        auto t2 = Clock::now();
        sch_seq(A()=Cplx{1.0,0.0})(B()=Cplx{1.0,0.0})(C()=Cplx{0.0,0.0}).execute();
        auto t3 = Clock::now();
        init_time += std::chrono::duration<double>(t3 - t2).count();

        auto t4 = Clock::now();
        sch_seq(C(l,p1,p2,r)=A(l,p1,b)*B(b,p2,r)).execute(tamm::ExecutionHW::CPU,false);
        auto t5 = Clock::now();
        exec_time += std::chrono::duration<double>(t5 - t4).count();

        auto t6 = Clock::now();
        sch_seq.deallocate(A,B,C).execute();
        auto t7 = Clock::now();
        dealloc_time += std::chrono::duration<double>(t7 - t6).count();
    }
    auto t_seq_end = Clock::now();
    double seq_total = std::chrono::duration<double>(t_seq_end - t_seq_start).count();
    double avg_time = seq_total / ntasks;

    auto t_it0 = Clock::now();
    for(int i = 0; i < ntasks; ++i) {
        size_t M  = tasks[static_cast<size_t>(i)];
        size_t bt = std::min(M, size_t(64));

        itensor::Index l(M,"l");
        itensor::Index p1(2,"p1");
        itensor::Index p2(2,"p2");
        itensor::Index b(bt,"b");
        itensor::Index r(M,"r");

        itensor::ITensor A_it(l,p1,b);
        itensor::ITensor B_it(b,p2,r);
        itensor::ITensor C_it(l,p1,p2,r);

        A_it.fill(1.0);
        B_it.fill(1.0);
        C_it = A_it * B_it;
    }
    auto t_it1 = Clock::now();
    double titensor = std::chrono::duration<double>(t_it1 - t_it0).count();

    std::ofstream ofs(filename);
    ofs << "#subranks parallel_time[s] tamm_sequential_time[s] itensor_sequential_time[s]\n";
    ofs << "#tamm_seq_alloc[s] tamm_seq_init[s] tamm_seq_exec[s] tamm_seq_dealloc[s] "
        << "tamm_seq_total[s] tamm_seq_avg_per_task[s]\n";
    if(world_pg.rank().value() == 0) {
        std::cout << "tamm_seq_alloc " << alloc_time << "\n";
        std::cout << "tamm_seq_init  " << init_time  << "\n";
        std::cout << "tamm_seq_exec  " << exec_time  << "\n";
        std::cout << "tamm_seq_dealloc " << dealloc_time << "\n";
        std::cout << "tamm_seq_total " << seq_total << "\n";
        std::cout << "tamm_seq_avg_per_task " << avg_time << "\n";
    }

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

        auto t0 = Clock::now();
        while(next < ntasks) {
            size_t M = tasks[static_cast<size_t>(next)];
            tamm::Tile bt = static_cast<tamm::Tile>(164);
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
        auto t1 = Clock::now();
        double tpar = std::chrono::duration<double>(t1 - t0).count();
        ac.deallocate();

        if(world_pg.rank().value() == 0) {
            ofs << subranks << " "
                << std::fixed << std::setprecision(6) << tpar      << " "
                << std::fixed << std::setprecision(6) << seq_total << " "
                << std::fixed << std::setprecision(6) << titensor << "\n";
        }
    }

    tamm::finalize();
    return 0;
}

