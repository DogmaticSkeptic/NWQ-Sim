#include <chrono>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <string>
#include <tamm/tamm.hpp>
#include <itensor/all.h>

int main(int argc, char* argv[]) {
    using Cplx  = std::complex<double>;
    using Clock = std::chrono::high_resolution_clock;
    if(argc < 6) {
        std::cerr << "Usage: " << argv[0] << " subrank ntasks nsteps tile_size nrepeats\n";
        return 1;
    }
    int subrank     = std::stoi(argv[1]);
    int ntasks      = std::stoi(argv[2]);
    size_t nsteps   = std::stoul(argv[3]);
    size_t step_size= std::stoul(argv[4]);
    size_t tile_size= std::stoul(argv[5]);
    int nrepeats    = std::stoi(argv[6]);
    std::string filename = "results.txt";

    tamm::initialize(argc, argv);
    tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();
    bool is_root = (world_pg.rank().value() == 0);

    std::ofstream ofs(filename);
    ofs << "#M seq_mean seq_std par_mean par_std itensor_mean itensor_std\n";

    for(size_t step = 1; step <= nsteps; ++step) {
        size_t M = step_size * step;
        if(is_root) {
            std::cout << "Starting M = " << M << std::endl;
        }
        std::vector<double> seq_times(nrepeats), par_times(nrepeats), itensor_times(nrepeats);

        for(int rep = 0; rep < nrepeats; ++rep) {
            if(is_root) {
                std::cout << "  Repeat " << (rep + 1) << " of " << nrepeats << std::endl;
            }
            std::vector<size_t> tasks(ntasks, M);

            tamm::ExecutionContext ec_seq{world_pg, tamm::DistributionKind::nw, tamm::MemoryManagerKind::ga};
            tamm::Scheduler sch_seq{ec_seq};
            double seq_gpu_time = 0.0;

            for(int i = 0; i < ntasks; ++i) {
                tamm::Tile bt = static_cast<tamm::Tile>(tile_size);
                tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(M)}, bt};
                tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
                auto [l,b,r] = bond.labels<3>("all");
                auto [p1,p2] = phys.labels<2>("all");
                tamm::Tensor<Cplx> A({l,p1,b}), B({b,p2,r}), C({l,p1,p2,r});
                sch_seq.allocate(A,B,C).execute();
                sch_seq(A()=Cplx{1.0,0.0})(B()=Cplx{1.0,0.0})(C()=Cplx{0.0,0.0}).execute();
                auto t0 = Clock::now();
                sch_seq(C(l,p1,p2,r)=A(l,p1,b)*B(b,p2,r)).execute(tamm::ExecutionHW::GPU, false);
                auto t1 = Clock::now();
                seq_gpu_time += std::chrono::duration<double>(t1 - t0).count();
                sch_seq.deallocate(A,B,C).execute();
            }
            seq_times[rep] = seq_gpu_time;

            tamm::ProcGroup task_pg = tamm::ProcGroup::create_subgroups(world_pg, subrank);
            tamm::ExecutionContext ec_par{task_pg, tamm::DistributionKind::nw, tamm::MemoryManagerKind::ga};
            tamm::Scheduler sch_par{ec_par};
            tamm::AtomicCounterGA ac{world_pg,1};
            ac.allocate(0);
            int64_t next = -1;
            if(task_pg.rank().value() == 0) next = ac.fetch_add(0,1);
            task_pg.broadcast(&next,0);

            double par_gpu_time = 0.0;
            while(next < ntasks) {
                size_t Mi = tasks[next];
                tamm::Tile bt = static_cast<tamm::Tile>(tile_size);
                tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(Mi)}, bt};
                tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
                auto [l,b,r] = bond.labels<3>("all");
                auto [p1,p2] = phys.labels<2>("all");
                tamm::Tensor<Cplx> A({l,p1,b}), B({b,p2,r}), C({l,p1,p2,r});
                sch_par.allocate(A,B,C).execute();
                sch_par(A()=Cplx{1.0,0.0})(B()=Cplx{1.0,0.0})(C()=Cplx{0.0,0.0}).execute();
                auto t0 = Clock::now();
                sch_par(C(l,p1,p2,r)=A(l,p1,b)*B(b,p2,r)).execute(ec_par.exhw(), false);
                auto t1 = Clock::now();
                par_gpu_time += std::chrono::duration<double>(t1 - t0).count();
                sch_par.deallocate(A,B,C).execute();
                if(task_pg.rank().value() == 0) next = ac.fetch_add(0,1);
                task_pg.broadcast(&next,0);
            }
            ac.deallocate();
            par_times[rep] = par_gpu_time;

            double itensor_time = 0.0;
            for(int i = 0; i < ntasks; ++i) {
                itensor::Index l(int(tasks[i]), "l");
                itensor::Index p1(2, "p1");
                itensor::Index p2(2, "p2");
                itensor::Index b(int(tasks[i]), "b");
                itensor::Index r(int(tasks[i]), "r");
                itensor::ITensor A_it(l,p1,b), B_it(b,p2,r), C_it(l,p1,p2,r);
                A_it.fill(1.0);
                B_it.fill(1.0);
                auto t0 = Clock::now();
                C_it = A_it * B_it;
                auto t1 = Clock::now();
                itensor_time += std::chrono::duration<double>(t1 - t0).count();
            }
            itensor_times[rep] = itensor_time;

            if(is_root) {
                std::cout << "    seq=" << seq_gpu_time
                          << " par=" << par_gpu_time
                          << " itensor=" << itensor_time
                          << std::endl;
            }
        }

        auto compute_stats = [&](const std::vector<double>& v){
            double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
            double var  = 0.0;
            for(double x : v) var += (x - mean) * (x - mean);
            var /= v.size();
            return std::make_pair(mean, std::sqrt(var));
        };

        auto [seq_mean, seq_std]         = compute_stats(seq_times);
        auto [par_mean, par_std]         = compute_stats(par_times);
        auto [itensor_mean, itensor_std] = compute_stats(itensor_times);

        if(is_root) {
            std::cout << "Completed M = " << (step_size * step)
                      << " seq_mean=" << seq_mean << " seq_std=" << seq_std
                      << " par_mean=" << par_mean << " par_std=" << par_std
                      << " itensor_mean=" << itensor_mean << " itensor_std=" << itensor_std
                      << std::endl;
            ofs << (tile_size * step) << " "
                << std::fixed << std::setprecision(6)
                << seq_mean    << " " << seq_std    << " "
                << par_mean    << " " << par_std    << " "
                << itensor_mean<< " " << itensor_std<< "\n";
        }
    }

    tamm::finalize();
    return 0;
}

