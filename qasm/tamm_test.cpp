#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

#include <tamm/tamm.hpp>
#include <itensor/all.h>

int main(int argc, char* argv[]) {
    using Cplx = std::complex<double>;

    // Initialize TAMM
    tamm::initialize(argc, argv);
    tamm::ProcGroup pg = tamm::ProcGroup::create_world_coll();

    // 1) Bond dimensions to test
    std::vector<size_t> sizes;
    const size_t N_min = 1, N_max = 1000;
    const int num_N = 10;
    const double stepN = double(N_max - N_min) / double(num_N - 1);
    for(int idx=0; idx<num_N; ++idx)
        sizes.push_back(size_t(std::round(N_min + idx*stepN)));
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    // 2) TAMM tile sizes
    std::vector<size_t> tile_sizes;
    const size_t T_min=100, T_max=160;
    const int num_T=4;
    const double stepT = double(T_max - T_min) / double(num_T - 1);
    for(int idx=0; idx<num_T; ++idx)
        tile_sizes.push_back(size_t(std::round(T_min + idx*stepT)));
    std::sort(tile_sizes.begin(), tile_sizes.end());
    tile_sizes.erase(std::unique(tile_sizes.begin(), tile_sizes.end()), tile_sizes.end());

    // 3) Precompute ITensor merge times
    std::unordered_map<size_t,double> itensor_times;
    for(size_t N : sizes) {
        double t = std::numeric_limits<double>::quiet_NaN();
        try {
            itensor::Index l(int(N),"l"), b(int(N),"b"), p1(2,"p1"), p2(2,"p2"), r(int(N),"r");
            itensor::ITensor A(l,p1,b), B(b,p2,r);
            for(int li=1; li<=int(N); ++li)
            for(int pi=1; pi<=2;    ++pi)
            for(int bi=1; bi<=int(N); ++bi)
                A.set(l(li),p1(pi),b(bi), Cplx{1.0,0.0});
            for(int bi=1; bi<=int(N); ++bi)
            for(int pi=1; pi<=2;    ++pi)
            for(int ri=1; ri<=int(N); ++ri)
                B.set(b(bi),p2(pi),r(ri), Cplx{1.0,0.0});
            auto t0 = std::chrono::high_resolution_clock::now();
            itensor::ITensor C = A * B;
            auto t1 = std::chrono::high_resolution_clock::now();
            t = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
            if(pg.rank().value()==0) 
                std::cout<<"ITensor merge: N="<<N<<", "<<t<<" s\n";
        }
        catch(const std::exception &e){
            if(pg.rank().value()==0)
                std::cerr<<"ITensor merge N="<<N<<" failed: "<<e.what()<<"\n";
        }
        itensor_times[N] = t;
    }

    // 4) TAMM distributions
    std::vector<std::pair<tamm::DistributionKind,std::string>> dist_settings = {
        {tamm::DistributionKind::nw,    "nw"},
        {tamm::DistributionKind::dense, "dense"}
    };

    std::ofstream csv("merge_times.csv");
    csv<<"distribution,tile,N,cpu_time,gpu_time,itensor_time\n";
    if(pg.rank().value()==0)
        std::cout<<"distribution,tile,N,cpu_time,gpu_time,itensor_time\n";

    // 5) Benchmark loops
    for(auto const& [dist_kind,dist_name] : dist_settings) {
        tamm::ExecutionContext ec{pg, dist_kind, tamm::MemoryManagerKind::ga};
        if(!ec.has_gpu() && pg.rank().value()==0)
            std::cerr<<"Warning("<<dist_name<<"): no GPU → gpu_time=0\n";
        auto cpu_hw = tamm::ExecutionHW::CPU;
        auto gpu_hw = ec.exhw();
        bool dense = dist_kind==tamm::DistributionKind::dense;

        for(size_t tile_val : tile_sizes) {
        for(size_t N : sizes) {
            // a) TiledIndexSpaces
            auto btile = static_cast<tamm::Tile>(tile_val);
            tamm::TiledIndexSpace bond_tis{tamm::IndexSpace{tamm::range(N)}, btile};
            tamm::TiledIndexSpace phys_tis{tamm::IndexSpace{tamm::range(2)},   1};

            // b) Extract three bond labels, two phys labels with prefix "all"
            auto [l,b,r] = bond_tis.labels<3>("all");
            auto [p1,p2] = phys_tis.labels<2>("all");

            // c) Build A,B,C
            tamm::Tensor<Cplx> A({l,p1,b}), B({b,p2,r}), C({l,p1,p2,r});
            if(dense) { A.set_dense(); B.set_dense(); C.set_dense(); }

            // d) Allocate & init
            tamm::Scheduler sch{ec};
            sch.allocate(A,B,C).execute();
            sch(A()=Cplx{1,0})(B()=Cplx{1,0})(C()=Cplx{0,0}).execute();

            // e) CPU merge
            auto t0 = std::chrono::high_resolution_clock::now();
            sch(C(l,p1,p2,r) = A(l,p1,b)*B(b,p2,r)).execute(cpu_hw,false);
            auto t1 = std::chrono::high_resolution_clock::now();
            double cpu_t = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();

            // reset C
            sch(C()=Cplx{0,0}).execute();

            // f) GPU merge
            double gpu_t = 0.0;
            if(ec.has_gpu()){
                auto t2 = std::chrono::high_resolution_clock::now();
                sch(C(l,p1,p2,r)=A(l,p1,b)*B(b,p2,r)).execute(gpu_hw,false);
                auto t3 = std::chrono::high_resolution_clock::now();
                gpu_t = std::chrono::duration_cast<std::chrono::duration<double>>(t3 - t2).count();
            }

            // g) ITensor time lookup
            double it_t = itensor_times[N];

            // h) Record
            csv<<dist_name<<","<<tile_val<<","<<N<<","<<cpu_t<<","<<gpu_t<<","
               <<(std::isnan(it_t)?"NA":std::to_string(it_t))<<"\n";
            if(pg.rank().value()==0){
                std::cout<<dist_name<<","<<tile_val<<","<<N<<","<<cpu_t<<","<<gpu_t<<","
                         <<(std::isnan(it_t)?"NA\n":std::to_string(it_t)+"\n");
            }

            sch.deallocate(A,B,C).execute();
        }}
    }

    csv.close();
    tamm::finalize();
    return 0;
}

