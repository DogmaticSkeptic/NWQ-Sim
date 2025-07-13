#include <blacspp/grid.hpp>
#include <scalapackpp/block_cyclic_matrix.hpp>
#include <scalapackpp/pblas/gemm.hpp>
#include <scalapackpp/svd.hpp>   // or the header that provides pgesvd wrapper
#include <tamm/tamm.hpp>

using Cplx = std::complex<double>;

int main(int argc, char* argv[]) {
  tamm::initialize(argc, argv);
  tamm::ProcGroup world_pg = tamm::ProcGroup::create_world_coll();

  int min_sub   = std::stoi(argv[1]);
  int max_sub   = std::stoi(argv[2]);
  int step_sub  = std::stoi(argv[3]);
  int ntasks    = std::stoi(argv[4]);
  size_t N      = static_cast<size_t>(std::stoi(argv[5]));
  std::string filename = argv[6];

  std::ofstream ofs(filename);
  ofs << "#subranks parallel_time[s] sequential_time[s]\n";

  for(int subranks = min_sub; subranks <= max_sub; subranks += step_sub) {
    std::vector<size_t> tasks(ntasks, N);

    // Create a subgroup of 'subranks' ranks on which to run each contraction+SVD
    tamm::ProcGroup task_pg =
      tamm::ProcGroup::create_subgroups(world_pg, subranks);
    tamm::ExecutionContext ec_par{
      task_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga
    };
    tamm::Scheduler sch_par{ec_par};

    // Atomic counter to distribute tasks
    tamm::AtomicCounterGA ac{world_pg,1};
    ac.allocate(0);
    int64_t next = -1;
    if(task_pg.rank().value() == 0) next = ac.fetch_add(0,1);
    task_pg.broadcast(&next,0);

    auto t0 = std::chrono::high_resolution_clock::now();
    while(next < ntasks) {
      size_t M = tasks[static_cast<size_t>(next)];
      tamm::Tile bt = static_cast<tamm::Tile>(std::min(M,size_t(64)));

      // 1) Build index spaces and labels
      tamm::TiledIndexSpace bond{tamm::IndexSpace{tamm::range(M)}, bt};
      tamm::TiledIndexSpace phys{tamm::IndexSpace{tamm::range(2)}, 1};
      auto [l,b,r] = bond.labels<3>("all");
      auto [p1,p2] = phys.labels<2>("all");

      // 2) Allocate tensors A, B, C and do the contraction C = A * B
      tamm::Tensor<Cplx> A({l,p1,b}), B({b,p2,r}), C({l,p1,p2,r});
      A.set_dense(); B.set_dense(); C.set_dense();
      sch_par.allocate(A,B,C).execute();
      sch_par(A()=Cplx{1.0,0.0})(B()=Cplx{1.0,0.0})(C()=Cplx{0.0,0.0}).execute();
      sch_par(C(l,p1,p2,r)=A(l,p1,b)*B(b,p2,r)).execute(ec_par.exhw(),false);

      // 3) Convert C into a block-cyclic ScaLAPACK matrix
      //    We view C as a 2-D matrix of size (M*2-by-M*2), or reshape as needed
      int64_t rows = M * 2;       // for example
      int64_t cols = M * 2;
      int nb = static_cast<int>(bt);

      // determine a “square” process grid on task_pg
      int world_size = task_pg.size().value();
      int npr = std::floor(std::sqrt(world_size));
      int npc = world_size / npr;
      while(npr * npc != world_size) {
        --npr;
        npc = world_size / npr;
      }

      // BLACS grid and BlockCyclicDist2D
      std::vector<int> ranks(world_size);
      std::iota(ranks.begin(),ranks.end(),0);
      blacspp::Grid grid{ task_pg.comm(), npr, npc, ranks.data(), npr };
      scalapackpp::BlockCyclicDist2D bcd{ grid, nb, nb, 0, 0 };

      // Create a ScaLAPACK wrapper matrix; data() points to local storage
      scalapackpp::BlockCyclicMatrix<Cplx> M_sca{ grid, rows, cols, nb, nb };
      auto desc_M = bcd.descinit_noerror(rows, cols, nb, nb);

      // 4) Copy the contracted result C into M_sca’s local buffer
      //    (You must extract each tile of C into the correct local submatrix in M_sca.data())
      auto Cbuf = C.access_local_buf();            // pointer to GA‐allocated tile data
      // You would loop over tiles of C, copy to M_sca.data() respecting the BLACS blocking.
      // For brevity, that copy loop is omitted here.

      // 5) Allocate space for singular values and (optionally) U, VT
      int64_t k = std::min(rows,cols);
      std::vector<double> S(k);
      scalapackpp::BlockCyclicMatrix<Cplx> U_sca{ grid, rows, rows, nb, nb };
      scalapackpp::BlockCyclicMatrix<Cplx> VT_sca{ grid, cols, cols, nb, nb };

      // 6) Call the parallel SVD
      //    Replace `pgesvd` below with whatever wrapper scalapackpp provides
      auto info = scalapackpp::pgesvd(
        'A', 'A',                 // compute all U and VT
        rows, cols,
        M_sca.data(), 1,1, desc_M,
        S.data(),
        U_sca.data(), 1,1, U_sca.desc(),
        VT_sca.data(),1,1, VT_sca.desc()
      );
      if(info != 0 && task_pg.rank().value()==0) {
        std::cerr<<"SVD failed: info="<<info<<std::endl;
      }

      // 7) (Optionally) gather U, S, VT or post-process in block-cyclic form

      sch_par.deallocate(A,B,C).execute();
      if(task_pg.rank().value()==0) next = ac.fetch_add(0,1);
      task_pg.broadcast(&next,0);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double tpar = std::chrono::duration<double>(t1 - t0).count();
    ac.deallocate();

    // Sequential timing omitted for brevity…
    if(world_pg.rank().value()==0) ofs<<subranks<<" "<<tpar<<"\n";
  }

  tamm::finalize();
  return 0;
}

