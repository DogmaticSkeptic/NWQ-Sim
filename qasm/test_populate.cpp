#include <tamm/tamm.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include <array>

using namespace tamm;

using IdxType = size_t;
using Cplx = std::complex<double>;

void print_4d_tensor(Tensor<Cplx>& t, const std::string& name) {
    // This function should only be called from rank 0 on a local tensor
    if (t.execution_context()->pg().rank() != 0) return;

    std::cout << "\n--- Contents of Tensor: " << name << " ---" << std::endl;
    
    // Access the buffer directly, just like during population
    const Cplx* buf_ptr = t.access_local_buf();

    auto dims = t.tiled_index_spaces();
    IdxType d1 = dims[0].index_space().num_indices();
    IdxType d2 = dims[1].index_space().num_indices();
    IdxType d3 = dims[2].index_space().num_indices();
    IdxType d4 = dims[3].index_space().num_indices();

    size_t c = 0;
    // The tensor data is in row-major (Fortran-style) layout.
    // The last index is the fastest moving.
    for (int i = 0; i < d1; ++i) {
        for (int j = 0; j < d2; ++j) {
            std::cout << "[ ";
            for (int k = 0; k < d3; ++k) {
                for (int l = 0; l < d4; ++l, ++c) {
                    std::cout << std::fixed << std::setprecision(1) << std::setw(8) << buf_ptr[c] << " ";
                }
            }
            std::cout << "]" << std::endl;
        }
    }
    std::cout << "------------------------------------------" << std::endl;
}

int main(int argc, char* argv[]) {
    tamm::initialize(argc, argv);

    // Use a self ProcGroup and local memory for a non-distributed tensor
    ProcGroup pg_local = ProcGroup::create_self();
    ExecutionContext ec_local{pg_local, DistributionKind::dense, MemoryManagerKind::local};
    Scheduler sch_local{ec_local};

    if (ec_local.pg().rank() == 0) {
        std::cout << ">>> Populating a 2x2x2x2 local dense tensor..." << std::endl;
    }
    
    // Each dimension is size 2, with a single tile of size 2
    TiledIndexSpace phys_tis{IndexSpace{range(2)}, 2};

    Tensor<Cplx> G4_local({phys_tis, phys_tis, phys_tis, phys_tis});
    G4_local.set_dense();
    sch_local.allocate(G4_local).execute();

    std::array<Cplx, 16> U4;
    for (int i = 0; i < 16; ++i) {
        U4[i] = Cplx(i + 1.0, 0.0);
    }

    Cplx* g4_buffer_ptr = G4_local.access_local_buf();

    // TAMM stores dense tensors in a column-major (Fortran-style) layout.
    // The first index is the fastest-moving.
    size_t c = 0;
    for (int p1_in = 0; p1_in < 2; ++p1_in) {
        for (int p0_in = 0; p0_in < 2; ++p0_in) {
            for (int p1p = 0; p1p < 2; ++p1p) {
                for (int p0p = 0; p0p < 2; ++p0p, ++c) {
                    int row = p0p * 2 + p1p;
                    int col = p0_in * 2 + p1_in;
                    g4_buffer_ptr[c] = U4[row * 4 + col];
                }
            }
        }
    }

    print_4d_tensor(G4_local, "G4_local");
    
    sch_local.deallocate(G4_local).execute();
    pg_local.destroy_coll();

    tamm::finalize();
    return 0;
}
