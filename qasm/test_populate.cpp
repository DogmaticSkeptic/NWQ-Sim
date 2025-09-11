#include <tamm/tamm.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include <array>

// Use the tamm namespace
using namespace tamm;

// Define the missing type alias from your project.
using IdxType = size_t;
// Define a type alias for complex double for convenience
using Cplx = std::complex<double>;

/**
 * @brief Helper function to print a 4D tensor's contents.
 */
void print_4d_tensor(Tensor<Cplx>& t, const std::string& name) {
    // This function is only ever called from rank 0 in this test.
    if (t.execution_context()->pg().rank() != 0) return;

    std::cout << "\n--- Contents of Tensor: " << name << " ---" << std::endl;
    
    std::vector<Cplx> buf(t.size());
    t.get(*(t.loop_nest().begin()), buf);

    auto dims = t.tiled_index_spaces();
    IdxType d1 = dims[0].index_space().num_indices();
    IdxType d2 = dims[1].index_space().num_indices();
    IdxType d3 = dims[2].index_space().num_indices();
    IdxType d4 = dims[3].index_space().num_indices();

    size_t c = 0;
    for (int i = 0; i < d1 * d2; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < d3 * d4; ++j, ++c) {
            std::cout << std::fixed << std::setprecision(1) << std::setw(8) << buf[c] << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;
}


int main(int argc, char* argv[]) {
    tamm::initialize(argc, argv);

    // This test is designed to run on a single process.
    // We create a process group containing only this process.
    ProcGroup pg_local = ProcGroup::create_self();
    ExecutionContext ec_local{pg_local, DistributionKind::dense, MemoryManagerKind::local};
    Scheduler sch_local{ec_local};

    if (ec_local.pg().rank() == 0) {
        std::cout << ">>> Running final test with direct buffer access method..." << std::endl;
    }
    
    // Define the TiledIndexSpace for a qubit (dimension 2, tile size 1).
    TiledIndexSpace phys_tis{IndexSpace{range(2)}, 1};

    // Define the 4-index gate tensor. Dimensions will be (2, 2, 2, 2).
    Tensor<Cplx> G4_local({phys_tis, phys_tis, phys_tis, phys_tis});
    G4_local.set_dense();
    sch_local.allocate(G4_local).execute();

    // Define the source 4x4 gate matrix data (values 1-16 for easy checking).
    std::array<Cplx, 16> U4;
    for (int i = 0; i < 16; ++i) {
        U4[i] = Cplx(i + 1.0, 0.0);
    }

    // THE CORRECT AND ROBUST SOLUTION:
    // 1. Get a direct pointer to the tensor's entire local memory buffer.
    Cplx* g4_buffer_ptr = G4_local.access_local_buf();

    // 2. Fill this buffer directly, respecting the row-major memory layout.
    size_t c = 0;
    for (int p0p = 0; p0p < 2; ++p0p) {
        for (int p1p = 0; p1p < 2; ++p1p) {
            for (int p0_in = 0; p0_in < 2; ++p0_in) {
                for (int p1_in = 0; p1_in < 2; ++p1_in, ++c) {
                    int row = p0p * 2 + p1p;
                    int col = p0_in * 2 + p1_in;
                    g4_buffer_ptr[c] = U4[row * 4 + col];
                }
            }
        }
    }

    // 3. The data is now in the tensor. No 'put' or 'sch.execute()' is needed for this step.

    // VERIFICATION: Print the tensor to confirm it was populated correctly.
    print_4d_tensor(G4_local, "G4_local (Result of Correct Method)");
    
    // Final cleanup.
    sch_local.deallocate(G4_local).execute();
    pg_local.destroy_coll();

    tamm::finalize();
    return 0;
}
