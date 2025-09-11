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
 * 
 * This function reshapes the 4D tensor's data into a 2D matrix format (row-major)
 * to make it easy to visually compare with the source 4x4 U4 matrix.
 * It is designed to run only on rank 0 to avoid cluttered output in MPI.
 *
 * @param t The tensor to print.
 * @param name A descriptive name for the tensor to be printed in the header.
 */
void print_4d_tensor(Tensor<Cplx>& t, const std::string& name) {
    if (t.execution_context()->pg().rank() != 0) return;

    std::cout << "\n--- Contents of Tensor: " << name << " ---" << std::endl;
    
    // Get all tensor data into a local buffer on rank 0.
    std::vector<Cplx> buf(t.size());
    // Since it's a dense local tensor, its loop_nest has one block.
    // We get the data from that single block.
    t.get(*(t.loop_nest().begin()), buf);

    auto dims = t.tiled_index_spaces();
    IdxType d1 = dims[0].index_space().num_indices();
    IdxType d2 = dims[1].index_space().num_indices();
    IdxType d3 = dims[2].index_space().num_indices();
    IdxType d4 = dims[3].index_space().num_indices();

    // Reshape and print as a (d1*d2) x (d3*d4) matrix, which is 4x4 for this case.
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

    // Set up a local execution context, which runs on a single process.
    ProcGroup pg_local = ProcGroup::create_self();
    ExecutionContext ec_local{pg_local, DistributionKind::dense, MemoryManagerKind::ga};
    Scheduler sch_local{ec_local};

    // Define the TiledIndexSpace for a qubit (dimension 2, tile size 1).
    TiledIndexSpace phys_tis{IndexSpace{range(2)}, 1};

    // Define the source 4x4 gate matrix data (values 1-16 for easy checking).
    std::array<Cplx, 16> U4;
    for (int i = 0; i < 16; ++i) {
        U4[i] = Cplx(i + 1.0, 0.0);
    }

    // --- 1. DEMONSTRATE THE BUGGY BEHAVIOR ---
    if(ec_local.pg().rank() == 0) {
      std::cout << "\n>>> 1. Populating G4_local_buggy with the buggy block-by-block 'put' method..." << std::endl;
    }
    
    { // Use a new scope to manage the lifetime of the buggy tensor
        Tensor<Cplx> G4_local_buggy({phys_tis, phys_tis, phys_tis, phys_tis});
        G4_local_buggy.set_dense();
        sch_local.allocate(G4_local_buggy).execute();

        // The problematic code: loop and call put() for each of the 16 blocks.
        for (const auto& blockid : G4_local_buggy.loop_nest()) {
            IdxType p0_prime = blockid[0];
            IdxType p1_prime = blockid[1];
            IdxType p0_in    = blockid[2];
            IdxType p1_in    = blockid[3];

            int row = p0_prime * 2 + p1_prime;
            int col = p0_in * 2 + p1_in;
            Cplx value = U4[row * 4 + col];
            G4_local_buggy.put(blockid, {&value, 1});
        }
        sch_local.execute();
        
        print_4d_tensor(G4_local_buggy, "G4_local_buggy (Result of Buggy Method)");
        sch_local.deallocate(G4_local_buggy).execute();
    } // G4_local_buggy is now out of scope


    // --- 2. DEMONSTRATE THE CORRECT SOLUTION ---
    if(ec_local.pg().rank() == 0) {
      std::cout << "\n>>> 2. Populating G4_local_correct with the corrected 'pre-build buffer' method..." << std::endl;
    }
    
    { // Use a new scope for the correct tensor
        Tensor<Cplx> G4_local_correct({phys_tis, phys_tis, phys_tis, phys_tis});
        G4_local_correct.set_dense();
        sch_local.allocate(G4_local_correct).execute();

        // The robust strategy: Build a single host buffer with all 16 values first.
        std::vector<Cplx> full_g4_buffer(G4_local_correct.size());
        size_t c = 0;
        for (int p0p = 0; p0p < 2; ++p0p) {
            for (int p1p = 0; p1p < 2; ++p1p) {
                for (int p0_in = 0; p0_in < 2; ++p0_in) {
                    for (int p1_in = 0; p1_in < 2; ++p1_in, ++c) {
                        int row = p0p * 2 + p1p;
                        int col = p0_in * 2 + p1_in;
                        full_g4_buffer[c] = U4[row * 4 + col];
                    }
                }
            }
        }

        // Perform a SINGLE put operation for the entire tensor.
        G4_local_correct.put(*(G4_local_correct.loop_nest().begin()), full_g4_buffer);
        sch_local.execute();

        print_4d_tensor(G4_local_correct, "G4_local_correct (Result of Correct Method)");
        sch_local.deallocate(G4_local_correct).execute();
    } // G4_local_correct is now out of scope

    pg_local.destroy_coll();
    tamm::finalize();
    return 0;
}
