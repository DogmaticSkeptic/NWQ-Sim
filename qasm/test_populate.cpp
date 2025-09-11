#include <tamm/tamm.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include <array>

// Use shorter aliases for convenience
using Cplx = std::complex<double>;
using Tensor = tamm::Tensor<Cplx>;

/**
 * @brief Prints the contents of a 4-dimensional local TAMM tensor.
 * 
 * @param t The tensor to print.
 * @param name A descriptive name for the tensor.
 */
void print_4d_tensor_data(Tensor& t, const std::string& name) {
    // This function assumes it is only ever called by rank 0 on a local tensor.
    std::vector<Cplx> buf(t.size());
    t.get(*(t.loop_nest().begin()), buf);

    auto dims = t.tiled_index_spaces();
    size_t d1 = dims[0].index_space().num_indices();
    size_t d2 = dims[1].index_space().num_indices();
    size_t d3 = dims[2].index_space().num_indices();
    size_t d4 = dims[3].index_space().num_indices();
    
    std::cout << "\n--- Contents of Tensor: " << name << " ("
              << d1 << "x" << d2 << "x" << d3 << "x" << d4 << ") ---" << std::endl;

    std::cout << "[ ";
    for (size_t i = 0; i < buf.size(); ++i) {
        // Print small numbers as zero for clarity
        double real_part = std::abs(buf[i].real()) < 1e-9 ? 0.0 : buf[i].real();
        double imag_part = std::abs(buf[i].imag()) < 1e-9 ? 0.0 : buf[i].imag();
        std::cout << "(" << std::fixed << std::setprecision(3) << real_part << "," << imag_part << ") ";
    }
    std::cout << "]" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;
}

int main(int argc, char* argv[]) {
    tamm::initialize(argc, argv);

    // Get the global communicator for all processes launched by srun
    tamm::ProcGroup pg_world = tamm::ProcGroup::create_world_coll();

    // We only want ONE process to run the test to avoid MPI conflicts.
    if (pg_world.rank() == 0) {
        std::cout << ">>> Standalone TAMM Contraction Test (Running on Rank 0) <<<" << std::endl;
        std::cout << ">>> Replicating the M2 = G4 * M calculation." << std::endl;

        // 1. Set up a local execution context, containing only this one process (rank 0).
        tamm::ProcGroup pg_local = tamm::ProcGroup::create_self();
        tamm::ExecutionContext ec_local{pg_local, tamm::DistributionKind::dense, tamm::MemoryManagerKind::local};
        tamm::Scheduler sch_local{ec_local};

        // 2. Define the TiledIndexSpaces needed for the tensors
        tamm::TiledIndexSpace l_tis{tamm::IndexSpace{tamm::range(1)}}; // Left bond dim
        tamm::TiledIndexSpace r_tis{tamm::IndexSpace{tamm::range(1)}}; // Right bond dim
        tamm::TiledIndexSpace p_tis{tamm::IndexSpace{tamm::range(2)}}; // Physical dim (qubit)

        // 3. Create and allocate the three tensors in the local context
        Tensor G4_local({p_tis, p_tis, p_tis, p_tis});
        Tensor M_local({l_tis, p_tis, p_tis, r_tis});
        Tensor M2_local_result({l_tis, p_tis, p_tis, r_tis});
        
        // set_dense() is good practice before allocating
        G4_local.set_dense();
        M_local.set_dense();
        M2_local_result.set_dense();

        sch_local.allocate(G4_local, M_local, M2_local_result).execute();

        // 4. Populate the input tensors
        
        // Populate G4_local with the exact gate matrix from the failed run
        std::array<Cplx, 16> U4 = {
            Cplx(-0.308, 0.204), Cplx(-0.051, 0.398), Cplx(-0.562,-0.253), Cplx(-0.444, 0.354),
            Cplx( 0.176,-0.540), Cplx( 0.340, 0.404), Cplx( 0.274, 0.293), Cplx(-0.475, 0.108),
            Cplx(-0.359,-0.442), Cplx(-0.556, 0.261), Cplx(-0.052, 0.307), Cplx( 0.398, 0.206),
            Cplx(-0.441,-0.147), Cplx( 0.181,-0.387), Cplx(-0.391, 0.458), Cplx(-0.236,-0.428)
        };
        
        auto fill_g4 = [&](const tamm::IndexVector& bid, tamm::span<Cplx> buf){
            auto block_dims = G4_local.block_dims(bid);
            size_t d1 = block_dims[0];
            size_t d2 = block_dims[1];
            size_t d3 = block_dims[2];
            size_t d4 = block_dims[3];

            size_t c = 0; // Counter for the 1D buffer 'buf'
            for (size_t p0p = 0; p0p < d1; ++p0p) {
                for (size_t p1p = 0; p1p < d2; ++p1p) {
                    for (size_t p0_in = 0; p0_in < d3; ++p0_in) {
                        for (size_t p1_in = 0; p1_in < d4; ++p1_in) {
                            size_t row = p0p * 2 + p1p;
                            size_t col = p0_in * 2 + p1_in;
                            buf[c++] = U4[row * 4 + col];
                        }
                    }
                }
            }
        };
        // Use fill_tensor for initialization instead of update_tensor
        tamm::fill_tensor(G4_local, fill_g4);

        // Populate M_local to represent the |00> state vector [1, 0, 0, 0]
        Cplx one{1.0, 0.0};
        sch_local(M_local() = 0.0).execute(); // Zero out the tensor first
        M_local.put({0,0,0,0}, {&one, 1}); // Place the single non-zero element

        // Print the inputs to verify they are correct
        print_4d_tensor_data(G4_local, "G4_local (Input Gate)");
        print_4d_tensor_data(M_local, "M_local (Input State)");

        // 5. Perform the problematic contraction
        std::cout << "\n>>> Performing contraction: M2(l,p0',p1',r) = G4(p0',p1',p0,p1) * M(l,p0,p1,r)\n";
        sch_local(M2_local_result("l", "p0p", "p1p", "r") = G4_local("p0p", "p1p", "p0", "p1") * M_local("l", "p0", "p1", "r")).execute();

        // 6. Print the final result
        print_4d_tensor_data(M2_local_result, "M2_local_result (Actual Output)");

        // 7. For comparison, manually print the expected result
        std::cout << "\n--- For Reference: Expected Correct Output ---" << std::endl;
        std::cout << "[ (-0.308,0.204) (0.176,-0.540) (-0.359,-0.442) (-0.441,-0.147) ]" << std::endl;
        std::cout << "---------------------------------------------------------" << std::endl;

        // 8. Clean up local resources
        sch_local.deallocate(G4_local, M_local, M2_local_result).execute();
        pg_local.destroy_coll();
    }

    // All processes must participate in the final barrier and cleanup
    pg_world.barrier();
    pg_world.destroy_coll();

    tamm::finalize();
    return 0;
}
