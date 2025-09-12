#include <iostream>
#include <vector>
#include <complex>
#include <numeric> // For std::iota
#include <mpi.h>
#include <tamm/tamm.hpp>

using Cplx = std::complex<double>;
using Tensor = tamm::Tensor<Cplx>;

void print_vector(int rank, const std::string& name, const std::vector<Cplx>& vec) {
    std::cout << "[RANK " << rank << "] " << name << ": [ ";
    for (const auto& val : vec) {
        std::cout << "(" << val.real() << "," << val.imag() << ") ";
    }
    std::cout << "]" << std::endl;
}

// Helper to print the contents of a distributed tensor from a single rank (usually 0)
void print_distributed_tensor(Tensor& t, int rank_to_print_from) {
    if (t.execution_context()->pg().rank().value() != rank_to_print_from) return;

    std::cout << "\n--- Verifying contents of distributed tensor on RANK " << rank_to_print_from << " ---" << std::endl;
    for (const auto& blockid : t.loop_nest()) {
        std::vector<Cplx> buf(t.block_size(blockid));
        t.get(blockid, buf);
        std::cout << "  Block " << blockid << ": [ ";
        for(const auto& val : buf) {
            std::cout << "(" << val.real() << "," << val.imag() << ") ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "--------------------------------------------------------" << std::endl;
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    tamm::initialize(argc, argv);

    tamm::ProcGroup pg = tamm::ProcGroup::create_world_coll();
    tamm::ExecutionContext ec{pg, tamm::DistributionKind::nw, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch{ec};

    int rank = ec.pg().rank().value();
    int size = ec.pg().size().value();

    if (size < 2) {
        if (rank == 0) std::cerr << "ERROR: This test requires at least 2 MPI ranks to demonstrate distribution." << std::endl;
    } else {
        // -- SETUP --
        // Use a small TiledIndexSpace for the 'old' tensor. It's not distributed.
        tamm::TiledIndexSpace TIS_old{tamm::IndexSpace{tamm::range(1)}, 1}; 
        Tensor old_T0{TIS_old};
        sch.allocate(old_T0)(old_T0() = 0.0).execute();
        if (rank == 0) std::cout << "Step 1: Initialized 'old' tensor." << std::endl;

        // -- SIMULATING apply_collective_updates --
        struct UpdateMeta { int owner = 1; };
        UpdateMeta meta; // All ranks have this metadata.
        if (rank == 0) std::cout << "Step 2: Metadata exchanged. Rank " << meta.owner << " will update the new tensor." << std::endl;
        
        // **KEY CHANGE**: Create a TiledIndexSpace that WILL be tiled and distributed.
        // Index space of size 4, with each tile being of size 2. This creates 2 tiles.
        // A 2D tensor will therefore have 2x2 = 4 blocks, which will be distributed.
        const size_t N = 4;
        const size_t TILE_SIZE = 2;
        tamm::TiledIndexSpace TIS_new{tamm::IndexSpace{tamm::range(N)}, TILE_SIZE};

        Tensor new_T0{TIS_new, TIS_new};

        sch.deallocate(old_T0).allocate(new_T0).execute();
        if (rank == 0) std::cout << "Step 3: All ranks collectively deallocated 'old' and allocated 'new' distributed tensor." << std::endl;

        ec.pg().barrier();

        // **THE ACTION**: Only the owner rank prepares the full data and issues a single PUT.
        if (rank == meta.owner) {
            std::vector<Cplx> data_for_T0(N * N);
            // Fill with a recognizable pattern, e.g., 1.0, 2.0, 3.0 ...
            for(size_t i=0; i<data_for_T0.size(); ++i) data_for_T0[i] = Cplx{double(i+1), 0.0};
            
            print_vector(rank, "My local data for the ENTIRE new_T0", data_for_T0);
            std::cout << "[RANK " << rank << "] Issuing one-sided PUT for new_T0..." << std::endl;
            
            // This 'put' should write the contiguous `data_for_T0` into the memory
            // of the distributed `new_T0` tensor, wherever its blocks may live.
            // This is the operation that is failing in the main code.
            new_T0.put({}, data_for_T0); // {} means put to the whole tensor
        }

        // This barrier is essential. It ensures that all ranks wait until the owner's PUT
        // operation has completed before they proceed to the verification step.
        ec.pg().barrier();
        
        // -- VERIFICATION --
        // Print the tensor's contents from each rank's perspective.
        // We expect to see incorrect (zero) values on ranks that are not Rank 0.
        print_distributed_tensor(new_T0, 0);
        print_distributed_tensor(new_T0, 1);
        
        if (rank == 0) {
            bool success = true;
            size_t count = 1;
            for (const auto& blockid : new_T0.loop_nest()) {
                std::vector<Cplx> buf(new_T0.block_size(blockid));
                new_T0.get(blockid, buf);
                for(const auto& val : buf) {
                    if (val != Cplx{double(count++), 0.0}) {
                        success = false;
                        break;
                    }
                }
                if (!success) break;
            }

            if (success) {
                std::cout << "\nSUCCESS: Rank 0 verified the complete and correct data." << std::endl;
            } else {
                std::cout << "\nFAILURE: Rank 0 received incorrect or partial data. This reproduces the bug." << std::endl;
            }
        }
        
        sch.deallocate(new_T0).execute();
    } 

    tamm::finalize();
    MPI_Finalize();
    return 0;
}
