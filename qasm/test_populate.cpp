#include <iostream>
#include <vector>
#include <complex>
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    tamm::initialize(argc, argv);

    tamm::ProcGroup pg = tamm::ProcGroup::create_world_coll();
    tamm::ExecutionContext ec{pg, tamm::DistributionKind::nw, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch{ec};

    int rank = ec.pg().rank().value();
    int size = ec.pg().size().value();

    if (size < 3) {
        if (rank == 0) std::cerr << "ERROR: This test requires at least 3 MPI ranks." << std::endl;
        // Don't return early. Let all ranks finalize properly.
    } else {
        // -- SETUP --
        // **FIX**: Specify the tile size to be 2. This creates a TiledIndexSpace
        // where the entire 2x2 space is a single block.
        tamm::TiledIndexSpace TIS{tamm::IndexSpace{tamm::range(2)}, /*tile_size=*/2};
        
        Tensor old_T0{TIS, TIS};
        Tensor old_T1{TIS, TIS};
        sch.allocate(old_T0, old_T1)(old_T0() = 0.0)(old_T1() = 0.0).execute();
        if (rank == 0) std::cout << "Step 1: Initialized two 'old' distributed tensors." << std::endl;

        // -- SIMULATING apply_collective_updates --
        struct UpdateMeta { int owner_T0 = 1; int owner_T1 = 2; };
        UpdateMeta meta;
        if (rank == 0) std::cout << "Step 2: Metadata exchanged. Rank 1 will update T0, Rank 2 will update T1." << std::endl;

        Tensor new_T0{TIS, TIS};
        Tensor new_T1{TIS, TIS};
        sch.deallocate(old_T0, old_T1).allocate(new_T0, new_T1).execute();
        if (rank == 0) std::cout << "Step 3: All ranks collectively deallocated/reallocated tensors." << std::endl;

        ec.pg().barrier();

        if (rank == meta.owner_T0) {
            std::vector<Cplx> data_for_T0 = {Cplx{1,0}, Cplx{1,0}, Cplx{1,0}, Cplx{1,0}};
            print_vector(rank, "My local data for new_T0", data_for_T0);
            std::cout << "[RANK " << rank << "] Issuing one-sided PUT for new_T0..." << std::endl;
            // Now, {0,0} refers to the entire 2x2 block, so all 4 elements are written.
            new_T0.put({0,0}, data_for_T0);
        }
        if (rank == meta.owner_T1) {
            std::vector<Cplx> data_for_T1 = {Cplx{2,0}, Cplx{2,0}, Cplx{2,0}, Cplx{2,0}};
            print_vector(rank, "My local data for new_T1", data_for_T1);
            std::cout << "[RANK " << rank << "] Issuing one-sided PUT for new_T1..." << std::endl;
            new_T1.put({0,0}, data_for_T1);
        }

        // The barrier is still essential for synchronization.
        ec.pg().barrier();
        
        // -- VERIFICATION --
        if (rank == 0) {
            std::cout << "\n--- Verification on Rank 0 ---" << std::endl;
            
            std::vector<Cplx> data_from_T0(4);
            std::vector<Cplx> data_from_T1(4);

            new_T0.get({0,0}, data_from_T0);
            new_T1.get({0,0}, data_from_T1);

            print_vector(rank, "Data I received from new_T0", data_from_T0);
            print_vector(rank, "Data I received from new_T1", data_from_T1);

            // A more robust success check
            bool success = true;
            for(const auto& v : data_from_T0) if(v != Cplx{1,0}) success = false;
            for(const auto& v : data_from_T1) if(v != Cplx{2,0}) success = false;
            
            if (success) {
                std::cout << "SUCCESS: Rank 0 received the complete and correct data." << std::endl;
            } else {
                std::cout << "FAILURE: Rank 0 received incorrect or partial data." << std::endl;
            }
            std::cout << "------------------------------" << std::endl;
        }
        
        sch.deallocate(new_T0, new_T1).execute();
    } // End of the main logic block

    // Cleanup
    tamm::finalize();
    MPI_Finalize();

    return 0;
}
