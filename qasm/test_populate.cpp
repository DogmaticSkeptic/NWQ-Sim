#include <iostream>
#include <vector>
#include <complex>
#include <mpi.h>
#include <tamm/tamm.hpp>

using Cplx = std::complex<double>;
using Tensor = tamm::Tensor<Cplx>;

// A helper function to print the contents of a local vector
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
        tamm::finalize(); MPI_Finalize(); return 1;
    }

    // -- SETUP --
    // Corresponds to the state of mps_tensors BEFORE a layer
    tamm::TiledIndexSpace TIS{tamm::IndexSpace{tamm::range(2)}};
    Tensor old_T0{TIS, TIS};
    Tensor old_T1{TIS, TIS};
    sch.allocate(old_T0, old_T1)(old_T0() = 0.0)(old_T1() = 0.0).execute();
    if (rank == 0) std::cout << "Step 1: Initialized two 'old' distributed tensors." << std::endl;

    // -- SIMULATING apply_collective_updates --

    // Step 2: Metadata exchange (simulated). All ranks learn what needs to be done.
    // In this test, Rank 1 owns the update for T0, and Rank 2 owns the update for T1.
    struct UpdateMeta { int owner_T0 = 1; int owner_T1 = 2; };
    UpdateMeta meta;
    if (rank == 0) std::cout << "Step 2: Metadata exchanged. Rank 1 will update T0, Rank 2 will update T1." << std::endl;

    // Step 3: Collective Deallocation and Reallocation. ALL ranks participate.
    Tensor new_T0{TIS, TIS};
    Tensor new_T1{TIS, TIS};
    sch.deallocate(old_T0, old_T1).allocate(new_T0, new_T1).execute();
    if (rank == 0) std::cout << "Step 3: All ranks collectively deallocated old tensors and allocated new ones." << std::endl;

    ec.pg().barrier();

    // Step 4: Data Population. ONLY owner ranks perform one-sided PUTs.
    if (rank == meta.owner_T0) {
        std::vector<Cplx> data_for_T0 = {Cplx{1,0}, Cplx{1,0}, Cplx{1,0}, Cplx{1,0}};
        print_vector(rank, "My local data for new_T0", data_for_T0);
        std::cout << "[RANK " << rank << "] Issuing one-sided PUT for new_T0..." << std::endl;
        new_T0.put({0,0}, data_for_T0);
    }
    if (rank == meta.owner_T1) {
        std::vector<Cplx> data_for_T1 = {Cplx{2,0}, Cplx{2,0}, Cplx{2,0}, Cplx{2,0}};
        print_vector(rank, "My local data for new_T1", data_for_T1);
        std::cout << "[RANK " << rank << "] Issuing one-sided PUT for new_T1..." << std::endl;
        new_T1.put({0,0}, data_for_T1);
    }

    // Step 5: SYNCHRONIZATION (The missing piece)
    // This barrier forces all ranks to wait until the one-sided PUT operations from
    // Ranks 1 and 2 are globally visible and complete.
    
    //
    // *** COMMENT OUT THE LINE BELOW TO SEE THE BUG IN ACTION ***
    //
    ec.pg().barrier();
    //
    // **********************************************************


    // Step 6: Verification. Rank 0 (a non-owner) reads back the data.
    if (rank == 0) {
        std::cout << "\n--- Verification on Rank 0 ---" << std::endl;
        
        std::vector<Cplx> data_from_T0(4);
        std::vector<Cplx> data_from_T1(4);

        new_T0.get({0,0}, data_from_T0);
        new_T1.get({0,0}, data_from_T1);

        print_vector(rank, "Data I received from new_T0", data_from_T0);
        print_vector(rank, "Data I received from new_T1", data_from_T1);

        bool success_T0 = (data_from_T0[0] == Cplx{1,0});
        bool success_T1 = (data_from_T1[0] == Cplx{2,0});

        if (success_T0 && success_T1) {
            std::cout << "SUCCESS: Rank 0 correctly received updates from both Rank 1 and Rank 2." << std::endl;
        } else {
            std::cout << "FAILURE: Rank 0 read stale data. The updates were not synchronized." << std::endl;
        }
        std::cout << "------------------------------" << std::endl;
    }
    
    // Cleanup
    sch.deallocate(new_T0, new_T1).execute();
    tamm::finalize();
    MPI_Finalize();

    return 0;
}
