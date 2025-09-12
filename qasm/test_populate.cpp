#include <iostream>
#include <vector>
#include <complex>
#include <numeric>
#include <cstring> // For memcpy
#include <mpi.h>
#include <tamm/tamm.hpp>

// Simplified structures from your TN_TAMM code
struct LocalGateResult {
    bool is_valid = false;
    int owner_rank = -1;
    size_t new_bond_dim = 0;
    std::vector<std::complex<double>> new_T0_data;
};

struct GateUpdateMetadata {
    bool is_valid = false;
    int owner_rank = -1;
    size_t new_bond_dim = 0;
};


// Boilerplate and helper functions
using Cplx = std::complex<double>;
using Tensor = t::Tensor<Cplx>;

void print_local_data(int rank, const std::string& name, const std::vector<Cplx>& vec) {
    std::cout << "[RANK " << rank << "] " << name << ": [ ";
    for (const auto& val : vec) {
        std::cout << val.real() << " ";
    }
    std::cout << "]" << std::endl;
}

// Verification function executed by all ranks
bool verify_tensor_data(Tensor& t, size_t N) {
    bool local_success = true;
    int rank = t.execution_context()->pg().rank().value();

    std::cout << "[RANK " << rank << "] --- VERIFICATION PHASE ---" << std::endl;

    for (const auto& blockid : t.loop_nest()) {
        auto [owner_rank, offset] = t.distribution().locate(blockid);

        if (owner_rank == t.execution_context()->pg().rank()) {
            std::vector<Cplx> buf(t.block_size(blockid));
            t.get(blockid, buf);
            
            std::cout << "[RANK " << rank << "]   Verifying my local Block " << blockid << "... ";

            auto block_dims = t.block_dims(blockid);
            auto block_offsets = t.block_offsets(blockid);
            bool block_ok = true;
            size_t c = 0;
            for (size_t i = block_offsets[0]; i < block_offsets[0] + block_dims[0]; ++i) {
                for (size_t j = block_offsets[1]; j < block_offsets[1] + block_dims[1]; ++j) {
                    double expected_val = i * N + j + 1.0;
                    if (std::abs(buf[c].real() - expected_val) > 1e-9) {
                        block_ok = false;
                        break;
                    }
                    c++;
                }
                if (!block_ok) break;
            }
            
            if(block_ok) {
                std::cout << "Correct." << std::endl;
            } else {
                std::cout << "INCORRECT!" << std::endl;
                print_local_data(rank, "    Received", buf);
                local_success = false;
            }
        }
    }
    std::cout << "[RANK " << rank << "] --- VERIFICATION END ---" << std::endl;
    return local_success;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    tamm::initialize(argc, argv);

    tamm::ProcGroup pg_world = tamm::ProcGroup::create_world_coll();
    tamm::ExecutionContext ec{pg_world, tamm::DistributionKind::nw, tamm::MemoryManagerKind::ga};
    tamm::Scheduler sch{ec};

    int rank = ec.pg().rank().value();
    int pg_size = ec.pg().size().value();
    
    std::cout << "[RANK " << rank << "] Initialized. World size is " << pg_size << "." << std::endl;

    if (pg_size < 2) {
        if (rank == 0) std::cerr << "ERROR: This test requires at least 2 MPI ranks." << std::endl;
    } else {
        // -- 1. SIMULATE LOCAL COMPUTATION --
        int owner_rank = 1;
        std::vector<LocalGateResult> local_results;
        size_t new_dim_size = 4;

        std::cout << "[RANK " << rank << "] Entering Phase 1: Simulating local computation on owner_rank " << owner_rank << "." << std::endl;
        if (rank == owner_rank) {
            LocalGateResult res;
            res.is_valid = true;
            res.owner_rank = rank;
            res.new_bond_dim = new_dim_size;
            res.new_T0_data.resize(new_dim_size * new_dim_size);
            for(size_t i=0; i < res.new_T0_data.size(); ++i) {
                res.new_T0_data[i] = Cplx{double(i + 1.0), 0.0};
            }
            local_results.push_back(res);
            std::cout << "[RANK " << rank << "] I am the owner. I have computed a local result with new_bond_dim = " << res.new_bond_dim << std::endl;
        }
        std::cout << "[RANK " << rank << "] Finished Phase 1." << std::endl;

        // -- 2. METADATA EXCHANGE (from apply_collective_updates) --
        std::cout << "\n[RANK " << rank << "] Entering Phase 2: Exchanging metadata via MPI_Allgather(v)." << std::endl;
        
        std::vector<GateUpdateMetadata> local_metadata;
        for(const auto& res : local_results) {
            local_metadata.push_back({res.is_valid, res.owner_rank, res.new_bond_dim});
        }
        std::cout << "[RANK " << rank << "] Preparing to send " << local_metadata.size() << " metadata record(s)." << std::endl;
        
        std::vector<int> all_counts(pg_size);
        int local_count = local_metadata.size();
        MPI_Allgather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, pg_world.comm());
        
        std::cout << "[RANK " << rank << "] MPI_Allgather complete. Received counts: [ ";
        for(int c : all_counts) std::cout << c << " ";
        std::cout << "]" << std::endl;

        std::vector<int> displacements(pg_size, 0);
        int total_updates = 0;
        for(int i=0; i<pg_size; ++i) {
            displacements[i] = total_updates;
            total_updates += all_counts[i];
        }

        std::vector<GateUpdateMetadata> all_metadata(total_updates);
        MPI_Datatype mpi_meta_type;
        MPI_Type_contiguous(sizeof(GateUpdateMetadata), MPI_BYTE, &mpi_meta_type);
        MPI_Type_commit(&mpi_meta_type);
        MPI_Allgatherv(local_metadata.data(), local_count, mpi_meta_type,
                       all_metadata.data(), all_counts.data(), displacements.data(),
                       mpi_meta_type, pg_world.comm());
        MPI_Type_free(&mpi_meta_type);
        
        std::cout << "[RANK " << rank << "] MPI_Allgatherv complete. I now have " << all_metadata.size() << " total metadata record(s)." << std::endl;
        for(size_t i=0; i<all_metadata.size(); ++i) {
             std::cout << "[RANK " << rank << "]   Global Meta [" << i << "]: owner=" << all_metadata[i].owner_rank 
                       << ", new_bond_dim=" << all_metadata[i].new_bond_dim << std::endl;
        }
        std::cout << "[RANK " << rank << "] Finished Phase 2." << std::endl;
        ec.pg().barrier();
        
        // -- 3. DYNAMIC TENSOR REALLOCATION --
        std::cout << "\n[RANK " << rank << "] Entering Phase 3: Collectively allocating the new distributed tensor." << std::endl;
        
        const GateUpdateMetadata& meta = all_metadata[0];
        const size_t TILE_SIZE = 2;
        std::cout << "[RANK " << rank << "] Creating TiledIndexSpace with range(" << meta.new_bond_dim << ") and tile_size(" << TILE_SIZE << ")." << std::endl;
        tamm::TiledIndexSpace TIS_new{tamm::IndexSpace{tamm::range(meta.new_bond_dim)}, TILE_SIZE};
        Tensor global_tensor{TIS_new, TIS_new};
        
        sch.allocate(global_tensor).execute();
        std::cout << "[RANK " << rank << "] New global tensor allocated. Distribution:" << std::endl;
        for (const auto& blockid : global_tensor.loop_nest()) {
             auto [owner, offset] = global_tensor.distribution().locate(blockid);
             std::cout << "[RANK " << rank << "]   Block " << blockid << " is owned by RANK " << owner << std::endl;
        }
        std::cout << "[RANK " << rank << "] Finished Phase 3." << std::endl;
        ec.pg().barrier();

        // -- 4. THE FAULTY ONE-SIDED PUT --
        std::cout << "\n[RANK " << rank << "] Entering Phase 4: Owner rank performs one-sided PUT." << std::endl;

        if (rank == meta.owner_rank) {
             std::cout << "[RANK " << rank << "] >> I AM THE OWNER. EXECUTING PUT NOW. <<" << std::endl;
             print_local_data(rank, "   Data being sent", local_results[0].new_T0_data);
             global_tensor.put({0,0}, local_results[0].new_T0_data);
             std::cout << "[RANK " << rank << "] >> PUT call has returned. <<" << std::endl;
        } else {
             std::cout << "[RANK " << rank << "] I am not the owner. I am skipping the PUT call." << std::endl;
        }
        std::cout << "[RANK " << rank << "] Finished Phase 4." << std::endl;


        // -- SYNCHRONIZATION --
        std::cout << "[RANK " << rank << "] --- Reaching critical synchronization barrier ---" << std::endl;
        ec.pg().barrier();
        std::cout << "[RANK " << rank << "] --- Passed critical synchronization barrier ---" << std::endl;
        
        // -- 5. VERIFICATION --
        bool local_test_result = verify_tensor_data(global_tensor, new_dim_size);
        
        int global_success_flag = local_test_result ? 1 : 0;
        int final_result = 0;
        MPI_Allreduce(&global_success_flag, &final_result, 1, MPI_INT, MPI_MIN, pg_world.comm());

        if (rank == 0) {
            std::cout << "\n--- FINAL RESULT ---" << std::endl;
            if (final_result == 1) {
                std::cout << "SUCCESS: All ranks reported correct data." << std::endl;
            } else {
                std::cout << "FAILURE: At least one rank reported incorrect data. The data corruption bug is faithfully reproduced." << std::endl;
            }
            std::cout << "--------------------" << std::endl;
        }
        
        sch.deallocate(global_tensor).execute();
    } 

    tamm::finalize();
    MPI_Finalize();
    return 0;
}
