#pragma once

#include <mpi.h>

#include "../state.hpp"

#include "../nwq_util.hpp"
#include "../gate.hpp"
#include "../circuit.hpp"
#include "../config.hpp"
#include "private/exp_gate_declarations_host.hpp"

#include "../circuit_pass/fusion.hpp"
#include "../private/macros.hpp"
#include "../private/sim_gate.hpp"

#include <random>
#include <vector>
#include <string>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <sstream>

#include <tamm/errors.hpp>
#include <tamm/symbol.hpp>
#include <tamm/tensor.hpp>
#include <tamm/tiled_index_space.hpp>
#include <tamm/index_space.hpp>
#include <tamm/execution_context.hpp>
#include <tamm/scheduler.hpp>
#include <tamm/tamm_io.hpp>

#include <tamm/rmm_memory_pool.hpp>
#include <tamm/gpu_streams.hpp>

#include <tamm/tamm.hpp>

#include <gsl/span>
#include <iostream>

#include <complex>
#include <unordered_map>
#include <cstring>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <Eigen/Dense>

namespace NWQSim
{

    using Cplx = std::complex<ValType>;

    class TN_TAMM;

    struct LocalGateResult {
        bool is_valid = false;
        IdxType q0, q1;
        IdxType new_bond_dim;
        std::vector<Cplx> new_T0_data; // Raw data, not a TAMM tensor
        std::vector<Cplx> new_T1_data; // Raw data, not a TAMM tensor
        int original_rank;
    };

    struct GateUpdateResult {
        bool is_valid = false;
        IdxType q0, q1;
        tamm::Tensor<Cplx> new_T0_local;
        tamm::Tensor<Cplx> new_T1_local;
    };

    struct GateUpdateMetadata {
        bool is_valid = false;
        IdxType q0, q1;
        IdxType new_bond_dim;
        int original_rank; // The rank that computed this result
    };

    struct CuCtx {
        cusolverDnHandle_t solver = nullptr;
        cudaStream_t stream = nullptr;
        gesvdjInfo_t jp = nullptr;
        int lwork_jac = 0;
        cuDoubleComplex* d_work_jac = nullptr;

        CuCtx() {
            cusolverDnCreate(&solver);
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
            cusolverDnSetStream(solver, stream);
            cusolverDnCreateGesvdjInfo(&jp);
        }
        ~CuCtx() {
            if(d_work_jac) cudaFree(d_work_jac);
            if(jp) cusolverDnDestroyGesvdjInfo(jp);
            if(solver) cusolverDnDestroy(solver);
            if(stream) cudaStreamDestroy(stream);
        }
    };

    using Eigen::Index;
    class TN_TAMM : public QuantumState
    {
    public:
        TN_TAMM(IdxType n_qubits,
                IdxType max_bond_dim = 100,
                double sv_cutoff = 0.0,
                std::string backend = "TN_TAMM_CPU")
        : QuantumState(SimType::TN),
            n_qubits(n_qubits),
            block_size(2048),
            max_bond_dim(max_bond_dim),
            sv_cutoff(sv_cutoff),
            pg(init_pg()),
            ec(pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga)
        {
            // set the i_proc to prevent repeat prints
            i_proc = pg.rank().value();
            
            // print the tamm execution context, this can be commented out if desired
            if(ec.print()) {
                auto current_time   = std::chrono::system_clock::now();
                auto current_time_t = std::chrono::system_clock::to_time_t(current_time);
                auto cur_local_time = localtime(&current_time_t);
                std::cout << std::endl << "date: " << std::put_time(cur_local_time, "%c") << std::endl;
                std::cout << "nnodes: " << ec.nnodes() << ", ";
                std::cout << "nproc_per_node: " << ec.ppn() << ", ";
                std::cout << "nproc_total: " << ec.nnodes() * ec.ppn() << ", ";
                if(ec.has_gpu()) {
                  std::cout << "ngpus_per_node: " << ec.gpn() << ", ";
                  std::cout << "ngpus_total: " << ec.nnodes() * ec.gpn() << std::endl;
                }
                std::cout << std::endl;
                ec.print_mem_info();
                std::cout << std::endl;
            }
            // change between cpu or gpu tensor contraction
            if (backend == "TN_TAMM_CPU")
            {
                exec_hw = tamm::ExecutionHW::CPU;
            }
            else if(backend == "TN_TAMM_GPU")
            {
                exec_hw = tamm::ExecutionHW::GPU;
            }

            // initialize bond index spaces
            bond_tis.resize(n_qubits + 1);
            bond_dims.resize(n_qubits + 1);
            for (IdxType i = 0; i <= n_qubits; ++i)
            {
                bond_dims[i] = 1;
                tamm::IndexSpace is{ tamm::range(1) };
                bond_tis[i] = tamm::TiledIndexSpace(is, block_size);
            }
        
            // initialize physical index spaces
            phys_tis.resize(n_qubits);
            phys_dims.resize(n_qubits);
            for (IdxType i = 0; i < n_qubits; ++i)
            {
                phys_dims[i] = 2;
                tamm::IndexSpace is{ tamm::range(2) };
                phys_tis[i] = tamm::TiledIndexSpace(is, 1);
            }
        
            // allocate and initialize MPS tensors
            mps_tensors.reserve(n_qubits);
            for (IdxType i = 0; i < n_qubits; ++i)
            {
                tamm::Tensor<Cplx> T({ bond_tis[i], phys_tis[i], bond_tis[i + 1] });
                T.set_dense();
                T.allocate(&ec);
        
                for (const auto & blockid : T.loop_nest())
                {
                    const size_t block_size = T.block_size(blockid);
                    std::vector<Cplx> hostbuf(block_size);
        
                    auto dims = T.block_dims(blockid);
                    auto offsets = T.block_offsets(blockid);
        
                    for (size_t idx = 0; idx < block_size; ++idx)
                    {
                        size_t rem = idx;
                        size_t l_loc = rem % dims[0];
                        rem /= dims[0];
                        size_t p_loc = rem % dims[1];
                        rem /= dims[1];
                        size_t r_loc = rem % dims[2];
        
                        IdxType l = offsets[0] + l_loc;
                        IdxType p = offsets[1] + p_loc;
                        IdxType r = offsets[2] + r_loc;
        
                        hostbuf[idx] = (l == 0 && p == 0 && r == 0)
                                       ? Cplx{1.0, 0.0}
                                       : Cplx{0.0, 0.0};
                    }
        
                    T.put(blockid, hostbuf);
                }
        
                mps_tensors.push_back(std::move(T));
            }
        }

        ~TN_TAMM() noexcept override 
        {
            SAFE_FREE_HOST(results);
        }

        void reset_state() override {
            //printf("Inside reset gate\n");
            for(IdxType i = 0; i < n_qubits; ++i)
            {
                auto& T = mps_tensors[i];
                T.loop_nest().iterate([&](auto const& idxs)
                {
                    Cplx v = (idxs[0] == 0 && idxs[1] == 0 && idxs[2] == 0)
                             ? Cplx(1.0,0.0)
                             : Cplx(0.0,0.0);
                    T.put(idxs, gsl::span<Cplx>(&v,1));
                });
            }
        }

        void set_seed(IdxType seed) override
        {
            throw std::runtime_error("TN_TAMM does not use RNG seed, not accessible form cutensornet API\n");
        }

        void set_initial(std::string fpath, std::string format) override
        {
            std::cout << "set function was called" << std::endl;
        }

        void dump_res_state(std::string outpath) override
        {
            std::cout << "dump function was called" << std::endl;
        }

        void sim(std::shared_ptr<NWQSim::Circuit> circuit) override
        {
            // prepare fused single vector gates from the circuit
            IdxType original_gate_count = circuit->num_gates();
            std::vector<SVGate> gates = fuse_circuit_sv(circuit);
            IdxType fused_gate_count = gates.size();
            assert(circuit->num_qubits() == n_qubits);
        
            // execute the simulation kernel
            simulation_kernel(gates);
        }


        IdxType* get_results() override
        {
            return results;
        }

        IdxType measure(IdxType qubit) override
        {
            throw std::runtime_error("TN_TAMM::measure not implemented");
        }

        IdxType* measure_all(IdxType repetition) override 
        {
            MA_GATE(repetition);
            return results; 
        }

        ValType* get_real() const override
        {
            throw std::runtime_error("TN_TAMM::get_real not implemented");
        }

        ValType* get_imag() const override
        {
            throw std::runtime_error("TN_TAMM::get_imag not implemented");
        }

        ValType get_exp_z() override
        {
            throw std::runtime_error("TN_TAMM::get_exp_z() not implemented");
        }

        ValType get_exp_z(const std::vector<size_t>& in_bits) override
        {
            throw std::runtime_error("TN_TAMM::get_exp_z(bits) not implemented");
        }

        void print_res_state() override
        {
            throw std::runtime_error("TN_TAMM::print_res_state not implemented");
        }

        static SVGate make_swap_sv(int a, int b)
        {
            // CORRECTED: Call the constructor directly with the required values.
            // The constructor signature is (OP, qubit, ctrl, data).
            // So we pass 'b' as the qubit and 'a' as the control.
            SVGate s(OP::C2, b, a); 
        
            // The rest of the function remains the same.
            static const ValType real[16] = {
                1,0,0,0,
                0,0,1,0,
                0,1,0,0,
                0,0,0,1
            };
            static const ValType imag[16] = {0}; // All zeros
            memcpy(s.gm_real, real, 16 * sizeof(ValType));
            memcpy(s.gm_imag, imag, 16 * sizeof(ValType));
            return s;
        }
    
        static SVGate make_local_c2_sv(const SVGate& g, int left, int right)
        {
            // CORRECTED: Use the copy constructor to create 't' as a copy of 'g'.
            SVGate t(g); 
        
            // Now, simply modify the fields that need to be different.
            t.ctrl = left;
            t.qubit = right;
            
            // The gm_real and gm_imag are already correct because they were copied from g.
            // No memcpy is needed here.
            return t;
        }
    
        static void place_c1(const SVGate& s,
                                    std::vector<std::vector<SVGate>>& layers,
                                    std::unordered_map<int,int>& last_layer)
        {
            // Place in the layer immediately after this qubit was last used
            int L = last_layer[s.qubit] + 1;
            if (L > static_cast<int>(layers.size())) layers.resize(L);
            layers[L - 1].push_back(s);
            last_layer[s.qubit] = L;
        }
    
        static void place_c2(const SVGate& t, int a, int b,
                                    std::vector<std::vector<SVGate>>& layers,
                                    std::unordered_map<int,int>& last_layer)
        {
            // Place in the layer immediately after BOTH qubits were last used
            int la = last_layer[a];
            int lb = last_layer[b];
            int L = 1 + std::max(la, lb);
            
            SVGate x = t; // Create a mutable copy
            x.ctrl = a;
            x.qubit = b;
    
            if (L > static_cast<int>(layers.size())) layers.resize(L);
            layers[L - 1].push_back(x);
            // Update the last used layer for both qubits
            last_layer[a] = L;
            last_layer[b] = L;
        }
        
        // Optional but good for load balancing within a layer
        static void append_round_robin(const std::vector<SVGate>& layer, std::vector<SVGate>& out)
        {
            std::vector<SVGate> singles;
            std::vector<SVGate> twos;
            singles.reserve(layer.size());
            twos.reserve(layer.size());
            for (const auto& gate : layer)
            {
                if (gate.op_name == OP::C1) singles.push_back(gate);
                else twos.push_back(gate);
            }
            
            size_t i = 0, j = 0;
            bool pick_single = singles.size() >= twos.size();
            while (i < singles.size() || j < twos.size())
            {
                if (pick_single && i < singles.size()) out.push_back(singles[i++]);
                else if (!pick_single && j < twos.size()) out.push_back(twos[j++]);
                
                pick_single = !pick_single;
                
                // In case one vector is exhausted, append the rest of the other
                if (i >= singles.size() && j < twos.size()) out.insert(out.end(), twos.begin() + j, twos.end());
                if (j >= twos.size() && i < singles.size()) out.insert(out.end(), singles.begin() + i, singles.end());
            }
        }


    protected:
        IdxType n_qubits;
        IdxType* results = NULL;
        IdxType max_bond_dim;
        int block_size;
        double sv_cutoff;
        tamm::ExecutionHW exec_hw;

        tamm::ProcGroup pg;
        tamm::ExecutionContext ec;
        std::vector<IdxType> bond_dims;
        std::vector<IdxType> phys_dims;
        std::vector<tamm::TiledIndexSpace> bond_tis;
        std::vector<tamm::TiledIndexSpace> phys_tis;
        std::vector<tamm::Tensor<Cplx>> mps_tensors;
        IdxType* result = nullptr;
        CuCtx cu_ctx_;

        virtual void simulation_kernel(const std::vector<SVGate> &gates)
        {
            int rank = pg.rank().value();
            std::cout << "[RANK " << rank << "] ==> Entering simulation_kernel." << std::endl;
        
            std::vector<std::vector<SVGate>> layers;
            layers.reserve(gates.size());
            std::unordered_map<int,int> last_layer;
            last_layer.reserve(this->n_qubits);
        
            std::cout << "[RANK " << rank << "] simulation_kernel: Starting gate layering for " << gates.size() << " gates." << std::endl;
            // Gate layering logic remains the same...
            for (const auto& g : gates) {
                if (g.op_name == OP::C2) {
                    int a = g.ctrl;
                    int b = g.qubit;
                    if (std::abs(a - b) > 1) {
                        bool reversed = a > b;
                        if (reversed) std::swap(a, b);
                        for (int k = a; k < b - 1; ++k) place_c2(make_swap_sv(k, k + 1), k, k + 1, layers, last_layer);
                        SVGate local_gate = reversed ? make_local_c2_sv(g, b, b-1) : make_local_c2_sv(g, b-1, b);
                        place_c2(local_gate, b - 1, b, layers, last_layer);
                        for (int k = b - 1; k > a; --k) place_c2(make_swap_sv(k - 1, k), k - 1, k, layers, last_layer);
                    } else {
                        place_c2(g, g.ctrl, g.qubit, layers, last_layer);
                    }
                } else if (g.op_name == OP::C1) {
                    place_c1(g, layers, last_layer);
                }
            }
            std::cout << "[RANK " << rank << "] simulation_kernel: Gate layering complete. Created " << layers.size() << " layers." << std::endl;
        
        
            // Execute the scheduled layers
            int layer_idx = 0;
            for (const auto& layer : layers)
            {
                if (layer.empty()) {
                    layer_idx++;
                    continue;
                }
        
                std::cout << "[RANK " << rank << "] simulation_kernel: ---------- STARTING LAYER " << layer_idx << " ----------" << std::endl;
        
                std::vector<SVGate> batch;
                batch.reserve(layer.size());
                append_round_robin(layer, batch);
        
                // STAGE 1: Each rank computes its assigned gates and returns the raw data.
                std::cout << "[RANK " << rank << "] simulation_kernel: Layer " << layer_idx << " has " << batch.size() << " gates. Calling run_gates_parallel..." << std::endl;
                auto local_update_results = run_gates_parallel(batch);
                std::cout << "[RANK " << rank << "] simulation_kernel: Layer " << layer_idx << " returned from run_gates_parallel. This rank has " << local_update_results.size() << " local C2 results." << std::endl;
        
                // Ensure all local computations are finished before the collective update.
                pg.barrier();
        
                // STAGE 2: All ranks participate in updating the global MPS state.
                std::cout << "[RANK " << rank << "] simulation_kernel: Layer " << layer_idx << ": Calling apply_collective_updates..." << std::endl;
                apply_collective_updates(local_update_results);
                std::cout << "[RANK " << rank << "] simulation_kernel: Layer " << layer_idx << " returned from apply_collective_updates." << std::endl;
        
                // Ensure the global MPS is in a consistent state before starting the next layer.
                pg.barrier();
        
                std::cout << "[RANK " << rank << "] simulation_kernel: ---------- FINISHED LAYER " << layer_idx << " ----------" << std::endl;
                layer_idx++;
            }
            std::cout << "[RANK " << rank << "] <== Exiting simulation_kernel." << std::endl;
        }

        std::vector<LocalGateResult> run_gates_parallel(const std::vector<SVGate>& batch)
        {
            int rank = pg.rank().value();
            std::cout << "[RANK " << rank << "] >> Entering run_gates_parallel" << std::endl;
        
            tamm::AtomicCounterGA gate_counter(pg, 1);
            gate_counter.allocate(0);
            pg.barrier();
        
            std::vector<LocalGateResult> local_results;
            tamm::Scheduler sch_global{ec}; // Create one scheduler for all C1 gates on this rank
        
            while (true)
            {
                long long gate_idx = gate_counter.fetch_add(0, 1);
                if (gate_idx >= static_cast<long long>(batch.size())) {
                    break;
                }
        
                const SVGate& g = batch[gate_idx];
                
                if (g.op_name == OP::C1) {
                    std::cout << "[RANK " << rank << "] run_gates_parallel: Gate " << gate_idx << " is a C1 gate on qubit " << g.qubit << ". Scheduling on global scheduler." << std::endl;
                    std::array<Cplx, 4> U;
                    for (int i=0; i<4; ++i) U[i] = Cplx(g.gm_real[i], g.gm_imag[i]);
                    C1_GATE(U, g.qubit, sch_global);
                } else if (g.op_name == OP::C2) {
                    std::cout << "[RANK " << rank << "] run_gates_parallel: Gate " << gate_idx << " is a C2 gate on qubits (" << g.ctrl << ", " << g.qubit << "). Calling C2_GATE_COMPUTE." << std::endl;
                    std::array<Cplx, 16> U4;
                    for (int i=0; i<16; ++i) U4[i] = Cplx(g.gm_real[i], g.gm_imag[i]);
                    local_results.push_back(C2_GATE_COMPUTE(U4, g.ctrl, g.qubit));
                }
            }
        
            // Execute any C1 gates that were scheduled on this rank
            sch_global.execute(exec_hw);
        
            pg.barrier(); // All ranks finish their local computations and C1 gates
            gate_counter.deallocate();
        
            std::cout << "[RANK " << rank << "] << Exiting run_gates_parallel. Returning " << local_results.size() << " local C2 results." << std::endl;
            return local_results;
        }

        std::vector<GateUpdateMetadata> allgather_metadata(const std::vector<LocalGateResult>& local_results) 
        {
            int rank = pg.rank().value();
            std::cout << "[RANK " << rank << "] >> Entering allgather_metadata. Processing " << local_results.size() << " local results." << std::endl;
        
            // Create metadata from the raw local results
            std::vector<GateUpdateMetadata> local_metadata;
            local_metadata.reserve(local_results.size());
            for(const auto& res : local_results) {
                if (res.is_valid) {
                    local_metadata.push_back({
                        true,
                        res.q0,
                        res.q1,
                        res.new_bond_dim,
                        res.original_rank
                    });
                }
            }
            std::cout << "[RANK " << rank << "] allgather_metadata: Created " << local_metadata.size() << " local metadata entries." << std::endl;
        
            // The rest of the function is a collective communication and remains the same.
            int local_size_bytes = local_metadata.size() * sizeof(GateUpdateMetadata);
            std::vector<int> all_sizes_bytes(pg.size().value());
        
            pg.allgather(&local_size_bytes, 1, all_sizes_bytes.data(), 1);
        
            std::vector<int> displacements_bytes(pg.size().value(), 0);
            int total_size_bytes = 0;
            for (size_t i = 0; i < all_sizes_bytes.size(); ++i) {
                if (i > 0) {
                    displacements_bytes[i] = displacements_bytes[i-1] + all_sizes_bytes[i-1];
                }
                total_size_bytes += all_sizes_bytes[i];
            }
            
            std::vector<GateUpdateMetadata> all_metadata;
            if (total_size_bytes > 0) {
                all_metadata.resize(total_size_bytes / sizeof(GateUpdateMetadata));
                MPI_Allgatherv(local_metadata.data(),
                               local_size_bytes,
                               MPI_BYTE,
                               all_metadata.data(),
                               all_sizes_bytes.data(),
                               displacements_bytes.data(),
                               MPI_BYTE,
                               pg.comm());
            } else {
                 std::cout << "[RANK " << rank << "] allgather_metadata: No metadata to gather." << std::endl;
            }
        
            std::cout << "[RANK " << rank << "] << Exiting allgather_metadata. Total metadata entries gathered: " << all_metadata.size() << std::endl;
            return all_metadata;
        }

        void apply_collective_updates(std::vector<LocalGateResult>& local_results)
        {
            int rank = pg.rank().value();
            std::cout << "[RANK " << rank << "] >> Entering apply_collective_updates with " << local_results.size() << " local results." << std::endl;
        
            auto all_metadata = allgather_metadata(local_results);
            std::cout << "[RANK " << rank << "] apply_collective_updates: Metadata gathered. Total updates to apply: " << all_metadata.size() << std::endl;
        
            tamm::Scheduler sch_global{ec};
            int local_result_idx = 0;
        
            // A list of temporary local tensors that will need to be deallocated
            std::vector<tamm::Tensor<Cplx>> temp_tensors_to_deallocate;
        
            for (const auto& meta : all_metadata)
            {
                if (!meta.is_valid) continue;
        
                IdxType q0 = meta.q0;
                IdxType q1 = meta.q1;
        
                // ALL ranks schedule deallocation of old global tensors
                sch_global.deallocate(mps_tensors[q0], mps_tensors[q1]);
        
                // ALL ranks update their metadata
                bond_dims[q0 + 1] = meta.new_bond_dim;
                tamm::IndexSpace is_new_bond{tamm::range(meta.new_bond_dim)};
                bond_tis[q0 + 1] = tamm::TiledIndexSpace(is_new_bond, block_size);
        
                // ALL ranks create new global tensor handles
                mps_tensors[q0] = tamm::Tensor<Cplx>{bond_tis[q0], phys_tis[q0], bond_tis[q0 + 1]};
                mps_tensors[q1] = tamm::Tensor<Cplx>{bond_tis[q1], phys_tis[q1], bond_tis[q1 + 1]};
                mps_tensors[q0].set_dense();
                mps_tensors[q1].set_dense();
        
                // ALL ranks schedule allocation of new global tensors
                sch_global.allocate(mps_tensors[q0], mps_tensors[q1]);
        
                // ONLY the source rank creates and populates temporary local tensors
                // and schedules the copy operation.
                if (rank == meta.original_rank) {
                    auto& result_data = local_results[local_result_idx++];
                    assert(result_data.q0 == meta.q0 && result_data.q1 == meta.q1);
        
                    // Create a temporary local context for the data holders
                    tamm::ProcGroup self_pg_put = tamm::ProcGroup::create_self();
                    tamm::ExecutionContext ec_local_put{self_pg_put, tamm::DistributionKind::dense, tamm::MemoryManagerKind::local};
                    tamm::Scheduler sch_local_put{ec_local_put};
        
                    // Create LOCAL tensors to hold the new data
                    tamm::Tensor<Cplx> T0_put({bond_tis[q0], phys_tis[q0], bond_tis[q0 + 1]});
                    tamm::Tensor<Cplx> T1_put({bond_tis[q1], phys_tis[q1], bond_tis[q1 + 1]});
                    T0_put.set_dense();
                    T1_put.set_dense();
        
                    // Allocate and fill the LOCAL tensors
                    sch_local_put.allocate(T0_put, T1_put).execute();
                    T0_put.put(*(T0_put.loop_nest().begin()), result_data.new_T0_data);
                    T1_put.put(*(T1_put.loop_nest().begin()), result_data.new_T1_data);
        
                    // Schedule the copy from the LOCAL tensor to the new GLOBAL tensor
                    sch_global(mps_tensors[q0]("l","p","b") = T0_put("l","p","b"));
                    sch_global(mps_tensors[q1]("b","p","r") = T1_put("b","p","r"));
        
                    // Schedule the deallocation of the temporary LOCAL tensors
                    sch_local_put.deallocate(T0_put, T1_put).execute();
                    self_pg_put.destroy_coll();
                }
            }
        
            std::cout << "[RANK " << rank << "] apply_collective_updates: All operations for this layer have been scheduled. Calling sch.execute()..." << std::endl;
            sch_global.execute(exec_hw);
            std::cout << "[RANK " << rank << "] << Exiting apply_collective_updates." << std::endl;
        }

        void C1_GATE(const std::array<Cplx, 4> &U, IdxType site, tamm::Scheduler& sch_global)
        {
            auto& ec_global = sch_global.ec();
        
            // Create a local context just for the small gate matrix
            tamm::ProcGroup self_pg = tamm::ProcGroup::create_self();
            tamm::ExecutionContext ec_local{self_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::local};
            tamm::Scheduler sch_local{ec_local};
        
            tamm::Tensor<Cplx> G({phys_tis[site], phys_tis[site]});
            G.set_dense();
            sch_local.allocate(G).execute(); // Allocate and fill the local tensor
        
            auto fill_g = [&](const tamm::IndexVector& bid, tamm::span<Cplx> buf){
                auto offsets = G.block_offsets(bid);
                int pout = offsets[0], pin = offsets[1];
                buf[0] = U[pout * 2 + pin];
            };
            tamm::update_tensor(G, fill_g);
        
            // Create a temporary GLOBAL tensor for the result
            tamm::Tensor<Cplx> Tnew_global({bond_tis[site], phys_tis[site], bond_tis[site + 1]});
            Tnew_global.set_dense();
        
            // Schedule all global operations together
            sch_global
                .allocate(Tnew_global)
                (Tnew_global("l","p_prime","r") = G("p_prime","p") * mps_tensors[site]("l","p","r"))
                (mps_tensors[site]("l","p","r") = Tnew_global("l","p","r"))
                .deallocate(Tnew_global);
        
            sch_local.deallocate(G).execute(); // Clean up local tensor
            self_pg.destroy_coll();
        }


        LocalGateResult C2_GATE_COMPUTE(const std::array<Cplx, 16> &U4, IdxType q0, IdxType q1)
        {
            // 1. Create a truly local execution context for this one-shot computation.
            tamm::ProcGroup self_pg = tamm::ProcGroup::create_self();
            tamm::ExecutionContext ec_local{self_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::local};
            tamm::Scheduler sch_local{ec_local};
        
            // 2. Create LOCAL tensors for inputs and intermediates.
            tamm::Tensor<Cplx> T0_local({bond_tis[q0], phys_tis[q0], bond_tis[q0 + 1]});
            tamm::Tensor<Cplx> T1_local({bond_tis[q1], phys_tis[q1], bond_tis[q1 + 1]});
            T0_local.set_dense(); T1_local.set_dense();
            sch_local.allocate(T0_local, T1_local).execute();
        
            // 3. GET data from the global tensors into local std::vectors, then PUT to local tensors.
            // This is the only communication with the global state needed for input.
            std::vector<Cplx> t0_buf(T0_local.size());
            mps_tensors[q0].get(*(mps_tensors[q0].loop_nest().begin()), t0_buf);
            T0_local.put(*(T0_local.loop_nest().begin()), t0_buf);
        
            std::vector<Cplx> t1_buf(T1_local.size());
            mps_tensors[q1].get(*(mps_tensors[q1].loop_nest().begin()), t1_buf);
            T1_local.put(*(T1_local.loop_nest().begin()), t1_buf);
            
            // 4. Perform local computations using the local scheduler.
            tamm::Tensor<Cplx> M_local({bond_tis[q0], phys_tis[q0], phys_tis[q1], bond_tis[q1 + 1]});
            tamm::Tensor<Cplx> G4_local({phys_tis[q0], phys_tis[q1], phys_tis[q0], phys_tis[q1]});
            tamm::Tensor<Cplx> M2_local({bond_tis[q0], phys_tis[q0], phys_tis[q1], bond_tis[q1 + 1]});
            M_local.set_dense(); G4_local.set_dense(); M2_local.set_dense();
            sch_local.allocate(M_local, G4_local, M2_local).execute();
        
            sch_local(M_local("l","p0","p1","r") = T0_local("l","p0","b") * T1_local("b","p1","r")).execute();
            
            auto fill_g4 = [&](const tamm::IndexVector& bid, tamm::span<Cplx> buf){
                auto offsets = G4_local.block_offsets(bid);
                int p0p = offsets[0], p1p = offsets[1], p0 = offsets[2], p1 = offsets[3];
                buf[0] = U4[(p0p * 2 + p1p) * 4 + (p0 * 2 + p1)];
            };
            tamm::update_tensor(G4_local, fill_g4);
            
            sch_local(M2_local("l","p0p","p1p","r") = G4_local("p0p","p1p","p0","p1") * M_local("l","p0","p1","r")).execute();
        
            // 5. Perform SVD on the local M2_local tensor.
            // This function now returns data into std::vectors, not pre-allocated global tensors.
            std::vector<Cplx> Ti_new_data, Tj_new_data;
            IdxType new_bond_dim = local_svd_and_reconstruct_data(M2_local, Ti_new_data, Tj_new_data, q0, q1);
            
            // 6. Package the results into the POD struct.
            LocalGateResult result;
            result.is_valid = true;
            result.q0 = q0;
            result.q1 = q1;
            result.new_bond_dim = new_bond_dim;
            result.new_T0_data = std::move(Ti_new_data);
            result.new_T1_data = std::move(Tj_new_data);
            result.original_rank = pg.rank().value();
        
            // 7. Clean up all temporary local resources.
            sch_local.deallocate(T0_local, T1_local, M_local, G4_local, M2_local).execute();
            self_pg.destroy_coll();
        
            return result;
        }

        void gpu_svd_jacobi(
            const Cplx* A_h, int m, int n,
            std::vector<double>& S,
            std::vector<Cplx>& U_row,
            std::vector<Cplx>& VT_row)
        {
            int rank = pg.rank().value();
            std::cout << "[RANK " << rank << "] ---> gpu_svd_jacobi: Entered. Matrix dimensions (m, n): (" << m << ", " << n << ")." << std::endl;

            cusolverDnXgesvdjSetTolerance(cu_ctx_.jp, 1e-14);
            cusolverDnXgesvdjSetMaxSweeps(cu_ctx_.jp, 100);

            int lda = m;
            int ldu = m;
            int ldv = n;
            int econ = 1; // Economy SVD
            int k = std::min(m, n);

            cuDoubleComplex* d_A = nullptr;
            double* d_S = nullptr;
            cuDoubleComplex* d_U = nullptr;
            cuDoubleComplex* d_V = nullptr;
            int* d_info = nullptr;

            std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Allocating GPU memory for A, S, U, V, info." << std::endl;
            cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex) * (size_t)lda * (size_t)n);
            cudaMalloc((void**)&d_S, sizeof(double) * (size_t)k);
            cudaMalloc((void**)&d_U, sizeof(cuDoubleComplex) * (size_t)ldu * (size_t)k);
            cudaMalloc((void**)&d_V, sizeof(cuDoubleComplex) * (size_t)ldv * (size_t)k);
            cudaMalloc((void**)&d_info, sizeof(int));

            std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Copying host matrix A to device." << std::endl;
            cudaMemcpyAsync(d_A, reinterpret_cast<const cuDoubleComplex*>(A_h),
                            sizeof(cuDoubleComplex) * (size_t)lda * (size_t)n,
                            cudaMemcpyHostToDevice, cu_ctx_.stream);

            int lwork_req = 0;
            std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Querying buffer size for Zgesvdj." << std::endl;
            cusolverDnZgesvdj_bufferSize(cu_ctx_.solver, CUSOLVER_EIG_MODE_VECTOR, econ,
                                         m, n, d_A, lda, d_S, d_U, ldu, d_V, ldv,
                                         &lwork_req, cu_ctx_.jp);
            std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Required buffer size (lwork_req): " << lwork_req << "." << std::endl;

            if (lwork_req > cu_ctx_.lwork_jac) {
                std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Reallocating GPU workspace from " << cu_ctx_.lwork_jac << " to " << lwork_req << "." << std::endl;
                if (cu_ctx_.d_work_jac) cudaFree(cu_ctx_.d_work_jac);
                cu_ctx_.lwork_jac = lwork_req;
                cudaMalloc((void**)&cu_ctx_.d_work_jac, sizeof(cuDoubleComplex) * (size_t)cu_ctx_.lwork_jac);
            }

            std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Calling cusolverDnZgesvdj to perform SVD on GPU..." << std::endl;
            cusolverDnZgesvdj(cu_ctx_.solver, CUSOLVER_EIG_MODE_VECTOR, econ,
                              m, n, d_A, lda, d_S, d_U, ldu, d_V, ldv,
                              reinterpret_cast<cuDoubleComplex*>(cu_ctx_.d_work_jac),
                              cu_ctx_.lwork_jac, d_info, cu_ctx_.jp);

            std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Synchronizing CUDA stream..." << std::endl;
            cudaStreamSynchronize(cu_ctx_.stream);
            std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Stream synchronized. GPU computation finished." << std::endl;

            S.resize((size_t)k);
            std::vector<Cplx> U_col((size_t)ldu * (size_t)k);
            std::vector<Cplx> V_col((size_t)ldv * (size_t)k);

            std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Copying results S, U, V from device to host." << std::endl;
            cudaMemcpy(S.data(), d_S, sizeof(double) * (size_t)k, cudaMemcpyDeviceToHost);
            cudaMemcpy(U_col.data(), d_U, sizeof(Cplx) * (size_t)ldu * (size_t)k, cudaMemcpyDeviceToHost);
            cudaMemcpy(V_col.data(), d_V, sizeof(Cplx) * (size_t)ldv * (size_t)k, cudaMemcpyDeviceToHost);
            std::cout << "[RANK " << rank << "] gpu_svd_jacobi: D2H copy complete." << std::endl;

            std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Transposing U and V to row-major format." << std::endl;
            U_row.resize((size_t)m * (size_t)k);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < k; ++j)
                    U_row[(size_t)i * (size_t)k + (size_t)j] = U_col[(size_t)i + (size_t)j * (size_t)ldu];

            VT_row.resize((size_t)k * (size_t)n);
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < n; ++j)
                    VT_row[(size_t)i * (size_t)n + (size_t)j] = std::conj(V_col[(size_t)j + (size_t)i * (size_t)ldv]);

            std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Freeing GPU memory." << std::endl;
            cudaFree(d_info);
            cudaFree(d_V);
            cudaFree(d_U);
            cudaFree(d_S);
            cudaFree(d_A);

            std::cout << "[RANK " << rank << "] <--- gpu_svd_jacobi: Exiting." << std::endl;
        }

        IdxType local_svd_and_reconstruct_data(
            tamm::Tensor<Cplx>& M2_local,
            std::vector<Cplx>& Ti_new_data,
            std::vector<Cplx>& Tj_new_data,
            IdxType q0, IdxType q1)
        {
            int rank = pg.rank().value();
            std::cout << "[RANK " << rank << "] ---> local_svd_and_reconstruct_data: Entered for qubits (" << q0 << ", " << q1 << ")." << std::endl;
        
            const IdxType phys_dim = 2;
            IdxType Dl = M2_local.tiled_index_spaces()[0].index_space().num_indices();
            IdxType Dr = M2_local.tiled_index_spaces()[3].index_space().num_indices();
        
            int m = Dl * phys_dim;
            int n = phys_dim * Dr;
        
            // Reshape the row-major TAMM tensor data into a column-major matrix for cuSOLVER.
            std::vector<Cplx> M2_col_major(m * n);
            std::vector<Cplx> M2_hostbuf(M2_local.size());
            M2_local.get(*(M2_local.loop_nest().begin()), M2_hostbuf);
        
            size_t c = 0;
            for (size_t l = 0; l < Dl; ++l)
            for (size_t p0 = 0; p0 < phys_dim; ++p0)
            for (size_t p1 = 0; p1 < phys_dim; ++p1)
            for (size_t r = 0; r < Dr; ++r, ++c)
            {
                size_t row = l * phys_dim + p0;
                size_t col = p1 * Dr + r;
                M2_col_major[row + col * m] = M2_hostbuf[c];
            }
            std::cout << "[RANK " << rank << "] local_svd_and_reconstruct_data: Reshape complete." << std::endl;
        
            // Perform the SVD on the GPU.
            std::vector<double> S;
            std::vector<Cplx> U_row, VT_row;
            gpu_svd_jacobi(M2_col_major.data(), m, n, S, U_row, VT_row);
        
            // Truncate based on singular value cutoff and max bond dimension.
            std::vector<IdxType> keep;
            keep.reserve(S.size());
            for (size_t i = 0; i < S.size(); ++i) {
                if (S[i] >= sv_cutoff) {
                    keep.push_back(i);
                }
            }
            IdxType chi = std::min<IdxType>(max_bond_dim, IdxType(keep.size()));
            if (chi == 0) {
                chi = 1; // Prevent bond dimension from becoming zero.
            }
            std::cout << "[RANK " << rank << "] local_svd_and_reconstruct_data: Truncation complete. New bond dimension (chi): " << chi << "." << std::endl;
        
            // Populate the output vectors with the data for the new tensors.
            Ti_new_data.resize(Dl * phys_dim * chi);
            c = 0;
            for (size_t l = 0; l < Dl; ++l)
            for (size_t p0 = 0; p0 < phys_dim; ++p0)
            for (size_t b = 0; b < chi; ++b, ++c)
            {
                Ti_new_data[c] = U_row[(l * phys_dim + p0) * S.size() + keep[b]];
            }
        
            Tj_new_data.resize(chi * phys_dim * Dr);
            c = 0;
            for (size_t b = 0; b < chi; ++b)
            for (size_t p1 = 0; p1 < phys_dim; ++p1)
            for (size_t r = 0; r < Dr; ++r, ++c)
            {
                Tj_new_data[c] = Cplx(S[keep[b]], 0.0) * VT_row[keep[b] * n + (p1 * Dr + r)];
            }
        
            std::cout << "[RANK " << rank << "] <--- local_svd_and_reconstruct_data: Exiting." << std::endl;
            return chi;
        }
        void right_canonicalize(std::vector<tamm::Tensor<Cplx>> &MPS)
        {
            // canonicalize MPS from right end toward left
            for (IdxType i = n_qubits - 1; i > 0; --i)
            {
                // extract dimensions and assemble Mmat for SVD
                IdxType Dl_old = bond_dims[i];
                IdxType Dr     = bond_dims[i + 1];
                IdxType d      = phys_dims[i];
                Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Mmat(Dl_old, d * Dr);
                {
                    auto &T = MPS[i];
                    for (const auto &blockid : T.loop_nest())
                    {
                        const size_t bs = T.block_size(blockid);
                        std::vector<Cplx> hostbuf(bs);
                        T.get(blockid, hostbuf);
                        auto dims = T.block_dims(blockid);
                        auto offs = T.block_offsets(blockid);
                        size_t idx = 0;
                        for (size_t ll = offs[0]; ll < offs[0] + dims[0]; ++ll)
                        {
                            for (size_t pp = offs[1]; pp < offs[1] + dims[1]; ++pp)
                            {
                                for (size_t rr = offs[2]; rr < offs[2] + dims[2]; ++rr, ++idx)
                                {
                                    Mmat(ll, pp * Dr + rr) = hostbuf[idx];
                                }
                            }
                        }
                    }
                }
        
                // perform SVD and truncate to bond dimension
                Eigen::BDCSVD<decltype(Mmat)> svd(Mmat, Eigen::ComputeThinU | Eigen::ComputeThinV);
                auto svals = svd.singularValues();
                IdxType chi = std::min<IdxType>(IdxType(svals.size()), max_bond_dim);
                auto Umat  = svd.matrixU().leftCols(chi);
                auto Vh    = svd.matrixV().leftCols(chi).adjoint();
                bond_dims[i] = chi;
                {
                    tamm::IndexSpace is_new{ tamm::range(chi) };
                    bond_tis[i] = tamm::TiledIndexSpace(is_new, block_size);
                }
        
                // build updated right tensor via Vh
                tamm::Tensor<Cplx> Tnew({ bond_tis[i], phys_tis[i], bond_tis[i + 1] });
                Tnew.set_dense();
                Tnew.allocate(&ec);
                {
                    auto &T = Tnew;
                    for (const auto &blockid : T.loop_nest())
                    {
                        const size_t bs = T.block_size(blockid);
                        std::vector<Cplx> hostbuf(bs);
                        auto dims = T.block_dims(blockid);
                        auto offs = T.block_offsets(blockid);
                        size_t idx = 0;
                        for (size_t ll = offs[0]; ll < offs[0] + dims[0]; ++ll)
                        {
                            for (size_t pp = offs[1]; pp < offs[1] + dims[1]; ++pp)
                            {
                                for (size_t rr = offs[2]; rr < offs[2] + dims[2]; ++rr, ++idx)
                                {
                                    hostbuf[idx] = Vh(ll, pp * Dr + rr);
                                }
                            }
                        }
                        T.put(blockid, hostbuf);
                    }
                }
        
                // update left neighbor via Umat * S
                IdxType Dl_prev = bond_dims[i - 1];
                IdxType d_prev  = phys_dims[i - 1];
                Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Mprev(Dl_prev * d_prev, Dl_old);
                {
                    auto &Told = MPS[i - 1];
                    for (const auto &blockid : Told.loop_nest())
                    {
                        const size_t bs = Told.block_size(blockid);
                        std::vector<Cplx> hostbuf(bs);
                        Told.get(blockid, hostbuf);
                        auto dims = Told.block_dims(blockid);
                        auto offs = Told.block_offsets(blockid);
                        size_t idx = 0;
                        for (size_t ll = offs[0]; ll < offs[0] + dims[0]; ++ll)
                        {
                            for (size_t pp = offs[1]; pp < offs[1] + dims[1]; ++pp)
                            {
                                for (size_t rr = offs[2]; rr < offs[2] + dims[2]; ++rr, ++idx)
                                {
                                    Mprev(ll * d_prev + pp, rr) = hostbuf[idx];
                                }
                            }
                        }
                    }
                }
                auto US     = Umat * svals.head(chi).asDiagonal();
                auto Mprev2 = Mprev * US;
        
                // build updated left tensor via Mprev2
                tamm::Tensor<Cplx> Tprev({ bond_tis[i - 1], phys_tis[i - 1], bond_tis[i] });
                Tprev.set_dense();
                Tprev.allocate(&ec);
                {
                    auto &T = Tprev;
                    for (const auto &blockid : T.loop_nest())
                    {
                        const size_t bs = T.block_size(blockid);
                        std::vector<Cplx> hostbuf(bs);
                        auto dims = T.block_dims(blockid);
                        auto offs = T.block_offsets(blockid);
                        size_t idx = 0;
                        for (size_t ll = offs[0]; ll < offs[0] + dims[0]; ++ll)
                        {
                            for (size_t pp = offs[1]; pp < offs[1] + dims[1]; ++pp)
                            {
                                for (size_t rr = offs[2]; rr < offs[2] + dims[2]; ++rr, ++idx)
                                {
                                    hostbuf[idx] = Mprev2(ll * d_prev + pp, rr);
                                }
                            }
                        }
                        T.put(blockid, hostbuf);
                    }
                }
        
                // replace tensors in MPS
                MPS[i].deallocate();
                MPS[i - 1].deallocate();
                MPS[i]     = std::move(Tnew);
                MPS[i - 1] = std::move(Tprev);
            }
        }

        void left_canonicalize(std::vector<tamm::Tensor<Cplx>>& MPS)
        {
            for (IdxType i = 0; i < n_qubits - 1; ++i)
            {
                // assemble matrix from MPS[i]
                IdxType Dl     = bond_dims[i];
                IdxType Dr_old = bond_dims[i + 1];
                IdxType d      = phys_dims[i];
                Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Mmat(Dl * d, Dr_old);
                {
                    auto& T = MPS[i];
                    for (const auto& blockid : T.loop_nest())
                    {
                        size_t bs = T.block_size(blockid);
                        std::vector<Cplx> hostbuf(bs);
                        T.get(blockid, hostbuf);
        
                        auto dims = T.block_dims(blockid);
                        auto offs = T.block_offsets(blockid);
                        size_t idx = 0;
                        for (size_t ll = offs[0]; ll < offs[0] + dims[0]; ++ll)
                        {
                            for (size_t pp = offs[1]; pp < offs[1] + dims[1]; ++pp)
                            {
                                for (size_t rr = offs[2]; rr < offs[2] + dims[2]; ++rr, ++idx)
                                {
                                    Mmat(ll * d + pp, rr) = hostbuf[idx];
                                }
                            }
                        }
                    }
                }
        
                // compute truncated SVD of Mmat
                Eigen::BDCSVD<decltype(Mmat)> svd(Mmat, Eigen::ComputeThinU | Eigen::ComputeThinV);
                auto svals = svd.singularValues();
                IdxType chi = std::min<IdxType>(IdxType(svals.size()), max_bond_dim);
                auto Umat  = svd.matrixU().leftCols(chi);
                auto Sdiag  = svals.head(chi).asDiagonal();
                auto Vh     = svd.matrixV().leftCols(chi).adjoint();
        
                // update bond dimension and index space
                bond_dims[i + 1] = chi;
                {
                    tamm::IndexSpace is_new{ tamm::range(chi) };
                    bond_tis[i + 1] = tamm::TiledIndexSpace(is_new, block_size);
                }
        
                // build new left tensor from Umat
                tamm::Tensor<Cplx> Tleft({ bond_tis[i], phys_tis[i], bond_tis[i + 1] });
                Tleft.set_dense();
                Tleft.allocate(&ec);
                for (const auto& blockid : Tleft.loop_nest())
                {
                    size_t bs = Tleft.block_size(blockid);
                    std::vector<Cplx> hostbuf(bs);
        
                    auto dims = Tleft.block_dims(blockid);
                    auto offs = Tleft.block_offsets(blockid);
                    size_t idx = 0;
                    for (size_t ll = offs[0]; ll < offs[0] + dims[0]; ++ll)
                    {
                        for (size_t pp = offs[1]; pp < offs[1] + dims[1]; ++pp)
                        {
                            for (size_t bb = offs[2]; bb < offs[2] + dims[2]; ++bb, ++idx)
                            {
                                hostbuf[idx] = Umat(ll * d + pp, bb);
                            }
                        }
                    }
        
                    Tleft.put(blockid, hostbuf);
                }
        
                // assemble matrix from MPS[i+1]
                IdxType d_next   = phys_dims[i + 1];
                IdxType Dr_right = bond_dims[i + 2];
                Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Tnext_mat(chi, d_next * Dr_right);
                {
                    auto& Tnext_old = MPS[i + 1];
                    for (const auto& blockid : Tnext_old.loop_nest())
                    {
                        size_t bs = Tnext_old.block_size(blockid);
                        std::vector<Cplx> hostbuf(bs);
                        Tnext_old.get(blockid, hostbuf);
        
                        auto dims = Tnext_old.block_dims(blockid);
                        auto offs = Tnext_old.block_offsets(blockid);
                        size_t idx = 0;
                        for (size_t bb = offs[0]; bb < offs[0] + dims[0]; ++bb)
                        {
                            for (size_t pp = offs[1]; pp < offs[1] + dims[1]; ++pp)
                            {
                                for (size_t rr = offs[2]; rr < offs[2] + dims[2]; ++rr, ++idx)
                                {
                                    Tnext_mat(bb, pp * Dr_right + rr) = hostbuf[idx];
                                }
                            }
                        }
                    }
                }
        
                // apply spectrum to right neighbor tensor
                auto M2_eig    = Sdiag * Vh;
                auto Tnext_new = M2_eig * Tnext_mat;
                tamm::Tensor<Cplx> Tnext({ bond_tis[i + 1], phys_tis[i + 1], bond_tis[i + 2] });
                Tnext.set_dense();
                Tnext.allocate(&ec);
                for (const auto& blockid : Tnext.loop_nest())
                {
                    size_t bs = Tnext.block_size(blockid);
                    std::vector<Cplx> hostbuf(bs);
        
                    auto dims = Tnext.block_dims(blockid);
                    auto offs = Tnext.block_offsets(blockid);
                    size_t idx = 0;
                    for (size_t bb = offs[0]; bb < offs[0] + dims[0]; ++bb)
                    {
                        for (size_t pp = offs[1]; pp < offs[1] + dims[1]; ++pp)
                        {
                            for (size_t rr = offs[2]; rr < offs[2] + dims[2]; ++rr, ++idx)
                            {
                                hostbuf[idx] = Tnext_new(bb, pp * Dr_right + rr);
                            }
                        }
                    }
        
                    Tnext.put(blockid, hostbuf);
                }
        
                // replace tensors in MPS
                MPS[i].deallocate();
                MPS[i + 1].deallocate();
                MPS[i]     = std::move(Tleft);
                MPS[i + 1] = std::move(Tnext);
            }
        }

        /*MA Gate
        *1) Free and reallocate a buffer to store repetition number of measurement results.

        *2) Canonicalize the MPS from both left and right to stabilize subsequent contractions.
        
        *3) Initialize a random number generator for sampling measurement outcomes.
        
        *4) For each repetition:
        *a. Initialize the environment vector with amplitude 1.
        *b. Loop over all qubit sites from left to right:
        *i. Extract the MPS tensor at the current site.
        *ii. Contract the current environment with the tensor to produce two branch environments for qubit outcomes 0 and 1.
        *iii. Compute squared norms of both branches to determine measurement probabilities.
        *iv. Sample a measurement outcome based on these probabilities.
        *v. Record the sampled bit in the packed result.
        *vi. Renormalize the chosen branch and set it as the new environment.
        
        *5) Store the final bitstring for each repetition in the results buffer. */
        virtual void MA_GATE(const IdxType repetition)
        {
            // allocate result buffer
            SAFE_FREE_HOST(results);
            SAFE_ALOC_HOST(results, sizeof(IdxType) * repetition);
            std::memset(results, 0, sizeof(IdxType) * repetition);
        
            // bring MPS into canonical form
            left_canonicalize(mps_tensors);
            right_canonicalize(mps_tensors);
        
            // perform measurement assignment repetitions
            std::mt19937_64 rng{std::random_device{}()};
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (IdxType rep = 0; rep < repetition; ++rep)
            {
                IdxType packed = 0;
                std::vector<Cplx> env(1, Cplx{1.0, 0.0});
        
                // sweep through all qubits
                for (IdxType site = 0; site < n_qubits; ++site)
                {
                    auto& T = mps_tensors[site];
                    IdxType Dl = bond_dims[site];
                    IdxType Dr = bond_dims[site + 1];
        
                    std::vector<Cplx> env0(Dr, Cplx{0.0, 0.0});
                    std::vector<Cplx> env1(Dr, Cplx{0.0, 0.0});
        
                    // unpack tensor into branch environments
                    for (const auto& blockid : T.loop_nest())
                    {
                        size_t bs = T.block_size(blockid);
                        std::vector<Cplx> hostbuf(bs);
                        T.get(blockid, hostbuf);
        
                        auto dims = T.block_dims(blockid);
                        auto offs = T.block_offsets(blockid);
                        size_t idx = 0;
                        for (size_t jj = 0; jj < dims[0]; ++jj)
                        {
                            for (size_t kk = 0; kk < dims[1]; ++kk)
                            {
                                for (size_t ll = 0; ll < dims[2]; ++ll, ++idx)
                                {
                                    IdxType l = offs[0] + jj;
                                    IdxType s = offs[1] + kk;
                                    IdxType r = offs[2] + ll;
                                    Cplx prod = env[l] * hostbuf[idx];
                                    if (s == 0)
                                    {
                                        env0[r] += prod;
                                    }
                                    else
                                    {
                                        env1[r] += prod;
                                    }
                                }
                            }
                        }
                    }
        
                    // compute probabilities and sample outcome
                    long double w0 = 0.0L;
                    long double w1 = 0.0L;
                    for (IdxType r = 0; r < Dr; ++r)
                    {
                        w0 += std::norm(env0[r]);
                        w1 += std::norm(env1[r]);
                    }
                    long double sumw = w0 + w1;
                    long double p0 = sumw > 0.0L ? (w0 / sumw) : 0.0L;
                    bool outcome1 = (dist(rng) >= static_cast<double>(p0));
                    if (outcome1)
                    {
                        packed |= (IdxType(1) << site);
                    }
        
                    auto& chosen = outcome1 ? env1 : env0;
                    long double norm_branch = outcome1 ? w1 : w0;
                    long double invnorm = norm_branch > 0.0L
                        ? (1.0L / std::sqrt(norm_branch))
                        : 0.0L;
                    env.assign(Dr, Cplx{0.0, 0.0});
                    for (IdxType r = 0; r < Dr; ++r)
                    {
                        env[r] = chosen[r] * static_cast<Cplx>(invnorm);
                    }
                }
        
                results[rep] = packed;
            }
        }

        void local_left_step(IdxType i)
        {
            // extract dimensions for sites i and i+1
            const IdxType Dl       = bond_dims[i];
            const IdxType Dr       = bond_dims[i+1];
            const IdxType d        = phys_dims[i];
            const IdxType d_next   = phys_dims[i+1];
            const IdxType Dr_next  = bond_dims[i+2];
        
            // reshape MPS[i] into matrix Mmat of size (Dl*d) × Dr
            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Mmat(Dl * d, Dr);
            auto& Ti = mps_tensors[i];
            Ti.loop_nest().iterate([&](auto const& idxs)
            {
                IdxType l = idxs[0], p = idxs[1], r = idxs[2];
                Cplx v;
                Ti.get(idxs, gsl::span<Cplx>(&v,1));
                Mmat(l * d + p, r) = v;
            });
        
            // compute SVD of Mmat
            Eigen::BDCSVD<decltype(Mmat)> svd(
                Mmat, Eigen::ComputeThinU | Eigen::ComputeThinV);
            auto Umat  = svd.matrixU();
            auto Sdiag = svd.singularValues().asDiagonal();
            auto Vh    = svd.matrixV().adjoint();
        
            // update bond dimension at i+1
            const IdxType chi = Umat.cols();
            bond_dims[i+1] = chi;
            {
                tamm::IndexSpace is_new{ tamm::range(chi) };
                bond_tis[i+1] = tamm::TiledIndexSpace(is_new, 1);
            }
        
            // write back new left tensor Ti_new
            tamm::Tensor<Cplx> Ti_new({ bond_tis[i], phys_tis[i], bond_tis[i+1] });
            Ti_new.set_dense();
            Ti_new.allocate(&ec);
            Ti_new.loop_nest().iterate([&](auto const& idxs)
            {
                IdxType l = idxs[0], p = idxs[1], b = idxs[2];
                Cplx val = Umat(l * d + p, b);
                Ti_new.put(idxs, gsl::span<Cplx>(&val,1));
            });
        
            // absorb spectrum S·Vh into MPS[i+1]
            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> A = Sdiag * Vh;
            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Bmat(
                Dr, d_next * Dr_next);
            auto& Tj = mps_tensors[i+1];
            Tj.loop_nest().iterate([&](auto const& idxs)
            {
                IdxType b = idxs[0], p = idxs[1], r = idxs[2];
                Cplx v;
                Tj.get(idxs, gsl::span<Cplx>(&v,1));
                Bmat(b, p * Dr_next + r) = v;
            });
            auto Bnew = A * Bmat;
        
            // write back new right tensor Tj_new
            tamm::Tensor<Cplx> Tj_new({ bond_tis[i+1], phys_tis[i+1], bond_tis[i+2] });
            Tj_new.set_dense();
            Tj_new.allocate(&ec);
            Tj_new.loop_nest().iterate([&](auto const& idxs)
            {
                IdxType b = idxs[0], p = idxs[1], r = idxs[2];
                Cplx val = Bnew(b, p * Dr_next + r);
                Tj_new.put(idxs, gsl::span<Cplx>(&val,1));
            });
        
            // replace old tensors
            Ti.deallocate();
            Tj.deallocate();
            mps_tensors[i]   = std::move(Ti_new);
            mps_tensors[i+1] = std::move(Tj_new);
        }
        
        /* Sets the MPS in a mixed gauge centered around a specified position */
        void position(IdxType site)
        {
            assert(site < n_qubits);
        
            // bring orthogonality center to qubit 0
            right_canonicalize(mps_tensors);
        
            // sweep center from 0 to target site
            for (IdxType i = 0; i < site; ++i)
            {
                local_left_step(i);
            }
        }

        virtual void M_GATE(const IdxType qubit)
        {
            throw std::runtime_error("Not implemented");
        }

        virtual void EXPECT_GATE(ObservableList *o)
        {
            throw std::runtime_error("Not implemented");
        }

        virtual void RESET_GATE(const IdxType qubit)
        {
            throw std::runtime_error("Not implemented");
        }

        static tamm::ProcGroup init_pg()
        {
            int argc = 0; char** argv = nullptr;
            //MPI_Init(&argc,&argv);
            //GA_Initialize();
            //tamm::initialize(argc, argv);
            //tamm::ProcGroup::self_ga_pgroup(true);
            return tamm::ProcGroup::create_world_coll();
        }
    };
} // namespace NWQS
