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
#include <map>
#include <cstring>

#include <fstream>
#include <iomanip>

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
            ec(pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga), // Init Global EC
            sch_global(ec),                                                     // Init Global Scheduler
            ec_local(tamm::ProcGroup::create_self(), tamm::DistributionKind::dense, tamm::MemoryManagerKind::local), // Init Local EC
            sch_local(ec_local)                                                 // Init Local Scheduler
        {
            // ... (rest of the constructor code is correct) ...
            i_proc = pg.rank().value();
            if (backend == "TN_TAMM_CPU") exec_hw = tamm::ExecutionHW::CPU;
            else if(backend == "TN_TAMM_GPU") exec_hw = tamm::ExecutionHW::GPU;
            int world_size = pg.size().value();
            IdxType qubits_per_rank = n_qubits / world_size;
            IdxType remainder = n_qubits % world_size;
            start_qubit = pg.rank().value() * qubits_per_rank + std::min((IdxType)pg.rank().value(), remainder);
            end_qubit = start_qubit + qubits_per_rank + (pg.rank().value() < remainder ? 1 : 0);
            bond_tis.resize(n_qubits + 1);
            bond_dims.resize(n_qubits + 1);
            for (IdxType i = 0; i <= n_qubits; ++i) {
                bond_dims[i] = 1;
                tamm::IndexSpace is{ tamm::range(1) };
                bond_tis[i] = tamm::TiledIndexSpace(is, block_size);
            }
            phys_tis.resize(n_qubits);
            phys_dims.resize(n_qubits);
            for (IdxType i = 0; i < n_qubits; ++i) {
                phys_dims[i] = 2;
                tamm::IndexSpace is{ tamm::range(2) };
                phys_tis[i] = tamm::TiledIndexSpace(is, 1);
            }
            mps_tensors.resize(n_qubits);
            for (IdxType i = start_qubit; i < end_qubit; ++i) {
                mps_tensors[i] = tamm::Tensor<Cplx>({ bond_tis[i], phys_tis[i], bond_tis[i + 1] });
                mps_tensors[i].set_dense();
                sch_local.allocate(mps_tensors[i]);
            }
            sch_local.execute(exec_hw);
            if (start_qubit == 0) {
                auto& T = mps_tensors[0];
                T.loop_nest().iterate([&](auto const& idxs){
                    Cplx v = (idxs[0] == 0 && idxs[1] == 0 && idxs[2] == 0) ? Cplx(1.0,0.0) : Cplx(0.0,0.0);
                    T.put(idxs, gsl::span<Cplx>(&v,1));
                });
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
            IdxType original_gate_count = circuit->num_gates();
            std::vector<SVGate> gates = fuse_circuit_sv(circuit);
            IdxType fused_gate_count = gates.size();
            assert(circuit->num_qubits() == n_qubits);
        
            pg.barrier();
            auto start_time = std::chrono::high_resolution_clock::now();
            simulation_kernel(gates);
            auto end_time = std::chrono::high_resolution_clock::now();
            pg.barrier();
        
            std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        
            if (pg.rank().value() == 0) {
                std::cout << "simulation_kernel execution time: "
                          << elapsed_seconds.count() << " seconds." << std::endl;
        
                std::ofstream csv_file("timings.csv", std::ios::app);
                if (csv_file.is_open()) {
                    csv_file << std::fixed << std::setprecision(6)
                             << elapsed_seconds.count() << "\n";
                    csv_file.close();
                } else {
                    std::cerr << "Error: Unable to open timings.csv for writing." << std::endl;
                }
            }
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
            SVGate t(g); 
        
            t.ctrl = left;
            t.qubit = right;
            
            return t;
        }
    
        // Helper function to check for conflicts in a layer
        bool has_conflict(int qubit, const std::vector<SVGate>& layer) {
            for (const auto& gate_in_layer : layer) {
                if (gate_in_layer.op_name == OP::C1) {
                    if (gate_in_layer.qubit == qubit) return true;
                } else if (gate_in_layer.op_name == OP::C2) {
                    if (gate_in_layer.qubit == qubit || gate_in_layer.ctrl == qubit) return true;
                }
            }
            return false;
        }
        
        // Overload for two-qubit gates
        bool has_conflict(int qubit1, int qubit2, const std::vector<SVGate>& layer) {
            return has_conflict(qubit1, layer) || has_conflict(qubit2, layer);
        }
        
        
        // In the TN_TAMM class
        void place_c1(const SVGate& s,
                      std::vector<std::vector<SVGate>>& layers,
                      std::map<int,int>& last_layer_map) // CHANGED: from std::unordered_map
        {
            int q = s.qubit;
            // Determine the earliest possible layer based on data dependency
            int L = last_layer_map[q] + 1;
        
            // Find the first layer (starting from L) that has no qubit conflict
            while (true) {
                // Ensure the layers vector is large enough
                if (L > layers.size()) {
                    layers.resize(L);
                }
                
                // If there's no conflict in the target layer (index L-1), place the gate
                if (!has_conflict(q, layers[L - 1])) {
                    layers[L - 1].push_back(s);
                    last_layer_map[q] = L; // Update the last-used layer for this qubit
                    return; // Done
                }
                
                // Conflict found, try the next layer
                L++;
            }
        } 

        // In the TN_TAMM class
        void place_c2(const SVGate& t, int a, int b,
                      std::vector<std::vector<SVGate>>& layers,
                      std::map<int,int>& last_layer_map) // CHANGED: from std::unordered_map
        {
            // Determine the earliest possible layer based on data dependencies
            int L = 1 + std::max(last_layer_map[a], last_layer_map[b]);
        
            SVGate x = t; // Create a mutable copy to set qubits correctly
            x.ctrl = a;
            x.qubit = b;
        
            // Find the first layer (starting from L) that has no qubit conflict
            while (true) {
                if (L > layers.size()) {
                    layers.resize(L);
                }
        
                if (!has_conflict(a, b, layers[L - 1])) {
                    layers[L - 1].push_back(x);
                    last_layer_map[a] = L; // Update last-used layer for both qubits
                    last_layer_map[b] = L;
                    return;
                }
                
                L++;
            }
        }

        // In the TN_TAMM class
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
            
            // Interleave while both lists have elements
            while (i < singles.size() && j < twos.size())
            {
                // Prioritize the longer list to start, for better balance
                if (singles.size() >= twos.size()) {
                     out.push_back(singles[i++]);
                     if (j < twos.size()) out.push_back(twos[j++]);
                } else {
                     out.push_back(twos[j++]);
                     if (i < singles.size()) out.push_back(singles[i++]);
                }
            }
        
            // Append the remainder of whichever list is not yet empty.
            // Only one of these two loops will execute.
            while (i < singles.size()) {
                out.push_back(singles[i++]);
            }
            while (j < twos.size()) {
                out.push_back(twos[j++]);
            }
        }        

        // In the TN_TAMM class
        std::string gate_to_string(const SVGate& g) {
            std::stringstream ss;
            if (g.op_name == OP::C1) {
                ss << "C1(" << g.qubit << ")";
            } else if (g.op_name == OP::C2) {
                ss << "C2(" << g.ctrl << "," << g.qubit << ")";
            } else {
                ss << "UNKNOWN";
            }
            return ss.str();
        }



        protected:
            IdxType n_qubits;
            IdxType* results = NULL;
            IdxType max_bond_dim;
            int block_size;
            double sv_cutoff;
            tamm::ExecutionHW exec_hw;
        
            // --- ADD/MODIFY THESE MEMBERS ---
            tamm::ProcGroup pg;             // Global process group
            tamm::ExecutionContext ec;      // Global execution context
            tamm::Scheduler sch_global;     // Scheduler for GLOBAL operations (communication)
        
            tamm::ExecutionContext ec_local;  // LOCAL execution context (per-rank)
            tamm::Scheduler sch_local;      // Scheduler for LOCAL operations (computation)
        
            // Qubit partitioning info for this rank
            IdxType start_qubit;
            IdxType end_qubit;
            // --- END OF ADDITIONS/MODIFICATIONS ---
        
            std::vector<IdxType> bond_dims;
            std::vector<IdxType> phys_dims;
            std::vector<tamm::TiledIndexSpace> bond_tis;
            std::vector<tamm::TiledIndexSpace> phys_tis;
            std::vector<tamm::Tensor<Cplx>> mps_tensors;
            IdxType* result = nullptr;
            CuCtx cu_ctx_;

        // In the TN_TAMM class
        virtual void simulation_kernel(const std::vector<SVGate> &gates)
        {
            int rank = pg.rank().value();
            
            // The gate sorting and flattening logic remains the same.
            std::vector<SVGate> parallel_gates;
            std::vector<SVGate> sequential_gates;
            for (const auto& g : gates) {
                if (g.op_name == OP::C1 || g.op_name == OP::C2) {
                    parallel_gates.push_back(g);
                } else if (g.op_name == OP::M || g.op_name == OP::MA || g.op_name == OP::RESET) {
                    sequential_gates.push_back(g);
                }
            }
            
            std::vector<std::vector<SVGate>> layers;
            if (!parallel_gates.empty()) {
                std::vector<SVGate> flat_gates;
                flat_gates.reserve(parallel_gates.size() * 2); 
        
                for (const auto& g : parallel_gates) {
                    if (g.op_name == OP::C1) {
                        flat_gates.push_back(g);
                    } else { 
                        int a = g.ctrl;
                        int b = g.qubit;
                        if (b - a > 1) {
                            for (int k = a; k < b - 1; ++k) flat_gates.push_back(make_swap_sv(k, k + 1));
                            flat_gates.push_back(make_local_c2_sv(g, b - 1, b));
                            for (int k = b - 2; k >= a; --k) flat_gates.push_back(make_swap_sv(k, k + 1));
                        } else {
                            flat_gates.push_back(g);
                        }
                    }
                }
                
                layers.reserve(flat_gates.size());
                std::map<int, int> last_layer_map;
        
                for (const auto& g : flat_gates) {
                    if (g.op_name == OP::C1) place_c1(g, layers, last_layer_map);
                    else place_c2(g, g.ctrl, g.qubit, layers, last_layer_map);
                }
            }
            
            pg.barrier();
        
            // EXECUTION of layers
            for (int layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
                const auto& layer = layers[layer_idx];
                if (layer.empty()) continue;
                
                auto start_layer = std::chrono::high_resolution_clock::now();
        
                for (const auto& g : layer) {
                    if (g.op_name == OP::C1) {
                        // 1-qubit gates are always local to the owning rank
                        if (g.qubit >= start_qubit && g.qubit < end_qubit) {
                            std::array<Cplx, 4> U;
                            for (int i = 0; i < 4; ++i) U[i] = Cplx(g.gm_real[i], g.gm_imag[i]);
                            C1_GATE_local_kernel(mps_tensors[g.qubit], U);
                        }
                    } else if (g.op_name == OP::C2) {
                        IdxType q0 = g.ctrl;
                        IdxType q1 = g.qubit;
        
                        bool q0_is_local = (q0 >= start_qubit && q0 < end_qubit);
                        bool q1_is_local = (q1 >= start_qubit && q1 < end_qubit);
        
                        if (q0_is_local && q1_is_local) {
                            // --- INTERNAL GATE: Purely local computation ---
                            std::array<Cplx, 16> U4;
                            for (int i = 0; i < 16; ++i) U4[i] = Cplx(g.gm_real[i], g.gm_imag[i]);
                            C2_GATE_local_op(U4, q0, q1);
                        } else if (q0_is_local || q1_is_local) {
                            // --- BOUNDARY GATE: Requires communication ---
                            std::array<Cplx, 16> U4;
                            for (int i = 0; i < 16; ++i) U4[i] = Cplx(g.gm_real[i], g.gm_imag[i]);
                            C2_GATE_boundary_op(U4, q0, q1);
                        }
                    }
                }
                pg.barrier(); // Barrier after each layer to ensure global consistency
                
                auto end_layer = std::chrono::high_resolution_clock::now();
                double layer_time = std::chrono::duration<double>(end_layer - start_layer).count();
                if (i_proc == 0) {
                    std::cout << "Layer " << layer_idx << " | total_time = " << layer_time << " s" << std::endl;
                }
            }
            
            // Sequential gates (if any) would go here...
            pg.barrier();
        }

        // Add this function inside the TN_TAMM class
        void C2_GATE_local_op(const std::array<Cplx, 16> &U4, IdxType q0, IdxType q1)
        {
            // This entire operation uses the LOCAL scheduler and is network-free.
            
            // Perform contraction and SVD on local tensors
            tamm::Tensor<Cplx> M_local({bond_tis[q0], phys_tis[q0], phys_tis[q1], bond_tis[q1 + 1]});
            tamm::Tensor<Cplx> G4_local({phys_tis[q0], phys_tis[q1], phys_tis[q0], phys_tis[q1]});
            tamm::Tensor<Cplx> M2_local({bond_tis[q0], phys_tis[q0], phys_tis[q1], bond_tis[q1 + 1]});
            M_local.set_dense(); G4_local.set_dense(); M2_local.set_dense();
            sch_local.allocate(M_local, G4_local, M2_local).execute(exec_hw);
        
            sch_local(M_local("l","p0","p1","r") = mps_tensors[q0]("l","p0","b") * mps_tensors[q1]("b","p1","r")).execute(exec_hw);
            
            auto fill_g4 = [&](const tamm::IndexVector& bid, tamm::span<Cplx> buf){
                auto offsets = G4_local.block_offsets(bid);
                int p0p = offsets[0], p1p = offsets[1], p0 = offsets[2], p1 = offsets[3];
                buf[0] = U4[(p0p * 2 + p1p) * 4 + (p0 * 2 + p1)];
            };
            tamm::update_tensor(G4_local, fill_g4);
            
            sch_local(M2_local("l","p0p","p1p","r") = G4_local("p0p","p1p","p0","p1") * M_local("l","p0","p1","r")).execute(exec_hw);
        
            std::vector<Cplx> Ti_new_data, Tj_new_data;
            IdxType new_bond_dim = local_svd_and_reconstruct_data(M2_local, Ti_new_data, Tj_new_data, q0, q1);
            
            sch_local.deallocate(mps_tensors[q0], mps_tensors[q1]).execute(exec_hw);
        
            bond_dims[q0 + 1] = new_bond_dim;
            tamm::IndexSpace is_new_bond{tamm::range(new_bond_dim)};
            bond_tis[q0 + 1] = tamm::TiledIndexSpace(is_new_bond, block_size);
            
            mps_tensors[q0] = tamm::Tensor<Cplx>{bond_tis[q0], phys_tis[q0], bond_tis[q0 + 1]};
            mps_tensors[q1] = tamm::Tensor<Cplx>{bond_tis[q1], phys_tis[q1], bond_tis[q1 + 1]};
            mps_tensors[q0].set_dense();
            mps_tensors[q1].set_dense();
            sch_local.allocate(mps_tensors[q0], mps_tensors[q1]).execute(exec_hw);
        
            mps_tensors[q0].put(*(mps_tensors[q0].loop_nest().begin()), Ti_new_data);
            mps_tensors[q1].put(*(mps_tensors[q1].loop_nest().begin()), Tj_new_data);
        
            sch_local.deallocate(M_local, G4_local, M2_local).execute(exec_hw);
        }

        int get_owner_rank(IdxType qubit_idx) const {
            int world_size = pg.size().value();
            if (world_size == 1) return 0;
        
            IdxType qubits_per_rank = n_qubits / world_size; // CORRECTED
            IdxType remainder = n_qubits % world_size;       // CORRECTED
            
            IdxType boundary_qubit = remainder * (qubits_per_rank + 1);
        
            if (qubit_idx < boundary_qubit) {
                return qubit_idx / (qubits_per_rank + 1);
            } else {
                return remainder + (qubit_idx - boundary_qubit) / qubits_per_rank;
            }
        }

        void C2_GATE_boundary_op(const std::array<Cplx, 16> &U4, IdxType q0, IdxType q1)
        {
            int rank = pg.rank().value();
            
            // Step 1: Identify the two ranks involved.
            // By convention from the circuit flattener, q0 is always the left qubit.
            int left_rank = get_owner_rank(q0);
            int right_rank = get_owner_rank(q1);
        
            // Only the two ranks involved in the boundary gate participate.
            if (rank != left_rank && rank != right_rank) {
                return;
            }
        
            // Step 2 & 3: Exchange Tensors.
            // The right rank sends its tensor (T[q1]) to the left rank.
            // The left rank will perform the computation.
            tamm::Tensor<Cplx> T1_local; // Will hold the received tensor on left_rank
            
            if (rank == left_rank) {
                // Prepare to receive T[q1] from the right rank.
                T1_local = tamm::Tensor<Cplx>({bond_tis[q1], phys_tis[q1], bond_tis[q1 + 1]});
                T1_local.set_dense();
                sch_local.allocate(T1_local).execute(exec_hw);
        
                std::vector<Cplx> t1_buf(T1_local.size());
                MPI_Recv(t1_buf.data(), t1_buf.size() * sizeof(Cplx), MPI_BYTE,
                         right_rank, 0, pg.comm(), MPI_STATUS_IGNORE);
                T1_local.put(*(T1_local.loop_nest().begin()), t1_buf);
            }
            else { // rank == right_rank
                // Send T[q1] to the left rank.
                auto& T1_to_send = mps_tensors[q1];
                std::vector<Cplx> t1_buf(T1_to_send.size());
                T1_to_send.get(*(T1_to_send.loop_nest().begin()), t1_buf);
                MPI_Send(t1_buf.data(), t1_buf.size() * sizeof(Cplx), MPI_BYTE,
                         left_rank, 0, pg.comm());
            }
        
            // Step 4: Local Computation on the left_rank
            std::vector<Cplx> new_T0_data;
            std::vector<Cplx> new_T1_data;
            IdxType new_bond_dim = 0;
        
            if (rank == left_rank) {
                // Now left_rank has its own mps_tensors[q0] and the received T1_local.
                // The logic here is identical to the C2_GATE_local_op.
                tamm::Tensor<Cplx>& T0_local = mps_tensors[q0];
        
                tamm::Tensor<Cplx> M_local({bond_tis[q0], phys_tis[q0], phys_tis[q1], bond_tis[q1 + 1]});
                tamm::Tensor<Cplx> G4_local({phys_tis[q0], phys_tis[q1], phys_tis[q0], phys_tis[q1]});
                tamm::Tensor<Cplx> M2_local({bond_tis[q0], phys_tis[q0], phys_tis[q1], bond_tis[q1 + 1]});
                M_local.set_dense(); G4_local.set_dense(); M2_local.set_dense();
                sch_local.allocate(M_local, G4_local, M2_local).execute(exec_hw);
        
                sch_local(M_local("l","p0","p1","r") = T0_local("l","p0","b") * T1_local("b","p1","r")).execute(exec_hw);
                
                auto fill_g4 = [&](const tamm::IndexVector& bid, tamm::span<Cplx> buf){
                    auto offsets = G4_local.block_offsets(bid);
                    int p0p = offsets[0], p1p = offsets[1], p0 = offsets[2], p1 = offsets[3];
                    buf[0] = U4[(p0p * 2 + p1p) * 4 + (p0 * 2 + p1)];
                };
                tamm::update_tensor(G4_local, fill_g4);
                
                sch_local(M2_local("l","p0p","p1p","r") = G4_local("p0p","p1p","p0","p1") * M_local("l","p0","p1","r")).execute(exec_hw);
        
                new_bond_dim = local_svd_and_reconstruct_data(M2_local, new_T0_data, new_T1_data, q0, q1);
        
                sch_local.deallocate(M_local, G4_local, M2_local, T1_local).execute(exec_hw);
            }
            
            // Step 5: Broadcast new bond dimension from left_rank to all ranks.
            // All ranks must participate in this collective.
            MPI_Bcast(&new_bond_dim, 1, MPI_UNSIGNED_LONG_LONG, left_rank, pg.comm());
        
            // Step 6: All ranks update metadata and deallocate/reallocate tensors.
            bond_dims[q0 + 1] = new_bond_dim;
            tamm::IndexSpace is_new_bond{tamm::range(new_bond_dim)};
            bond_tis[q0 + 1] = tamm::TiledIndexSpace(is_new_bond, block_size);
        
            if (rank == left_rank) {
                sch_local.deallocate(mps_tensors[q0]).execute(exec_hw);
                mps_tensors[q0] = tamm::Tensor<Cplx>{bond_tis[q0], phys_tis[q0], bond_tis[q0 + 1]};
                mps_tensors[q0].set_dense();
                sch_local.allocate(mps_tensors[q0]).execute(exec_hw);
            }
            if (rank == right_rank) {
                sch_local.deallocate(mps_tensors[q1]).execute(exec_hw);
                mps_tensors[q1] = tamm::Tensor<Cplx>{bond_tis[q1], phys_tis[q1], bond_tis[q1 + 1]};
                mps_tensors[q1].set_dense();
                sch_local.allocate(mps_tensors[q1]).execute(exec_hw);
            }
            
            // Step 7: Distribute results back.
            // left_rank sends the new T[q1] data to right_rank.
            if (rank == left_rank) {
                // Put its own new data
                mps_tensors[q0].put(*(mps_tensors[q0].loop_nest().begin()), new_T0_data);
                // Send the other tensor's data
                MPI_Send(new_T1_data.data(), new_T1_data.size() * sizeof(Cplx), MPI_BYTE,
                         right_rank, 1, pg.comm());
            }
            else { // rank == right_rank
                // Receive the new data
                std::vector<Cplx> new_T1_buf(mps_tensors[q1].size());
                MPI_Recv(new_T1_buf.data(), new_T1_buf.size() * sizeof(Cplx), MPI_BYTE,
                         left_rank, 1, pg.comm(), MPI_STATUS_IGNORE);
                mps_tensors[q1].put(*(mps_tensors[q1].loop_nest().begin()), new_T1_buf);
            }
        }


        // In the TN_TAMM class
        std::vector<LocalGateResult> run_gates_parallel(const std::vector<SVGate>& batch)
        {
            int rank = pg.rank().value();
            //if (rank == 0) {
                //std::cout << ">> Starting parallel execution of a layer with " << batch.size() << " gates." << std::endl;
            //}
        
            // This is the distributed atomic counter. Each rank will fetch-and-add
            // to get a unique gate index, ensuring each gate in the batch is
            // processed exactly once across all ranks.
            tamm::AtomicCounterGA gate_counter(pg, 1);
            gate_counter.allocate(0);
            pg.barrier();
        
            std::vector<LocalGateResult> local_results;
        
            while (true)
            {
                // Atomically get the next available gate index from the distributed counter.
                long long gate_idx = gate_counter.fetch_add(0, 1);
        
                // If the index is out of bounds, all gates have been claimed. Exit the loop.
                if (gate_idx >= static_cast<long long>(batch.size())) {
                    break;
                }
        
                const SVGate& g = batch[gate_idx];
        
                if (g.op_name == OP::C1) {
                    // STRATEGY for C1 GATES: In-place update by the owner.
                    // The rank that owns the tensor data for the target qubit will perform the update directly.
                    // No result needs to be returned, as the update is applied immediately.
        
                    auto& tensor_to_update = mps_tensors[g.qubit];
                    
                    // Find which process owns the data for this tensor.
                    // We only need to check the first block since our tensors are not distributed block-wise.
                    auto [owner_proc, offset] = tensor_to_update.distribution().locate(*(tensor_to_update.loop_nest().begin()));
        
                    // Only the owner rank performs the computation.
                    if (pg.rank() == owner_proc) {
                        // //std::cout << "[RANK " << rank << "] Applying C1 gate on qubit " << g.qubit << " (owned locally)." << std::endl;
                        std::array<Cplx, 4> U;
                        for (int i = 0; i < 4; ++i) U[i] = Cplx(g.gm_real[i], g.gm_imag[i]);
                        C1_GATE_local_kernel(tensor_to_update, U);
                    }
        
                } else if (g.op_name == OP::C2) {
                    // STRATEGY for C2 GATES: Compute locally, then update collectively.
                    // Any available rank can compute the result of the C2 gate.
                    // The result (new tensor data) is stored in `local_results` and will be
                    // applied to the global state in the `apply_collective_updates` function.
                    
                    // //std::cout << "[RANK " << rank << "] Computing C2 gate on qubits (" << g.ctrl << ", " << g.qubit << ")." << std::endl;
                    std::array<Cplx, 16> U4;
                    for (int i = 0; i < 16; ++i) U4[i] = Cplx(g.gm_real[i], g.gm_imag[i]);
                    
                    LocalGateResult result = C2_GATE_COMPUTE(U4, g.ctrl, g.qubit);
                    if (result.is_valid) {
                        local_results.push_back(std::move(result));
                    }
                }
            }
        
            // Ensure all ranks have finished their assigned tasks before proceeding.
            pg.barrier(); 
            gate_counter.deallocate();
        
            //if (rank == 0) {
                 //std::cout << "<< Finished parallel execution of layer." << std::endl;
            //}
            
            return local_results;
        }

        std::vector<GateUpdateMetadata> allgather_metadata(const std::vector<LocalGateResult>& local_results) 
        {
            int rank = pg.rank().value();
            //std::cout << "[RANK " << rank << "] >> Entering allgather_metadata. Processing " << local_results.size() << " local results." << std::endl;
        
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
            //std::cout << "[RANK " << rank << "] allgather_metadata: Created " << local_metadata.size() << " local metadata entries." << std::endl;
        
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
            } //else {
                 //std::cout << "[RANK " << rank << "] allgather_metadata: No metadata to gather." << std::endl;
            //}
        
            //std::cout << "[RANK " << rank << "] << Exiting allgather_metadata. Total metadata entries gathered: " << all_metadata.size() << std::endl;
            return all_metadata;
        }

        // In the TN_TAM_class
        void apply_collective_updates(std::vector<LocalGateResult>& local_results)
        {
            int rank = pg.rank().value();
            int world_size = pg.size().value();
        
            // PHASE 1: Gather metadata about all updates across all ranks.
            // This tells every rank what happened and what the new tensor structures will be.
            auto all_metadata = allgather_metadata(local_results);
        
            tamm::Scheduler sch_global{ec};
            
            // PHASE 2: Deallocate old tensors and update bond dimension metadata.
            // This is a collective preparation step.
            std::set<IdxType> deallocated_sites; 
            for (const auto& meta : all_metadata)
            {
                if (!meta.is_valid) continue;
                
                if (deallocated_sites.find(meta.q0) == deallocated_sites.end()) {
                    sch_global.deallocate(mps_tensors[meta.q0]);
                    deallocated_sites.insert(meta.q0);
                }
                if (deallocated_sites.find(meta.q1) == deallocated_sites.end()) {
                    sch_global.deallocate(mps_tensors[meta.q1]);
                    deallocated_sites.insert(meta.q1);
                }
        
                bond_dims[meta.q0 + 1] = meta.new_bond_dim;
                tamm::IndexSpace is_new_bond{tamm::range(meta.new_bond_dim)};
                bond_tis[meta.q0 + 1] = tamm::TiledIndexSpace(is_new_bond, block_size);
            }
        
            // PHASE 3: Allocate new tensors.
            std::map<IdxType, tamm::Tensor<Cplx>> site_to_new_tensor;
            for(const auto& site : deallocated_sites) {
                site_to_new_tensor.emplace(
                    site,
                    tamm::Tensor<Cplx>{bond_tis[site], phys_tis[site], bond_tis[site + 1]}
                );
                site_to_new_tensor.at(site).set_dense();
                sch_global.allocate(site_to_new_tensor.at(site));
            }
            
            sch_global.execute(exec_hw);
            pg.barrier(); // Ensure allocations are visible everywhere.
        
            // PHASE 4: Efficient Data Redistribution with MPI
            std::vector<MPI_Request> requests;
            std::vector<std::vector<Cplx>> recv_buffers; // Buffers to hold incoming data
        
            // Determine what this rank needs to receive and post Irecvs
            for (const auto& meta : all_metadata) {
                if (!meta.is_valid) continue;
        
                auto& new_T0 = site_to_new_tensor.at(meta.q0);
                auto [owner_T0, offset_T0] = new_T0.distribution().locate(*(new_T0.loop_nest().begin()));
                
                auto& new_T1 = site_to_new_tensor.at(meta.q1);
                auto [owner_T1, offset_T1] = new_T1.distribution().locate(*(new_T1.loop_nest().begin()));
                
                if (rank == owner_T0.value()) {
                    recv_buffers.emplace_back(new_T0.size());
                    MPI_Request req;
                    MPI_Irecv(recv_buffers.back().data(), new_T0.size() * sizeof(Cplx), MPI_BYTE,
                              meta.original_rank, meta.q0, pg.comm(), &req);
                    requests.push_back(req);
                }
        
                if (rank == owner_T1.value()) {
                    recv_buffers.emplace_back(new_T1.size());
                    MPI_Request req;
                    MPI_Irecv(recv_buffers.back().data(), new_T1.size() * sizeof(Cplx), MPI_BYTE,
                              meta.original_rank, meta.q1, pg.comm(), &req);
                    requests.push_back(req);
                }
            }
        
            // Each rank sends the results it computed
            for (auto& result : local_results) {
                if (!result.is_valid) continue;
        
                auto& new_T0_dest = site_to_new_tensor.at(result.q0);
                auto [owner_T0, offset_T0] = new_T0_dest.distribution().locate(*(new_T0_dest.loop_nest().begin()));
        
                auto& new_T1_dest = site_to_new_tensor.at(result.q1);
                auto [owner_T1, offset_T1] = new_T1_dest.distribution().locate(*(new_T1_dest.loop_nest().begin()));
        
                MPI_Request req;
                MPI_Isend(result.new_T0_data.data(), result.new_T0_data.size() * sizeof(Cplx), MPI_BYTE,
                          owner_T0.value(), result.q0, pg.comm(), &req);
                requests.push_back(req);
        
                MPI_Isend(result.new_T1_data.data(), result.new_T1_data.size() * sizeof(Cplx), MPI_BYTE,
                          owner_T1.value(), result.q1, pg.comm(), &req);
                requests.push_back(req);
            }
            
            // Wait for all non-blocking communications to complete
            MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        
            // PHASE 5: Local data placement.
            // Each rank now has the data it needs in its recv_buffers. This is just a fast memory copy.
            int recv_buf_idx = 0;
            for (const auto& meta : all_metadata) {
                if (!meta.is_valid) continue;
        
                auto& new_T0 = site_to_new_tensor.at(meta.q0);
                auto [owner_T0, offset_T0] = new_T0.distribution().locate(*(new_T0.loop_nest().begin()));
        
                auto& new_T1 = site_to_new_tensor.at(meta.q1);
                auto [owner_T1, offset_T1] = new_T1.distribution().locate(*(new_T1.loop_nest().begin()));
        
                if (rank == owner_T0.value()) {
                    new_T0.put(*(new_T0.loop_nest().begin()), recv_buffers[recv_buf_idx++]);
                }
                if (rank == owner_T1.value()) {
                    new_T1.put(*(new_T1.loop_nest().begin()), recv_buffers[recv_buf_idx++]);
                }
            }
        
            pg.barrier(); // Final sync before updating the main state vector.
        
            // PHASE 6: Update the main MPS state with the new tensors.
            for(auto const& [site, new_tensor] : site_to_new_tensor) {
                mps_tensors[site] = new_tensor;
            }
        }
        // This function is now a "local kernel". It will be called via RPC
        // on the rank that owns the target tensor block.
        void C1_GATE_local_kernel(tamm::Tensor<Cplx>& target_tensor, const std::array<Cplx, 4>& U)
        {
            // 1. Create a local execution context for this operation.
            tamm::ProcGroup self_pg = tamm::ProcGroup::create_self();
            tamm::ExecutionContext ec_local{self_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::local};
            tamm::Scheduler sch_local{ec_local};
        
            // 2. Get the TiledIndexSpaces from the target tensor.
            auto tis_l = target_tensor.tiled_index_spaces()[0];
            auto tis_p = target_tensor.tiled_index_spaces()[1];
            auto tis_r = target_tensor.tiled_index_spaces()[2];
        
            // 3. Create a local tensor to hold the gate matrix.
            tamm::Tensor<Cplx> G({tis_p, tis_p});
            G.set_dense();
        
            // 4. Create a local tensor for the result.
            tamm::Tensor<Cplx> T_new({tis_l, tis_p, tis_r});
            T_new.set_dense();
            
            // 5. Create a local copy of the input tensor data.
            tamm::Tensor<Cplx> T_in({tis_l, tis_p, tis_r});
            T_in.set_dense();
        
            // 6. Allocate and fill all local tensors.
            sch_local.allocate(G, T_new, T_in).execute(exec_hw);
        
            auto fill_g = [&](const tamm::IndexVector& bid, tamm::span<Cplx> buf){
                auto offsets = G.block_offsets(bid);
                int pout = offsets[0], pin = offsets[1];
                buf[0] = U[pout * 2 + pin];
            };
            tamm::update_tensor(G, fill_g);
        
            std::vector<Cplx> t_in_buf(T_in.size());
            target_tensor.get(*(target_tensor.loop_nest().begin()), t_in_buf);
            T_in.put(*(T_in.loop_nest().begin()), t_in_buf);
        
            // 7. Perform the contraction locally.
            sch_local(T_new("l","p'","r") = G("p'","p") * T_in("l","p","r")).execute(exec_hw);
        
            // 8. Get the result back into a host buffer.
            std::vector<Cplx> t_out_buf(T_new.size());
            T_new.get(*(T_new.loop_nest().begin()), t_out_buf);
        
            // 9. Put the result back into the original global tensor.
            target_tensor.put(*(target_tensor.loop_nest().begin()), t_out_buf);
        
            // 10. Clean up local resources.
            sch_local.deallocate(G, T_new, T_in).execute(exec_hw);
            self_pg.destroy_coll();
        }

        // This function now returns a struct containing the new data and metadata
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
            sch_local.allocate(T0_local, T1_local).execute(exec_hw);
        
            // 3. GET data from the global mps_tensors into local std::vectors, then PUT to local tensors.
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
            sch_local.allocate(M_local, G4_local, M2_local).execute(exec_hw);
        
            sch_local(M_local("l","p0","p1","r") = T0_local("l","p0","b") * T1_local("b","p1","r")).execute(exec_hw);
            
            auto fill_g4 = [&](const tamm::IndexVector& bid, tamm::span<Cplx> buf){
                auto offsets = G4_local.block_offsets(bid);
                int p0p = offsets[0], p1p = offsets[1], p0 = offsets[2], p1 = offsets[3];
                buf[0] = U4[(p0p * 2 + p1p) * 4 + (p0 * 2 + p1)];
            };
            tamm::update_tensor(G4_local, fill_g4);
            
            sch_local(M2_local("l","p0p","p1p","r") = G4_local("p0p","p1p","p0","p1") * M_local("l","p0","p1","r")).execute(exec_hw);
        
            // 5. Perform SVD on the local M2_local tensor.
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
            sch_local.deallocate(T0_local, T1_local, M_local, G4_local, M2_local).execute(exec_hw);
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
            //std::cout << "[RANK " << rank << "] ---> gpu_svd_jacobi: Entered. Matrix dimensions (m, n): (" << m << ", " << n << ")." << std::endl;

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

            //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Allocating GPU memory for A, S, U, V, info." << std::endl;
            cudaMalloc((void**)&d_A, sizeof(cuDoubleComplex) * (size_t)lda * (size_t)n);
            cudaMalloc((void**)&d_S, sizeof(double) * (size_t)k);
            cudaMalloc((void**)&d_U, sizeof(cuDoubleComplex) * (size_t)ldu * (size_t)k);
            cudaMalloc((void**)&d_V, sizeof(cuDoubleComplex) * (size_t)ldv * (size_t)k);
            cudaMalloc((void**)&d_info, sizeof(int));

            //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Copying host matrix A to device." << std::endl;
            cudaMemcpyAsync(d_A, reinterpret_cast<const cuDoubleComplex*>(A_h),
                            sizeof(cuDoubleComplex) * (size_t)lda * (size_t)n,
                            cudaMemcpyHostToDevice, cu_ctx_.stream);

            int lwork_req = 0;
            //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Querying buffer size for Zgesvdj." << std::endl;
            cusolverDnZgesvdj_bufferSize(cu_ctx_.solver, CUSOLVER_EIG_MODE_VECTOR, econ,
                                         m, n, d_A, lda, d_S, d_U, ldu, d_V, ldv,
                                         &lwork_req, cu_ctx_.jp);
            //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Required buffer size (lwork_req): " << lwork_req << "." << std::endl;

            if (lwork_req > cu_ctx_.lwork_jac) {
                //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Reallocating GPU workspace from " << cu_ctx_.lwork_jac << " to " << lwork_req << "." << std::endl;
                if (cu_ctx_.d_work_jac) cudaFree(cu_ctx_.d_work_jac);
                cu_ctx_.lwork_jac = lwork_req;
                cudaMalloc((void**)&cu_ctx_.d_work_jac, sizeof(cuDoubleComplex) * (size_t)cu_ctx_.lwork_jac);
            }

            //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Calling cusolverDnZgesvdj to perform SVD on GPU..." << std::endl;
            cusolverDnZgesvdj(cu_ctx_.solver, CUSOLVER_EIG_MODE_VECTOR, econ,
                              m, n, d_A, lda, d_S, d_U, ldu, d_V, ldv,
                              reinterpret_cast<cuDoubleComplex*>(cu_ctx_.d_work_jac),
                              cu_ctx_.lwork_jac, d_info, cu_ctx_.jp);

            //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Synchronizing CUDA stream..." << std::endl;
            cudaStreamSynchronize(cu_ctx_.stream);
            //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Stream synchronized. GPU computation finished." << std::endl;

            S.resize((size_t)k);
            std::vector<Cplx> U_col((size_t)ldu * (size_t)k);
            std::vector<Cplx> V_col((size_t)ldv * (size_t)k);

            //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Copying results S, U, V from device to host." << std::endl;
            cudaMemcpy(S.data(), d_S, sizeof(double) * (size_t)k, cudaMemcpyDeviceToHost);
            cudaMemcpy(U_col.data(), d_U, sizeof(Cplx) * (size_t)ldu * (size_t)k, cudaMemcpyDeviceToHost);
            cudaMemcpy(V_col.data(), d_V, sizeof(Cplx) * (size_t)ldv * (size_t)k, cudaMemcpyDeviceToHost);
            //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: D2H copy complete." << std::endl;

            //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Transposing U and V to row-major format." << std::endl;
            U_row.resize((size_t)m * (size_t)k);
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < k; ++j)
                    U_row[(size_t)i * (size_t)k + (size_t)j] = U_col[(size_t)i + (size_t)j * (size_t)ldu];

            VT_row.resize((size_t)k * (size_t)n);
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < n; ++j)
                    VT_row[(size_t)i * (size_t)n + (size_t)j] = std::conj(V_col[(size_t)j + (size_t)i * (size_t)ldv]);

            //std::cout << "[RANK " << rank << "] gpu_svd_jacobi: Freeing GPU memory." << std::endl;
            cudaFree(d_info);
            cudaFree(d_V);
            cudaFree(d_U);
            cudaFree(d_S);
            cudaFree(d_A);

            //std::cout << "[RANK " << rank << "] <--- gpu_svd_jacobi: Exiting." << std::endl;
        }

//        IdxType local_svd_and_reconstruct_data(
//            tamm::Tensor<Cplx>& M2_local,
//            std::vector<Cplx>& Ti_new_data,
//            std::vector<Cplx>& Tj_new_data,
//            IdxType q0, IdxType q1)
//        {
//            int rank = pg.rank().value();
//            //std::cout << "[RANK " << rank << "] ---> local_svd_and_reconstruct_data (EIGEN): Entered for qubits (" << q0 << ", " << q1 << ")." << std::endl;
//        
//            // 1. Extract dimensions from the input tensor
//            const IdxType phys_dim = 2;
//            IdxType Dl = M2_local.tiled_index_spaces()[0].index_space().num_indices();
//            IdxType Dr = M2_local.tiled_index_spaces()[3].index_space().num_indices();
//        
//            Eigen::Index m = Dl * phys_dim;
//            Eigen::Index n = phys_dim * Dr;
//        
//            // 2. Reshape the row-major TAMM tensor data into a column-major Eigen matrix.
//            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> mat(m, n);
//            
//            std::vector<Cplx> M2_hostbuf(M2_local.size());
//            M2_local.get(*(M2_local.loop_nest().begin()), M2_hostbuf);
//        
//            size_t c = 0;
//            for (size_t l = 0; l < Dl; ++l) {
//                for (size_t p0 = 0; p0 < phys_dim; ++p0) {
//                    for (size_t p1 = 0; p1 < phys_dim; ++p1) {
//                        for (size_t r = 0; r < Dr; ++r, ++c) {
//                            // Eigen's operator() handles the column-major layout automatically.
//                            mat(l * phys_dim + p0, p1 * Dr + r) = M2_hostbuf[c];
//                        }
//                    }
//                }
//            }
//            //std::cout << "[RANK " << rank << "] local_svd_and_reconstruct_data (EIGEN): Reshape to Eigen matrix complete." << std::endl;
//            
//            // 3. Compute the SVD using Eigen's robust BDCSVD.
//            Eigen::BDCSVD<decltype(mat)> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
//            auto svals = svd.singularValues();
//            //std::cout << "[RANK " << rank << "] local_svd_and_reconstruct_data (EIGEN): SVD computation complete." << std::endl;
//        
//            // 4. Truncate based on singular value cutoff and max bond dimension.
//            std::vector<IdxType> keep;
//            keep.reserve(svals.size());
//            for (IdxType i = 0; i < svals.size(); ++i) {
//                if (std::abs(svals(i)) >= sv_cutoff) {
//                    keep.push_back(i);
//                }
//            }
//            
//            IdxType chi = std::min<IdxType>(max_bond_dim, IdxType(keep.size()));
//            if (chi == 0 && svals.size() > 0) {
//                chi = 1; // Prevent bond dimension from ever becoming zero.
//            }
//            //std::cout << "[RANK " << rank << "] local_svd_and_reconstruct_data (EIGEN): Truncation complete. New bond dimension (chi): " << chi << "." << std::endl;
//        
//            // 5. Extract the truncated U, S, and Vh matrices.
//            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Umat(mat.rows(), chi);
//            Eigen::Matrix<Cplx, Eigen::Dynamic, 1> kept_svals(chi);
//            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Vh(chi, mat.cols());
//            for (IdxType k = 0; k < chi; ++k) {
//                IdxType i = keep[k];
//                Umat.col(k)   = svd.matrixU().col(i);
//                kept_svals(k) = svals(i);
//                Vh.row(k)     = svd.matrixV().col(i).adjoint();
//            }
//            
//            // 6. Populate the output vectors with the data for the new tensors.
//            
//            // 6a. Populate the new left tensor data (from Umat)
//            Ti_new_data.resize(Dl * phys_dim * chi);
//            c = 0;
//            for (size_t l = 0; l < Dl; ++l) {
//                for (size_t p0 = 0; p0 < phys_dim; ++p0) {
//                    for (size_t b = 0; b < chi; ++b, ++c) {
//                        Ti_new_data[c] = Umat(l * phys_dim + p0, b);
//                    }
//                }
//            }
//        
//            // 6b. Populate the new right tensor data (from S * Vh)
//            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> SV = kept_svals.asDiagonal() * Vh;
//            Tj_new_data.resize(chi * phys_dim * Dr);
//            c = 0;
//            for (size_t b = 0; b < chi; ++b) {
//                for (size_t p1 = 0; p1 < phys_dim; ++p1) {
//                    for (size_t r = 0; r < Dr; ++r, ++c) {
//                        Tj_new_data[c] = SV(b, p1 * Dr + r);
//                    }
//                }
//            }
//        
//            //std::cout << "[RANK " << rank << "] <--- local_svd_and_reconstruct_data (EIGEN): Exiting." << std::endl;
//            return chi;
//        }


        IdxType local_svd_and_reconstruct_data(
            tamm::Tensor<Cplx>& M2_local,
            std::vector<Cplx>& Ti_new_data,
            std::vector<Cplx>& Tj_new_data,
            IdxType q0, IdxType q1)
        {
            int rank = pg.rank().value();
            //std::cout << "[RANK " << rank << "] ---> local_svd_and_reconstruct_data: Entered for qubits (" << q0 << ", " << q1 << ")." << std::endl;
        
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
            //std::cout << "[RANK " << rank << "] local_svd_and_reconstruct_data: Reshape complete." << std::endl;
        
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
            //std::cout << "[RANK " << rank << "] local_svd_and_reconstruct_data: Truncation complete. New bond dimension (chi): " << chi << "." << std::endl;
        
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
        
            //std::cout << "[RANK " << rank << "] <--- local_svd_and_reconstruct_data: Exiting." << std::endl;
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
