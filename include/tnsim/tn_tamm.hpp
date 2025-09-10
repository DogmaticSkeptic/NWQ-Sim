#pragma once

#include <mpi.h>
#include <numeric> 

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
        std::vector<Cplx> new_T0_data;
        std::vector<Cplx> new_T1_data;
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
        int original_rank;
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
            block_size(1024),
            max_bond_dim(max_bond_dim),
            sv_cutoff(sv_cutoff),
            pg(init_pg()),
            ec(pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::ga)
        {
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
            // Initialize timing accumulators at the start of the simulation
            total_synchronization_time = std::chrono::duration<double>(0.0);
            total_contraction_time = std::chrono::duration<double>(0.0);
            total_svd_time = std::chrono::duration<double>(0.0);
            total_data_movement_time = std::chrono::duration<double>(0.0);
            total_scheduling_time = std::chrono::duration<double>(0.0);
            total_resource_management_time = std::chrono::duration<double>(0.0);
        
            IdxType original_gate_count = circuit->num_gates();
            std::vector<SVGate> gates = fuse_circuit_sv(circuit);
            IdxType fused_gate_count = gates.size();
            assert(circuit->num_qubits() == n_qubits);
        
            pg.barrier();
            auto start_time = std::chrono::high_resolution_clock::now();
            simulation_kernel(gates);
            auto end_time = std::chrono::high_resolution_clock::now();
            pg.barrier();
        
            std::chrono::duration<double> total_simulation_time = end_time - start_time;
        
            if (pg.rank().value() == 0) {
                std::cout << "-----------------------------------------------------" << std::endl;
                std::cout << "Simulation Timing Results:" << std::endl;
                std::cout << "-----------------------------------------------------" << std::endl;
                std::cout << std::fixed << std::setprecision(6);
                std::cout << "Total simulation time          : " << total_simulation_time.count() << " seconds." << std::endl;
                std::cout << "  - Total tensor contraction     : " << total_contraction_time.count() << " seconds." << std::endl;
                std::cout << "  - Total SVD execution          : " << total_svd_time.count() << " seconds." << std::endl;
                std::cout << "  - Total synchronization (barriers) : " << total_synchronization_time.count() << " seconds." << std::endl;
                std::cout << "  - Total data movement (get/put/gather) : " << total_data_movement_time.count() << " seconds." << std::endl;
                std::cout << "  - Total scheduling (gate layering) : " << total_scheduling_time.count() << " seconds." << std::endl;
                std::cout << "  - Total resource management (alloc/dealloc) : " << total_resource_management_time.count() << " seconds." << std::endl;
                
                double accounted_time = total_contraction_time.count() +
                                        total_svd_time.count() +
                                        total_synchronization_time.count() +
                                        total_data_movement_time.count() +
                                        total_scheduling_time.count() +
                                        total_resource_management_time.count();
                
                std::cout << "-----------------------------------------------------" << std::endl;
                std::cout << "Total accounted time           : " << accounted_time << " seconds." << std::endl;
                std::cout << "Unaccounted time               : " << total_simulation_time.count() - accounted_time << " seconds." << std::endl;
                std::cout << "-----------------------------------------------------" << std::endl;
        
                // Write the detailed timing breakdown to a CSV file
                std::ofstream csv_file("timings.csv", std::ios::app);
                if (csv_file.is_open()) {
                    csv_file << total_simulation_time.count() << ","
                             << total_contraction_time.count() << ","
                             << total_svd_time.count() << ","
                             << total_synchronization_time.count() << ","
                             << total_data_movement_time.count() << ","
                             << total_scheduling_time.count() << ","
                             << total_resource_management_time.count() << "\n";
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

        // Add this helper function inside the TN_TAMM class, maybe near the top.
        void print_buffer_diag(const std::string& name, IdxType q_idx, const std::vector<Cplx>& buf) {
            if (buf.empty()) {
                std::cout << "[RANK " << pg.rank().value() << "] DIAG " << name << " q=" << q_idx << ": EMPTY" << std::endl;
                return;
            }
            double norm_sq = 0.0;
            for(const auto& val : buf) norm_sq += std::norm(val);
            
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3);
            // Print first few elements to get a sense of the data
            for(size_t i = 0; i < std::min((size_t)4, buf.size()); ++i) {
                ss << "(" << buf[i].real() << "," << buf[i].imag() << ") ";
            }
        
            std::cout << "[RANK " << pg.rank().value() << "] DIAG " << name << " q=" << q_idx 
                      << " | norm=" << std::sqrt(norm_sq) << " | data=[" << ss.str() << "...]" << std::endl;
        }

        void print_mps_tensor(IdxType site)
        {
            // Only rank 0 is responsible for printing to avoid console spam
            if (pg.rank().value() != 0) {
                return;
            }

            // Get the tensor and its dimensions
            auto& T = mps_tensors[site];
            IdxType Dl = bond_dims[site];
            IdxType Dp = phys_dims[site];
            IdxType Dr = bond_dims[site + 1];

            // On rank 0, create a host buffer and use T.get() to pull the data
            // from its distributed location into this local buffer.
            std::vector<Cplx> hostbuf(T.size());
            T.get(*(T.loop_nest().begin()), hostbuf);

            // Print the formatted output, identical to the serial version
            printf("--- Tensor T_%lld ---\n", site);
            printf("Dimensions: [l=%lld, p=%lld, r=%lld]\n", Dl, Dp, Dr);
            
            size_t idx = 0;
            for (IdxType l = 0; l < Dl; ++l) {
                printf("  l=%lld:\n", l);
                for (IdxType p = 0; p < Dp; ++p) {
                    printf("    p=%lld: [ ", p);
                    for (IdxType r = 0; r < Dr; ++r) {
                        // To avoid printing tiny numbers from floating point inaccuracies
                        double real_part = std::abs(hostbuf[idx].real()) < 1e-10 ? 0.0 : hostbuf[idx].real();
                        double imag_part = std::abs(hostbuf[idx].imag()) < 1e-10 ? 0.0 : hostbuf[idx].imag();
                        printf("(%.3f, %.3f) ", real_part, imag_part);
                        idx++;
                    }
                    printf("]\n");
                }
            }
        }


        // Helper to print a 4-index tensor like the merged M2
        void print_4_index_tensor(tamm::Tensor<Cplx>& T, const std::string& name, IdxType q0, IdxType q1) {
            int rank = pg.rank().value();

            // Get dimensions directly from the local tensor's index spaces
            IdxType Dl = T.tiled_index_spaces()[0].index_space().num_indices();
            IdxType Dp0 = T.tiled_index_spaces()[1].index_space().num_indices();
            IdxType Dp1 = T.tiled_index_spaces()[2].index_space().num_indices();
            IdxType Dr = T.tiled_index_spaces()[3].index_space().num_indices();

            std::vector<Cplx> hostbuf(T.size());
            T.get(*(T.loop_nest().begin()), hostbuf);

            printf("[RANK %d] --- Contents of %s for Qubits (%lld, %lld) ---\n", rank, name.c_str(), q0, q1);
            printf("[RANK %d] Dimensions: [l=%lld, p0=%lld, p1=%lld, r=%lld]\n", rank, Dl, Dp0, Dp1, Dr);

            size_t idx = 0;
            for (IdxType l = 0; l < Dl; ++l) {
                for (IdxType p0 = 0; p0 < Dp0; ++p0) {
                    for (IdxType p1 = 0; p1 < Dp1; ++p1) {
                         printf("[RANK %d]   l=%lld, p0=%lld, p1=%lld: [ ", rank, l, p0, p1);
                         for (IdxType r = 0; r < Dr; ++r) {
                             double real_part = std::abs(hostbuf[idx].real()) < 1e-10 ? 0.0 : hostbuf[idx].real();
                             double imag_part = std::abs(hostbuf[idx].imag()) < 1e-10 ? 0.0 : hostbuf[idx].imag();
                             printf("(%.3f, %.3f) ", real_part, imag_part);
                             idx++;
                         }
                         printf("]\n");
                    }
                }
            }
            printf("[RANK %d] --- End of %s ---\n", rank, name.c_str());
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

        std::chrono::duration<double> total_synchronization_time{0.0};
        std::chrono::duration<double> total_contraction_time{0.0};
        std::chrono::duration<double> total_svd_time{0.0};
        std::chrono::duration<double> total_data_movement_time{0.0};
        std::chrono::duration<double> total_scheduling_time{0.0};
        std::chrono::duration<double> total_resource_management_time{0.0};

        virtual void simulation_kernel(const std::vector<SVGate> &gates)
        {
            int rank = pg.rank().value();
        
            std::vector<SVGate> parallel_gates;
            std::vector<SVGate> sequential_gates;
            for (const auto& g : gates) {
                if (g.op_name == OP::C1 || g.op_name == OP::C2) {
                    parallel_gates.push_back(g);
                } else if (g.op_name == OP::M || g.op_name == OP::MA || g.op_name == OP::RESET) {
                    sequential_gates.push_back(g);
                }
            }
        
            // === NEW: Print Initial State ===
            pg.barrier();
            if (rank == 0) {
                std::cout << "\n<==================== Starting Simulation Kernel ====================>" << std::endl;
                std::cout << "===== Initial MPS State =====" << std::endl;
                for (IdxType q_idx = 0; q_idx < n_qubits; ++q_idx) {
                    print_mps_tensor(q_idx);
                }
                std::cout << "=============================" << std::endl;
            }
            // =============================

            if (!parallel_gates.empty()) {
                auto start_scheduling = std::chrono::high_resolution_clock::now();
                std::vector<SVGate> flat_gates;
                flat_gates.reserve(parallel_gates.size() * 2);
        
                for (const auto& g : parallel_gates) {
                    if (g.op_name == OP::C1) {
                        flat_gates.push_back(g);
                    } else { // It must be OP::C2
                        int a = g.ctrl;
                        int b = g.qubit;
        
                        if (std::abs(b - a) > 1) { // Fixed to handle both a > b and b > a
                            int start = std::min(a,b);
                            int end = std::max(a,b);
                            for (int k = start; k < end - 1; ++k) {
                                flat_gates.push_back(make_swap_sv(k, k + 1));
                            }
                            flat_gates.push_back(make_local_c2_sv(g, end - 1, end));
                            for (int k = end - 2; k >= start; --k) {
                                flat_gates.push_back(make_swap_sv(k, k + 1));
                            }
                        } else {
                            flat_gates.push_back(g);
                        }
                    }
                }
        
                std::vector<std::vector<SVGate>> layers;
                layers.reserve(flat_gates.size());
                std::map<int, int> last_layer_map;
        
                for (const auto& g : flat_gates) {
                    if (g.op_name == OP::C1) {
                        place_c1(g, layers, last_layer_map);
                    } else {
                        place_c2(g, g.ctrl, g.qubit, layers, last_layer_map);
                    }
                }
                auto end_scheduling = std::chrono::high_resolution_clock::now();
                total_scheduling_time += (end_scheduling - start_scheduling);
        
                pg.barrier();
        
                // EXECUTION of layers
                for (int layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
                    const auto& layer = layers[layer_idx];
                    if (layer.empty()) {
                        continue;
                    }

                    if (pg.rank().value() == 0) {
                        std::cout << "\n<====================== STARTING LAYER " << layer_idx << " ======================>" << std::endl;
                    }
        
                    pg.barrier();
                    auto local_update_results = run_gates_parallel(layer);
        
                    auto start_sync = std::chrono::high_resolution_clock::now();
                    pg.barrier(); 
                    apply_collective_updates(local_update_results);
                    pg.barrier();
                    auto end_sync = std::chrono::high_resolution_clock::now();
                    
                    total_synchronization_time += (end_sync - start_sync);

                    // === NEW: Print State After Layer ===
                    pg.barrier();
                    if (rank == 0) {
                        std::cout << "\n===== MPS State after Layer " << layer_idx << " =====\n";
                        for (IdxType q_idx = 0; q_idx < n_qubits; ++q_idx) {
                            print_mps_tensor(q_idx);
                        }
                        std::cout << "===================================\n";
                        std::cout << "<====================== FINISHED LAYER " << layer_idx << " ======================>\n" << std::endl;
                    }
                    pg.barrier();
                    // =====================================
                }
            }
        
            pg.barrier();
        
            for (const auto &g : sequential_gates) {
                if (g.op_name == OP::RESET) {
                    RESET_GATE(g.qubit);
                } else if (g.op_name == OP::M) {
                    M_GATE(g.qubit);
                } else if (g.op_name == OP::MA) {
                    MA_GATE(g.qubit); 
                }
            }
        
            pg.barrier();
            if (rank == 0) {
                std::cout << "\n<==================== Simulation Kernel Finished ====================>\n";
            }
        }

        // Replace the existing run_gates_parallel function with this one
        std::vector<LocalGateResult> run_gates_parallel(const std::vector<SVGate>& batch)
        {
            int rank = pg.rank().value();
            if (rank == 0) {
                std::cout << ">> Starting parallel execution of a layer with " << batch.size() << " gates." << std::endl;
            }
        
            tamm::AtomicCounterGA gate_counter(pg, 1);
            gate_counter.allocate(0);
            pg.barrier();
        
            std::vector<LocalGateResult> local_results;
        
            while (true)
            {
                long long gate_idx = gate_counter.fetch_add(0, 1);
                if (gate_idx >= static_cast<long long>(batch.size())) break;
        
                const SVGate& g = batch[gate_idx];
                std::cout << "[RANK " << rank << "] picked up gate index " << gate_idx << " (op: " 
                          << g.op_name << ", q: " << g.qubit << ", c: " << g.ctrl << ")" << std::endl;
        
                if (g.op_name == OP::C1) {
                    auto& tensor_to_update = mps_tensors[g.qubit];
                    auto [owner_proc, offset] = tensor_to_update.distribution().locate(*(tensor_to_update.loop_nest().begin()));
                    
                    std::cout << "[RANK " << rank << "] Gate " << gate_idx << " is C1 on qubit " << g.qubit 
                              << ". Owner is rank " << owner_proc.value() << "." << std::endl;
        
                    if (pg.rank() == owner_proc) {
                        std::array<Cplx, 4> U;
                        for (int i = 0; i < 4; ++i) U[i] = Cplx(g.gm_real[i], g.gm_imag[i]);
                        C1_GATE_local_kernel(tensor_to_update, g.qubit, U);
                    }
                } else if (g.op_name == OP::C2) {
                    std::cout << "[RANK " << rank << "] Computing C2 gate on qubits (" << g.ctrl << ", " << g.qubit << ")." << std::endl;
                    std::array<Cplx, 16> U4;
                    for (int i = 0; i < 16; ++i) U4[i] = Cplx(g.gm_real[i], g.gm_imag[i]);
                    
                    LocalGateResult result = C2_GATE_COMPUTE(U4, g.ctrl, g.qubit);
                    if (result.is_valid) {
                        local_results.push_back(std::move(result));
                    }
                }
            }
        
            pg.barrier(); 
            gate_counter.deallocate();
        
            if (rank == 0) {
                 std::cout << "<< Finished parallel execution of layer." << std::endl;
            }
            
            return local_results;
        }

        std::vector<GateUpdateMetadata> allgather_metadata(const std::vector<LocalGateResult>& local_results)
        {
            int rank = pg.rank().value();
            std::cout << "[RANK " << rank << "] >> Entering allgather_metadata. Processing "
                      << local_results.size() << " local results." << std::endl;
        
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
        
            // --- DIAGNOSTIC: Print local metadata before communication ---
            std::cout << "[RANK " << rank << "] allgather_metadata: Created " << local_metadata.size()
                      << " local metadata entries to share." << std::endl;
            for (size_t i = 0; i < local_metadata.size(); ++i) {
                const auto& m = local_metadata[i];
                std::cout << "[RANK " << rank << "]   - Local Meta [" << i << "]: q0=" << m.q0
                          << ", q1=" << m.q1 << ", new_bond_dim=" << m.new_bond_dim
                          << ", owner=" << m.original_rank << std::endl;
            }
        
            int local_size_bytes = local_metadata.size() * sizeof(GateUpdateMetadata);
            std::vector<int> all_sizes_bytes(pg.size().value());
        
            // --- DIAGNOSTIC: Print parameters for the first collective (Allgather) ---
            std::cout << "[RANK " << rank << "] allgather_metadata: Preparing for pg.allgather. My local_size_bytes = "
                      << local_size_bytes << std::endl;
        
            pg.allgather(&local_size_bytes, 1, all_sizes_bytes.data(), 1);
        
            std::vector<int> displacements_bytes(pg.size().value(), 0);
            int total_size_bytes = 0;
            for (size_t i = 0; i < all_sizes_bytes.size(); ++i) {
                if (i > 0) {
                    displacements_bytes[i] = displacements_bytes[i-1] + all_sizes_bytes[i-1];
                }
                total_size_bytes += all_sizes_bytes[i];
            }
        
            // --- DIAGNOSTIC: Print parameters for the second collective (Allgatherv) ---
            std::stringstream ss_sizes, ss_displs;
            for(int s : all_sizes_bytes) ss_sizes << s << " ";
            for(int d : displacements_bytes) ss_displs << d << " ";
            std::cout << "[RANK " << rank << "] allgather_metadata: Preparing for MPI_Allgatherv."
                      << "\n[RANK " << rank << "]   - Total size (bytes): " << total_size_bytes
                      << "\n[RANK " << rank << "]   - All sizes (bytes):  [ " << ss_sizes.str() << "]"
                      << "\n[RANK " << rank << "]   - Displacements (bytes): [ " << ss_displs.str() << "]" << std::endl;
        
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
                 std::cout << "[RANK " << rank << "] allgather_metadata: No metadata to gather across all ranks." << std::endl;
            }
        
            // --- DIAGNOSTIC: Print the combined global metadata received by this rank ---
            std::cout << "[RANK " << rank << "] allgather_metadata: MPI_Allgatherv complete. Received "
                      << all_metadata.size() << " total metadata entries." << std::endl;
            for (size_t i = 0; i < all_metadata.size(); ++i) {
                const auto& m = all_metadata[i];
                std::cout << "[RANK " << rank << "]   - Global Meta [" << i << "]: q0=" << m.q0
                          << ", q1=" << m.q1 << ", new_bond_dim=" << m.new_bond_dim
                          << ", owner=" << m.original_rank << std::endl;
            }
        
        
            std::cout << "[RANK " << rank << "] << Exiting allgather_metadata." << std::endl;
            return all_metadata;
        }
        
        void apply_collective_updates(std::vector<LocalGateResult>& local_results)
        {
            int rank = pg.rank().value();
            std::cout << "[RANK " << rank << "] >> Entering apply_collective_updates with "
                      << local_results.size() << " local results." << std::endl;
        
            auto start_gather = std::chrono::high_resolution_clock::now();
            auto all_metadata = allgather_metadata(local_results);
            auto end_gather = std::chrono::high_resolution_clock::now();
            total_data_movement_time += (end_gather - start_gather);
        
            std::cout << "[RANK " << rank << "] apply_collective_updates: Metadata gathered. Total updates to apply: "
                      << all_metadata.size() << std::endl;
        
            tamm::Scheduler sch_global{ec};
        
            // --- PHASE 1: Deallocate old tensors and update bond dimension metadata ---
            std::cout << "[RANK " << rank << "] --- apply_collective_updates: PHASE 1 [Deallocation Planning] ---" << std::endl;
            std::set<IdxType> deallocated_sites;
            for (const auto& meta : all_metadata)
            {
                if (!meta.is_valid) continue;
        
                if (deallocated_sites.find(meta.q0) == deallocated_sites.end()) {
                    std::cout << "[RANK " << rank << "]   - Queuing deallocation for site " << meta.q0 << std::endl;
                    sch_global.deallocate(mps_tensors[meta.q0]);
                    deallocated_sites.insert(meta.q0);
                }
                if (deallocated_sites.find(meta.q1) == deallocated_sites.end()) {
                    std::cout << "[RANK " << rank << "]   - Queuing deallocation for site " << meta.q1 << std::endl;
                    sch_global.deallocate(mps_tensors[meta.q1]);
                    deallocated_sites.insert(meta.q1);
                }
        
                std::cout << "[RANK " << rank << "]   - Updating bond dimension for link " << meta.q0+1
                          << " to " << meta.new_bond_dim << std::endl;
                bond_dims[meta.q0 + 1] = meta.new_bond_dim;
                tamm::IndexSpace is_new_bond{tamm::range(meta.new_bond_dim)};
                bond_tis[meta.q0 + 1] = tamm::TiledIndexSpace(is_new_bond, block_size);
            }
        
            // --- PHASE 2: Allocate new tensors with the now-consistent dimensions ---
            std::cout << "[RANK " << rank << "] --- apply_collective_updates: PHASE 2 [Allocation Planning] ---" << std::endl;
            std::map<IdxType, tamm::Tensor<Cplx>> site_to_new_tensor;
            for(const auto& site : deallocated_sites) {
                site_to_new_tensor.emplace(
                    site,
                    tamm::Tensor<Cplx>{bond_tis[site], phys_tis[site], bond_tis[site + 1]}
                );
                site_to_new_tensor.at(site).set_dense();
                std::cout << "[RANK " << rank << "]   - Queuing allocation for new tensor at site " << site << std::endl;
                sch_global.allocate(site_to_new_tensor.at(site));
            }
        
            std::cout << "[RANK " << rank << "] apply_collective_updates: Executing collective deallocations and allocations..." << std::endl;
            auto start_res_mgmt = std::chrono::high_resolution_clock::now();
            sch_global.execute(exec_hw);
            auto end_res_mgmt = std::chrono::high_resolution_clock::now();
            total_resource_management_time += (end_res_mgmt - start_res_mgmt);
            std::cout << "[RANK " << rank << "] apply_collective_updates: Deallocations and allocations complete." << std::endl;
        
            // --- PHASE 3: Transfer data from compute ranks to new tensors ---
            std::cout << "[RANK " << rank << "] --- apply_collective_updates: PHASE 3 [Data Transfer] ---" << std::endl;
            std::cout << "[RANK " << rank << "]   - BARRIER before data puts." << std::endl;
            pg.barrier();
        
            int local_result_idx = 0;
            for (const auto& meta : all_metadata) {
                if (!meta.is_valid) continue;
        
                if (rank == meta.original_rank) {
                    auto& result_data = local_results[local_result_idx++];
                    assert(result_data.q0 == meta.q0 && result_data.q1 == meta.q1);
        
                    auto& new_T0_ref = site_to_new_tensor.at(meta.q0);
                    auto& new_T1_ref = site_to_new_tensor.at(meta.q1);
        
                    std::cout << "[RANK " << rank << "]   - I AM THE OWNER (" << meta.original_rank
                              << "). Putting data for qubits (" << meta.q0 << ", " << meta.q1 << ")." << std::endl;
                    
                    // --- DIAGNOSTIC: Print the data being transferred ---
                    print_buffer_diag("PUTTING", meta.q0, result_data.new_T0_data);
                    print_buffer_diag("PUTTING", meta.q1, result_data.new_T1_data);
        
                    tamm::span<Cplx> t0_span{result_data.new_T0_data};
                    tamm::span<Cplx> t1_span{result_data.new_T1_data};
        
                    new_T0_ref.put(*(new_T0_ref.loop_nest().begin()), t0_span);
                    new_T1_ref.put(*(new_T1_ref.loop_nest().begin()), t1_span);
                    
                    std::cout << "[RANK " << rank << "]   - Put issued for (" << meta.q0 << ", " << meta.q1 << ")." << std::endl;
                }
            }
        
            std::cout << "[RANK " << rank << "]   - BARRIER after data puts." << std::endl;
            pg.barrier();
        
            // --- PHASE 4: Update the main MPS state with the new tensors ---
            std::cout << "[RANK " << rank << "] --- apply_collective_updates: PHASE 4 [State Update] ---" << std::endl;
            for(auto const& [site, new_tensor] : site_to_new_tensor) {
                std::cout << "[RANK " << rank << "]   - Updating mps_tensors[" << site << "] with new tensor handle." << std::endl;
                mps_tensors[site] = new_tensor;
            }
        
            std::cout << "[RANK " << rank << "] << Exiting apply_collective_updates." << std::endl;
        }

        void C1_GATE_local_kernel(tamm::Tensor<Cplx>& target_tensor, IdxType q_idx, const std::array<Cplx, 4>& U)
        {
            int rank = pg.rank().value();
            std::cout << "[RANK " << rank << "] --> C1_GATE_local_kernel on qubit " << q_idx << "." << std::endl;
            
            auto start_res_mgmt = std::chrono::high_resolution_clock::now();
            tamm::ProcGroup self_pg = tamm::ProcGroup::create_self();
            tamm::ExecutionContext ec_local{self_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::local};
            tamm::Scheduler sch_local{ec_local};
        
            auto tis_l = target_tensor.tiled_index_spaces()[0];
            auto tis_p = target_tensor.tiled_index_spaces()[1];
            auto tis_r = target_tensor.tiled_index_spaces()[2];
        
            tamm::Tensor<Cplx> G_local({tis_p, tis_p});
            tamm::Tensor<Cplx> T_new_local({tis_l, tis_p, tis_r});
            tamm::Tensor<Cplx> T_in_local({tis_l, tis_p, tis_r});
            G_local.set_dense(); T_new_local.set_dense(); T_in_local.set_dense();
        
            sch_local.allocate(G_local, T_new_local, T_in_local).execute(exec_hw);
            auto end_res_mgmt = std::chrono::high_resolution_clock::now();
            total_resource_management_time += (end_res_mgmt - start_res_mgmt);
        
            // --- Fill Gate Matrix and print it ---
            auto fill_g = [&](const tamm::IndexVector& bid, tamm::span<Cplx> buf){
                auto offsets = G_local.block_offsets(bid);
                int pout = offsets[0], pin = offsets[1];
                buf[0] = U[pout * 2 + pin];
            };
            tamm::update_tensor(G_local, fill_g);
            std::cout << "[RANK " << rank << "] C1_GATE_local_kernel q=" << q_idx << ": Gate matrix U = [ (" 
                      << U[0].real() << "," << U[0].imag() << "), (" << U[1].real() << "," << U[1].imag() << "); ("
                      << U[2].real() << "," << U[2].imag() << "), (" << U[3].real() << "," << U[3].imag() << ") ]" << std::endl;
        
            // --- Get data from global tensor and print its diagnostics ---
            auto start_get = std::chrono::high_resolution_clock::now();
            std::vector<Cplx> t_in_buf(T_in_local.size());
            target_tensor.get(*(target_tensor.loop_nest().begin()), t_in_buf);
            T_in_local.put(*(T_in_local.loop_nest().begin()), t_in_buf);
            auto end_get = std::chrono::high_resolution_clock::now();
            total_data_movement_time += (end_get - start_get);
            
            print_buffer_diag("C1 INPUT TENSOR", q_idx, t_in_buf);
        
            // --- Perform Contraction ---
            auto start_contraction = std::chrono::high_resolution_clock::now();
            sch_local(T_new_local("l","p'","r") = G_local("p'","p") * T_in_local("l","p","r")).execute(exec_hw);
            auto end_contraction = std::chrono::high_resolution_clock::now();
            total_contraction_time += (end_contraction - start_contraction);
        
            // --- Get result back and print its diagnostics ---
            std::vector<Cplx> t_out_buf(T_new_local.size());
            T_new_local.get(*(T_new_local.loop_nest().begin()), t_out_buf);
            print_buffer_diag("C1 OUTPUT TENSOR", q_idx, t_out_buf);
        
            // --- Put result back to global tensor ---
            auto start_put = std::chrono::high_resolution_clock::now();
            target_tensor.put(*(target_tensor.loop_nest().begin()), t_out_buf);
            auto end_put = std::chrono::high_resolution_clock::now();
            total_data_movement_time += (end_put - start_put);
            std::cout << "[RANK " << rank << "] C1_GATE_local_kernel q=" << q_idx << ": Put back to global tensor complete." << std::endl;
        
            // --- Clean up ---
            start_res_mgmt = std::chrono::high_resolution_clock::now();
            sch_local.deallocate(G_local, T_new_local, T_in_local).execute(exec_hw);
            self_pg.destroy_coll();
            end_res_mgmt = std::chrono::high_resolution_clock::now();
            total_resource_management_time += (end_res_mgmt - start_res_mgmt);
            
            std::cout << "[RANK " << rank << "] <-- C1_GATE_local_kernel on qubit " << q_idx << " finished." << std::endl;
        }

        // This function now returns a struct containing the new data and metadata
        LocalGateResult C2_GATE_COMPUTE(const std::array<Cplx, 16> &U4, IdxType q0, IdxType q1)
        {
            int rank = pg.rank().value();
        
            // 1. Create a truly local execution context for this one-shot computation.
            tamm::ProcGroup self_pg = tamm::ProcGroup::create_self();
            tamm::ExecutionContext ec_local{self_pg, tamm::DistributionKind::dense, tamm::MemoryManagerKind::local};
            tamm::Scheduler sch_local{ec_local};
        
            // 2. Create LOCAL tensors for inputs and intermediates.
            tamm::Tensor<Cplx> T0_local({bond_tis[q0], phys_tis[q0], bond_tis[q0 + 1]});
            tamm::Tensor<Cplx> T1_local({bond_tis[q1], phys_tis[q1], bond_tis[q1 + 1]});
            T0_local.set_dense(); T1_local.set_dense();
            sch_local.allocate(T0_local, T1_local).execute(exec_hw);
        
            auto start_get = std::chrono::high_resolution_clock::now();
        
            // 3. GET data from the global mps_tensors into local std::vectors, then PUT to local tensors.
            std::vector<Cplx> t0_buf(T0_local.size());
            mps_tensors[q0].get(*(mps_tensors[q0].loop_nest().begin()), t0_buf);
            T0_local.put(*(T0_local.loop_nest().begin()), t0_buf);
        
            std::vector<Cplx> t1_buf(T1_local.size());
            mps_tensors[q1].get(*(mps_tensors[q1].loop_nest().begin()), t1_buf);
            T1_local.put(*(T1_local.loop_nest().begin()), t1_buf);
        
            auto end_get = std::chrono::high_resolution_clock::now();
            total_data_movement_time += (end_get - start_get);
        
            // 4. Perform local computations using the local scheduler.
            tamm::Tensor<Cplx> M_local({bond_tis[q0], phys_tis[q0], phys_tis[q1], bond_tis[q1 + 1]});
            tamm::Tensor<Cplx> G4_local({phys_tis[q0], phys_tis[q1], phys_tis[q0], phys_tis[q1]});
            tamm::Tensor<Cplx> M2_local({bond_tis[q0], phys_tis[q0], phys_tis[q1], bond_tis[q1 + 1]});
            M_local.set_dense(); G4_local.set_dense(); M2_local.set_dense();
            sch_local.allocate(M_local, G4_local, M2_local).execute(exec_hw);
        
            auto start_contraction = std::chrono::high_resolution_clock::now();
        
            sch_local(M_local("l","p0","p1","r") = T0_local("l","p0","b") * T1_local("b","p1","r")).execute(exec_hw);
            
            // *** MODIFIED SECTION: Populate G4_local using update_tensor for robustness. ***
            auto g4_filler =
              [&](const IndexVector& blockid_unused, tamm::span<Cplx> buff) {
                size_t c = 0;
                for (int p0p = 0; p0p < 2; ++p0p) {
                    for (int p1p = 0; p1p < 2; ++p1p) {
                        for (int p0_in = 0; p0_in < 2; ++p0_in) {
                            for (int p1_in = 0; p1_in < 2; ++p1_in, ++c) {
                                int row = p0p * 2 + p1p;
                                int col = p0_in * 2 + p1_in;
                                buff[c] = U4[row * 4 + col];
                            }
                        }
                    }
                }
            };
            tamm::update_tensor(G4_local, g4_filler);
        
            // *** DIAGNOSTIC: Print the contents of the gate tensor ***
            print_4_index_tensor(G4_local, "G4_local", q0, q1);
            
            sch_local(M2_local("l","p0p","p1p","r") = G4_local("p0p","p1p","p0","p1") * M_local("l","p0","p1","r")).execute(exec_hw);
        
            // *** DIAGNOSTIC: Print the contents of the M2 tensor AFTER contraction ***
            print_4_index_tensor(M2_local, "M2_local", q0, q1);
        
            auto end_contraction = std::chrono::high_resolution_clock::now();
            total_contraction_time += (end_contraction - start_contraction);
        
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
            result.original_rank = rank;
        
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

        IdxType local_svd_and_reconstruct_data(
            tamm::Tensor<Cplx>& M2_local,
            std::vector<Cplx>& Ti_new_data,
            std::vector<Cplx>& Tj_new_data,
            IdxType q0, IdxType q1)
        {
            int rank = pg.rank().value();
            //std::cout << "[RANK " << rank << "] ---> local_svd_and_reconstruct_data (EIGEN): Entered for qubits (" << q0 << ", " << q1 << ")." << std::endl;
        
            // 1. Extract dimensions from the input tensor
            const IdxType phys_dim = 2;
            IdxType Dl = M2_local.tiled_index_spaces()[0].index_space().num_indices();
            IdxType Dr = M2_local.tiled_index_spaces()[3].index_space().num_indices();
        
            Eigen::Index m = Dl * phys_dim;
            Eigen::Index n = phys_dim * Dr;
        
            // 2. Reshape the row-major TAMM tensor data into a column-major Eigen matrix.
            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> mat(m, n);
            
            std::vector<Cplx> M2_hostbuf(M2_local.size());
            M2_local.get(*(M2_local.loop_nest().begin()), M2_hostbuf);
        
            size_t c = 0;
            for (size_t l = 0; l < Dl; ++l) {
                for (size_t p0 = 0; p0 < phys_dim; ++p0) {
                    for (size_t p1 = 0; p1 < phys_dim; ++p1) {
                        for (size_t r = 0; r < Dr; ++r, ++c) {
                            // Eigen's operator() handles the column-major layout automatically.
                            mat(l * phys_dim + p0, p1 * Dr + r) = M2_hostbuf[c];
                        }
                    }
                }
            }

//std::cout << "[RANK " << rank << "] local_svd_and_reconstruct_data (EIGEN): Reshape to Eigen matrix complete." << std::endl;
            
            // 3. Compute the SVD using Eigen's robust BDCSVD.
            Eigen::BDCSVD<decltype(mat)> svd(mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
            auto svals = svd.singularValues();
            //std::cout << "[RANK " << rank << "] local_svd_and_reconstruct_data (EIGEN): SVD computation complete." << std::endl;
        
            // 4. Truncate based on singular value cutoff and max bond dimension.
            std::vector<IdxType> keep;
            keep.reserve(svals.size());
            for (IdxType i = 0; i < svals.size(); ++i) {
                if (std::abs(svals(i)) >= sv_cutoff) {
                    keep.push_back(i);
                }
            }
            
            IdxType chi = std::min<IdxType>(max_bond_dim, IdxType(keep.size()));
            if (chi == 0 && svals.size() > 0) {
                chi = 1; // Prevent bond dimension from ever becoming zero.
            }

            std::cout << "\n[PARALLEL SVD DIAG RANK " << rank << "] Qubits (" << q0 << ", " << q1 
                      << "), chi=" << chi << std::endl;
            
            // Print singular values
            std::cout << "  Singular values: ";
            for (IdxType k = 0; k < chi; ++k) {
                std::cout << svals(keep[k]) << " ";
            }
            std::cout << std::endl;
            
            // Extract and print first few elements of Umat and Vh
            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Umat_dbg(mat.rows(), chi);
            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Vh_dbg(chi, mat.cols());
            for (IdxType k = 0; k < chi; ++k) {
                IdxType i = keep[k];
                Umat_dbg.col(k) = svd.matrixU().col(i);
                Vh_dbg.row(k)   = svd.matrixV().col(i).adjoint();
            }
            
            std::cout << "  Umat (first 4): ";
            for(int i=0; i < std::min((long)4, Umat_dbg.size()); ++i) {
                std::cout << "(" << Umat_dbg.data()[i].real() << "," << Umat_dbg.data()[i].imag() << ") ";
            }
            std::cout << std::endl;
            
            std::cout << "  Vh (first 4): ";
            for(int i=0; i < std::min((long)4, Vh_dbg.size()); ++i) {
                std::cout << "(" << Vh_dbg.data()[i].real() << "," << Vh_dbg.data()[i].imag() << ") ";
            }
            std::cout << std::endl << std::endl;

            // 5. Extract the truncated U, S, and Vh matrices.
            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Umat(mat.rows(), chi);
            Eigen::Matrix<Cplx, Eigen::Dynamic, 1> kept_svals(chi);
            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> Vh(chi, mat.cols());
            for (IdxType k = 0; k < chi; ++k) {
                IdxType i = keep[k];
                Umat.col(k)   = svd.matrixU().col(i);
                kept_svals(k) = svals(i);
                Vh.row(k)     = svd.matrixV().col(i).adjoint();
            }
            
            // 6. Populate the output vectors with the data for the new tensors.
            
            // 6a. Populate the new left tensor data (from Umat)
            Ti_new_data.resize(Dl * phys_dim * chi);
            c = 0;
            for (size_t l = 0; l < Dl; ++l) {
                for (size_t p0 = 0; p0 < phys_dim; ++p0) {
                    for (size_t b = 0; b < chi; ++b, ++c) {
                        Ti_new_data[c] = Umat(l * phys_dim + p0, b);
                    }
                }
            }
        
            // 6b. Populate the new right tensor data (from S * Vh)
            Eigen::Matrix<Cplx, Eigen::Dynamic, Eigen::Dynamic> SV = kept_svals.asDiagonal() * Vh;
            Tj_new_data.resize(chi * phys_dim * Dr);
            c = 0;
            for (size_t b = 0; b < chi; ++b) {
                for (size_t p1 = 0; p1 < phys_dim; ++p1) {
                    for (size_t r = 0; r < Dr; ++r, ++c) {
                        Tj_new_data[c] = SV(b, p1 * Dr + r);
                    }
                }
            }
        
            //std::cout << "[RANK " << rank << "] <--- local_svd_and_reconstruct_data (EIGEN): Exiting." << std::endl;
            return chi;
        }


//        IdxType local_svd_and_reconstruct_data(
//            tamm::Tensor<Cplx>& M2_local,
//            std::vector<Cplx>& Ti_new_data,
//            std::vector<Cplx>& Tj_new_data,
//            IdxType q0, IdxType q1)
//        {
//            int rank = pg.rank().value();
//            //std::cout << "[RANK " << rank << "] ---> local_svd_and_reconstruct_data: Entered for qubits (" << q0 << ", " << q1 << ")." << std::endl;
//        
//            const IdxType phys_dim = 2;
//            IdxType Dl = M2_local.tiled_index_spaces()[0].index_space().num_indices();
//            IdxType Dr = M2_local.tiled_index_spaces()[3].index_space().num_indices();
//        
//            int m = Dl * phys_dim;
//            int n = phys_dim * Dr;
//        
//            // Reshape the row-major TAMM tensor data into a column-major matrix for cuSOLVER.
//            std::vector<Cplx> M2_col_major(m * n);
//            std::vector<Cplx> M2_hostbuf(M2_local.size());
//            M2_local.get(*(M2_local.loop_nest().begin()), M2_hostbuf);
//        
//            size_t c = 0;
//            for (size_t l = 0; l < Dl; ++l)
//            for (size_t p0 = 0; p0 < phys_dim; ++p0)
//            for (size_t p1 = 0; p1 < phys_dim; ++p1)
//            for (size_t r = 0; r < Dr; ++r, ++c)
//            {
//                size_t row = l * phys_dim + p0;
//                size_t col = p1 * Dr + r;
//                M2_col_major[row + col * m] = M2_hostbuf[c];
//            }
//            //std::cout << "[RANK " << rank << "] local_svd_and_reconstruct_data: Reshape complete." << std::endl;
//        
//            // --- Start Timing SVD ---
//            auto start_svd = std::chrono::high_resolution_clock::now();
//
//            // Perform the SVD on the GPU.
//            std::vector<double> S;
//            std::vector<Cplx> U_row, VT_row;
//            gpu_svd_jacobi(M2_col_major.data(), m, n, S, U_row, VT_row);
//
//            auto end_svd = std::chrono::high_resolution_clock::now();
//            // --- End Timing SVD ---
//            
//            // Accumulate the time for this SVD operation
//            total_svd_time += (end_svd - start_svd);
//
//            // Truncate based on singular value cutoff and max bond dimension.
//            std::vector<IdxType> keep;
//            keep.reserve(S.size());
//            for (size_t i = 0; i < S.size(); ++i) {
//                if (S[i] >= sv_cutoff) {
//                    keep.push_back(i);
//                }
//            }
//            IdxType chi = std::min<IdxType>(max_bond_dim, IdxType(keep.size()));
//            if (chi == 0) {
//                chi = 1; // Prevent bond dimension from becoming zero.
//            }
//            //std::cout << "[RANK " << rank << "] local_svd_and_reconstruct_data: Truncation complete. New bond dimension (chi): " << chi << "." << std::endl;
//        
//            // Populate the output vectors with the data for the new tensors.
//            Ti_new_data.resize(Dl * phys_dim * chi);
//            c = 0;
//            for (size_t l = 0; l < Dl; ++l)
//            for (size_t p0 = 0; p0 < phys_dim; ++p0)
//            for (size_t b = 0; b < chi; ++b, ++c)
//            {
//                Ti_new_data[c] = U_row[(l * phys_dim + p0) * S.size() + keep[b]];
//            }
//        
//            Tj_new_data.resize(chi * phys_dim * Dr);
//            c = 0;
//            for (size_t b = 0; b < chi; ++b)
//            for (size_t p1 = 0; p1 < phys_dim; ++p1)
//            for (size_t r = 0; r < Dr; ++r, ++c)
//            {
//                Tj_new_data[c] = Cplx(S[keep[b]], 0.0) * VT_row[keep[b] * n + (p1 * Dr + r)];
//            }
//        
//            //std::cout << "[RANK " << rank << "] <--- local_svd_and_reconstruct_data: Exiting." << std::endl;
//            return chi;
//        }

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
