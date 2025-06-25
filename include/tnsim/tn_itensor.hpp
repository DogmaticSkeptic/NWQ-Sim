#pragma once

#include "../state.hpp"

#include "../nwq_util.hpp"
#include "../gate.hpp"
#include "../circuit.hpp"
#include "../config.hpp"
#include "../private/exp_gate_declarations_host.hpp"

#include "../circuit_pass/fusion.hpp"
#include "../private/macros.hpp"
#include "../private/sim_gate.hpp"

#include <random>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <chrono> // Added for std::chrono

#include "tensor.h"

namespace NWQSim
{
    class TN_ITENSOR : public QuantumState
    {

    public:
        TN_ITENSOR(IdxType _n_qubits, IdxType max_dim, double sv_cutoff) : QuantumState(SimType::TN)
        {
            // Initialize CPU side
            n_qubits = _n_qubits;
            n_cpu = 1;

    	    //temporary cpu_mem
    	    cpu_mem = 0.0;

            // MPS Parameters
            MaxDim = int(max_dim);
            Cutoff = sv_cutoff;

            rng.seed(Config::RANDOM_SEED);

            // ITensor MPS Initialization
    	    auto sites = itensor::SpinHalf(int(n_qubits),{"ConserveQNs=", false});
    	    auto state = itensor::InitState(sites,"Up");
    	    auto network = itensor::MPS(state);

        }


        ~TN_ITENSOR()
        {
            // // Release for CPU side
        }

        void reset_state() override
        {
            // // Reset CPU input & output
	    
            // MPS initial state |00..0>
            sites = itensor::SpinHalf(int(n_qubits),{"ConserveQNs=", false});
            auto state = itensor::InitState(sites,"Up");
            network = itensor::MPS(state);
            network.position(1);
        }

        void set_seed(IdxType seed) override
        {
            rng.seed(seed);
        }

        virtual void set_initial(std::string fpath, std::string format) override
        {
            throw std::runtime_error("Not implemented");
        }

        virtual void dump_res_state(std::string outpath) override
        {
            throw std::runtime_error("Not implemented");
        };

        void sim(std::shared_ptr<NWQSim::Circuit> circuit) override
        {
            IdxType origional_gates = circuit->num_gates();
            std::vector<SVGate> gates = fuse_circuit_sv(circuit);
            IdxType n_gates = gates.size();
            assert(circuit->num_qubits() == n_qubits);
            
            auto sim_start = std::chrono::high_resolution_clock::now();
	    
            // Set Gauge of MPS to right-canonical
            network.position(1);
         
            simulation_kernel(gates);

            auto sim_end = std::chrono::high_resolution_clock::now();
            double elap_t = std::chrono::duration_cast<std::chrono::duration<double>>(sim_end - sim_start).count();

            if (Config::PRINT_SIM_TRACE)
            {
                std::cout<<"Total Simulation Time:"<<elap_t<<"\n";
                std::cout<<"Total Time - Tallied Time:"<<elap_t - 
                    (total_c2_merge + total_allocdealloc + total_gate_c1_set +
                     total_gate_c2_set + total_c1_exec + total_c2_exec +
                     total_svd_time + total_c2_tensor_set + total_c1_tensor_set)<<"\n";
                std::cout<<"Run Time Statistics:";
                std::cout<<"Total C1 Gates: "<<total_c1_gate<<"\n";
                std::cout<<"Total C2 Gates (local): "<<total_c2_gate_l<<"\n";
                std::cout<<"Total C2 Gates (non-local): "<<total_c2_gate_nl<<"\n";

                std::cout<<"Total Merge Execution Time: "<<total_c2_merge<<"\n";
                std::cout<<"Total Allocation/Deallocation Time: "<<total_allocdealloc<<"\n";
                std::cout<<"Total C1 Gate Set Time: "<<total_gate_c1_set<<"\n";
                std::cout<<"Total C2 Gate Set Time: "<<total_gate_c2_set<<"\n";
                std::cout<<"Total C1 Gate Execution Time: "<<total_c1_exec<<"\n";
                std::cout<<"Total C2 Gate Execution Time: "<<total_c2_exec<<"\n";
                std::cout<<"Total SVD Time: "<<total_svd_time<<"\n";
                std::cout<<"Total C2 Tensor Set: "<<total_c2_tensor_set<<"\n";
                std::cout<<"Total C2 Non-Local Time: "<<total_c2_nl_time<<"\n";

                std::cout<<"Avg Merge Execution Time: "<<(total_c2_gate_l > 0 ? (total_c2_merge / total_c2_gate_l) : 0.0)<<"\n";
                std::cout<<"Avg Allocation/Deallocation Time: "<<((total_c1_gate + total_c2_gate_l + total_c2_gate_nl) > 0 ? (total_allocdealloc / (total_c1_gate + total_c2_gate_l + total_c2_gate_nl)) : 0.0)<<"\n";
                std::cout<<"Avg C1 Gate Set Time: "<<(total_c1_gate > 0 ? (total_gate_c1_set / total_c1_gate) : 0.0)<<"\n";
                std::cout<<"Avg C2 Gate Set Time: "<<(total_c2_gate_l > 0 ? (total_gate_c2_set / total_c2_gate_l) : 0.0)<<"\n";
                std::cout<<"Avg C1 Gate Execution Time: "<<(total_c1_gate > 0 ? (total_c1_exec / total_c1_gate) : 0.0)<<"\n";
                std::cout<<"Avg C2 Gate Execution Time: "<<(total_c2_gate_l > 0 ? (total_c2_exec / total_c2_gate_l) : 0.0)<<"\n";
                std::cout<<"Avg SVD Time: "<<(total_c2_gate_l > 0 ? (total_svd_time / total_c2_gate_l) : 0.0)<<"\n";
                std::cout<<"Avg C2 Tensor Set: "<<(total_c2_gate_l > 0 ? (total_c2_tensor_set / total_c2_gate_l) : 0.0)<<"\n";
                std::cout<<"Avg C2 Non-Local Time: "<<(total_c2_gate_nl > 0 ? (total_c2_nl_time / total_c2_gate_nl) : 0.0)<<"\n";

                std::cout<<"Percentage c2_merge of total time: "
                         << (total_c2_merge / elap_t) * 100.0 << "%\n";
                std::cout<<"Percentage alloc/dealloc of total time: "
                         << (total_allocdealloc / elap_t) * 100.0 << "%\n";
                std::cout<<"Percentage c1_gate_set of total time: "
                         << (total_gate_c1_set / elap_t) * 100.0 << "%\n";
                std::cout<<"Percentage c2_gate_set of total time: "
                         << (total_gate_c2_set / elap_t) * 100.0 << "%\n";
                std::cout<<"Percentage c1_exec of total time: "
                         << (total_c1_exec / elap_t) * 100.0 << "%\n";
                std::cout<<"Percentage c2_exec of total time: "
                         << (total_c2_exec / elap_t) * 100.0 << "%\n";
                std::cout<<"Percentage svd_time of total time: "
                         << (total_svd_time / elap_t) * 100.0 << "%\n";
                std::cout<<"Percentage c2_tensor_set of total time: "
                         << (total_c2_tensor_set / elap_t) * 100.0 << "%\n";
                std::cout<<"Percentage c1_tensor_set of total time: "
                         << (total_c1_tensor_set / elap_t) * 100.0 << "%\n";

                std::cout<<"Percetanges added up: "
                         << (total_c2_merge + total_allocdealloc + total_gate_c1_set +
                             total_gate_c2_set + total_c1_exec + total_c2_exec +
                             total_svd_time + total_c2_tensor_set + total_c1_tensor_set) / elap_t * 100.0
                         << "%\n";
            }
        }

        IdxType *get_results() override
        {
            return results;
        }

        IdxType measure(IdxType qubit) override
        {
            throw std::runtime_error("Not implemented");
            //M_GATE(qubit);
            //return results[0];
        }

        IdxType *measure_all(IdxType repetition) override
        {
            MA_GATE(repetition);
            return results;
        }

        virtual ValType get_exp_z(const std::vector<size_t> &in_bits) override
        {
            throw std::runtime_error("Not implemented");
        }

        virtual ValType get_exp_z() override
        {
            throw std::runtime_error("Not implemented");
        }

        void print_res_state() override
        {
            throw std::runtime_error("Not implemented");
        }

    protected:
        // n_qubits is the number of qubits
        IdxType n_qubits;
        IdxType n_cpu;

        // MPS Params
        int MaxDim;
        double Cutoff;

        // MPS Objects
        itensor::SpinHalf sites;
        itensor::MPS network;

        IdxType *results = NULL;

        // Random
        std::mt19937 rng;
        std::uniform_real_distribution<ValType> uni_dist;

        // CPU memory usage
        ValType cpu_mem;

        // execution time statistics
        double total_allocdealloc = 0;
        double total_gate_c1_set = 0;
        double total_c1_exec = 0;
        double total_c1_tensor_set = 0;

        double total_gate_c2_set = 0;
        double total_c2_merge = 0;
        double total_c2_exec = 0;
        double total_svd_time = 0;
        double total_c2_tensor_set = 0;

        double total_c2_nl_time = 0;

        int total_c1_gate = 0;
        int total_c2_gate_l = 0;
        int total_c2_gate_nl = 0;

        virtual void simulation_kernel(const std::vector<SVGate> &gates)
        {
            int n_gates = gates.size();
            for (int i = 0; i < n_gates; i++)
            {
                auto g = gates[i];

                if (g.op_name == OP::C1)
                {
                    total_c1_gate++;
                    C1_GATE(g.gm_real, g.gm_imag, g.qubit);
                }
                else if (g.op_name == OP::C2)
                {
                    C2_GATE(g.gm_real, g.gm_imag, g.ctrl, g.qubit);
                }
                else if (g.op_name == OP::RESET)
                {
                    RESET_GATE(g.qubit);
                }
                else if (g.op_name == OP::M)
                {
                    M_GATE(g.qubit);
                }
                else if (g.op_name == OP::MA)
                {
                    MA_GATE(g.qubit);
                }
                else if (g.op_name == OP::EXPECT)
                {
                    ObservableList *o = (ObservableList *)(g.data);
                    EXPECT_GATE(o);
                }
                else
                {
                    std::cout << "Unrecognized gates" << std::endl
                              << OP_NAMES[g.op_name] << std::endl;
                    std::logic_error("Invalid gate type");
                }

// #ifdef PURITY_CHECK
//                 Purity_Check(g, i);
// #endif
            }
            if (Config::PRINT_SIM_TRACE)
            {
                std::cout << std::endl;
            }
        }

        //============== C1 Gate ================
        // Arbitrary 1-qubit gate
        virtual void C1_GATE(const ValType *gm_real, const ValType *gm_imag, const IdxType qubit)
        {
            // iTensor is 1 indexed
            int site = qubit + 1;

            //Initialize empty C1 Gate tensor
            auto alloc_start = std::chrono::high_resolution_clock::now();
            auto j = sites(site);
            auto gate = itensor::ITensor(prime(j),j);
            auto alloc_end = std::chrono::high_resolution_clock::now();
            total_allocdealloc += std::chrono::duration_cast<std::chrono::duration<double>>(alloc_end - alloc_start).count();

            // Set values of C1 gate tensor
            auto c1_gate_set = std::chrono::high_resolution_clock::now();
            gate.set(1,1,std::complex<double>(gm_real[0],gm_imag[0]));
            gate.set(1,2,std::complex<double>(gm_real[1],gm_imag[1]));
            gate.set(2,1,std::complex<double>(gm_real[2],gm_imag[2]));
            gate.set(2,2,std::complex<double>(gm_real[3],gm_imag[3]));
            auto c1_gate_set_end = std::chrono::high_resolution_clock::now();
            total_gate_c1_set += std::chrono::duration_cast<std::chrono::duration<double>>(c1_gate_set_end - c1_gate_set).count();

            // Move center of orthongality to qubit that will be contracted with gate
            if(isOrtho(network)){
                if(orthoCenter(network) != site){
                    network.position(site);
                }
            }
            else{
                network.position(site);
            }
         
            //Contract the 1-qubit gate with the MPS site at the qubit
            auto c1_exec_start = std::chrono::high_resolution_clock::now();
            auto temp = gate * network(site);
            auto c1_exec_end = std::chrono::high_resolution_clock::now();
            total_c1_exec += std::chrono::duration_cast<std::chrono::duration<double>>(c1_exec_end - c1_exec_start).count();
          
            temp.noPrime();
  
            
            // Set the MPS site to new values
            auto c1_tensor_set_start = std::chrono::high_resolution_clock::now();
            network.set(site,temp);
            auto c1_tensor_set_end = std::chrono::high_resolution_clock::now();
            total_c1_tensor_set += std::chrono::duration_cast<std::chrono::duration<double>>(c1_tensor_set_end - c1_tensor_set_start).count();

            // // For safety re-orthogonalize the network
            // network.orthogonalize();
        }

        //============== C2 Gate ================
        // Arbitrary 2-qubit gate
        virtual void C2_GATE(const ValType *gm_real, const ValType *gm_imag,
                             const IdxType qubit0, const IdxType qubit1)
        {
            assert(qubit0 != qubit1); // Non-cloning

            // iTensor is 1 indexed
            int site0 = qubit0 + 1;
            int site1 = qubit1 + 1;

            // Get iTensor index "names" of control and target qubit
            auto i = sites(site0);
            auto j = sites(site1);

            // Initialize 2-qubit gate tensor with correct indices
            //     (i')  (j')
            //      |     |   
            //    -----------
            //    | 2Q Gate |
            //    -----------
            //      |     |
            //     (i)   (j)

            auto gate_alloc_start = std::chrono::high_resolution_clock::now();
            auto gate = itensor::ITensor(dag(i),dag(j),prime(i),prime(j));
            auto gate_alloc_end = std::chrono::high_resolution_clock::now();
            total_allocdealloc += std::chrono::duration_cast<std::chrono::duration<double>>(gate_alloc_end - gate_alloc_start).count();

            // Set values of C2 gate tensor
            auto gate_set_start = std::chrono::high_resolution_clock::now();
            gate.set(1,1,1,1,std::complex<double>(gm_real[0],gm_imag[0]));
            gate.set(1,2,1,1,std::complex<double>(gm_real[1],gm_imag[1]));
            gate.set(2,1,1,1,std::complex<double>(gm_real[2],gm_imag[2]));
            gate.set(2,2,1,1,std::complex<double>(gm_real[3],gm_imag[3]));
                                                                        
            gate.set(1,1,1,2,std::complex<double>(gm_real[4],gm_imag[4]));
            gate.set(1,2,1,2,std::complex<double>(gm_real[5],gm_imag[5]));
            gate.set(2,1,1,2,std::complex<double>(gm_real[6],gm_imag[6]));
            gate.set(2,2,1,2,std::complex<double>(gm_real[7],gm_imag[7]));
                                                                    
            gate.set(1,1,2,1,std::complex<double>(gm_real[8],gm_imag[8]));
            gate.set(1,2,2,1,std::complex<double>(gm_real[9],gm_imag[9]));
            gate.set(2,1,2,1,std::complex<double>(gm_real[10],gm_imag[10]));
            gate.set(2,2,2,1,std::complex<double>(gm_real[11],gm_imag[11]));
                                                                        
            gate.set(1,1,2,2,std::complex<double>(gm_real[12],gm_imag[12]));
            gate.set(1,2,2,2,std::complex<double>(gm_real[13],gm_imag[13]));
            gate.set(2,1,2,2,std::complex<double>(gm_real[14],gm_imag[14]));
            gate.set(2,2,2,2,std::complex<double>(gm_real[15],gm_imag[15]));
            auto gate_set_end = std::chrono::high_resolution_clock::now();
            total_gate_c2_set += std::chrono::duration_cast<std::chrono::duration<double>>(gate_set_end - gate_set_start).count();

            //
            // Non-adjacent / Non-local 2-qubit gates
            //
            // qubit0 must be < then qubit1  (control must be to the left (smaller site number) of target)
            //
	        // method = true is Bond Propagation (fastest)
	        // method = false is MPO
            auto method = true;
            if(std::abs(qubit0 - qubit1) != 1){
                total_c2_gate_nl++; // Increment non-local counter
                //std::cout<<"Non local C2"<<std::endl;
                auto nl_start_time = std::chrono::high_resolution_clock::now();

                // 2Q Gate Decomposition into Control: u  ;  Target: s*v
                auto svd_start = std::chrono::high_resolution_clock::now();
                auto [u,s,v] = itensor::svd(gate,{i,prime(i)},{j,prime(j)},{"Cutoff=", Cutoff, "MaxDim=", MaxDim, "SVDMethod=", "gesdd"});
                auto svd_end = std::chrono::high_resolution_clock::now();
                total_svd_time += std::chrono::duration_cast<std::chrono::duration<double>>(svd_end - svd_start).count();
                auto sv = s*v;

                if(method){
                    // Sequential Contraction Method based on Quantinuum pytket
                    itensor::ITensor propagating_bond;
                    itensor::Index lindex;

                    // Move center of orthogonality
                    if(isOrtho(network)){
                        if(orthoCenter(network) != site0){
                            network.position(site0);
                        }
                    }
                    else{
                        network.position(site0);
                    }

      
                    // Initial contraction of control "gate" and control qubit
                    auto c2_exec_start = std::chrono::high_resolution_clock::now();
                    auto site0_contract = u*network(site0);
                    auto c2_exec_end = std::chrono::high_resolution_clock::now();
                    total_c2_exec += std::chrono::duration_cast<std::chrono::duration<double>>(c2_exec_end - c2_exec_start).count();

                    // Decompose contraction, U is the new site tensor in the circuit network
                    // S*V holds the "propagating bond" which is "pushed" through the circuit to the target site
                    // This "bond" is the dangling link of the initial gate SVD
                    svd_start = std::chrono::high_resolution_clock::now();
                    auto [U,S,V] = itensor::svd(site0_contract,{i,prime(i),leftLinkIndex(network,site0)},{"Cutoff=", Cutoff, "MaxDim=", MaxDim, "SVDMethod=", "gesdd"});
                    svd_end = std::chrono::high_resolution_clock::now();
                    total_svd_time += std::chrono::duration_cast<std::chrono::duration<double>>(svd_end - svd_start).count();

                    auto c2_tensor_set_start = std::chrono::high_resolution_clock::now();
                    network.set(site0,U);
                    network.position(site0);
                    propagating_bond =S*V;
                    auto c2_tensor_set_end = std::chrono::high_resolution_clock::now();
                    total_c2_tensor_set += std::chrono::duration_cast<std::chrono::duration<double>>(c2_tensor_set_end - c2_tensor_set_start).count();
                
                    // Track link index to insure correct indices are contracted at each step
                    lindex = commonInds(U,S)[0];

            
                    // Repeat propagation process through the intermediate sites
                    for(auto k = site0+1 ; k < site1; k++ ){

                        c2_exec_start = std::chrono::high_resolution_clock::now();
                        auto k_contract = propagating_bond * network(k);
                        c2_exec_end = std::chrono::high_resolution_clock::now();
                        total_c2_exec += std::chrono::duration_cast<std::chrono::duration<double>>(c2_exec_end - c2_exec_start).count();

                        svd_start = std::chrono::high_resolution_clock::now();
                        auto [u_prop,s_prop,v_prop] = itensor::svd(k_contract,{sites(k),lindex},{"Cutoff=", Cutoff, "MaxDim=", MaxDim, "SVDMethod=", "gesdd"});
                        svd_end = std::chrono::high_resolution_clock::now();
                        total_svd_time += std::chrono::duration_cast<std::chrono::duration<double>>(svd_end - svd_start).count();

                        c2_tensor_set_start = std::chrono::high_resolution_clock::now();
                        lindex = commonInds(u_prop,s_prop)[0];
                        network.set(k,u_prop);
                        // network.position(k);
                        propagating_bond = s_prop*v_prop;
                        c2_tensor_set_end = std::chrono::high_resolution_clock::now();
                        total_c2_tensor_set += std::chrono::duration_cast<std::chrono::duration<double>>(c2_tensor_set_end - c2_tensor_set_start).count();

                    }
   
                    // Final contraction with Target qubit, propagating bond is absorbed in U,Link on sv
                    c2_exec_start = std::chrono::high_resolution_clock::now();
                    auto site1_contract = sv*propagating_bond*network(site1);
                    c2_exec_end = std::chrono::high_resolution_clock::now();
                    total_c2_exec += std::chrono::duration_cast<std::chrono::duration<double>>(c2_exec_end - c2_exec_start).count();
    
                    // Target site is set to new values
                    c2_tensor_set_start = std::chrono::high_resolution_clock::now();
                    network.set(site1,site1_contract);
                    auto c2_tensor_set_end = std::chrono::high_resolution_clock::now();
                    total_c2_tensor_set += std::chrono::duration_cast<std::chrono::duration<double>>(c2_tensor_set_end - c2_tensor_set_start).count();

                    // network clean-up is performed after else statement
            
                }
                else{
                    // MPO Method
                    //
                    itensor::Index lright;
                    itensor::Index lleft;
                    network.position(1);

                    auto gate_MPO = itensor::MPO(n_qubits);
                    auto lusv = commonInds(u,s)[0];

                    for (auto k = 1; k<=n_qubits;k++){
                        if(not((k == site0) || (k == site1))){
                            if(k==1){
                                lright = itensor::Index(4,"Link");
                                auto diag1 = itensor::delta(sites(k),prime(sites(k)));
                                auto diag2 = itensor::delta(lright);
                                auto diag3 = toDense(diag2);
                                auto temp = diag1 * diag3;
                                gate_MPO.set(k,temp);
                            }
                            else if( k==n_qubits){
                                lleft = lright;
                                auto diag1 = itensor::delta(sites(k),prime(sites(k)));
                                auto diag2 = itensor::delta(lleft);
                                auto diag3 = toDense(diag2);
                                auto temp = diag1 * diag3;
                                gate_MPO.set(k,temp);
                            }
                            else{
                                lleft = lright;
                                lright = itensor::Index(4,"Link");
                                auto diag1 = itensor::delta(sites(k),prime(sites(k)));
                                auto diag2 = itensor::delta(lleft,lright);
                                auto diag3 = toDense(diag2);
                                auto temp = diag1 * diag3;
                                gate_MPO.set(k,temp);
                            }
                        }
                        else if(k==site0){
                            if(not(k==1 or k==n_qubits)){
                                lleft = lright;
                                lright = lusv;
                                gate_MPO.set(site0, u*itensor::delta(lleft));
                            }
                            else{
                                lright = lusv;
                                gate_MPO.set(site0, u);
                            }
                        }
                        else if(k==site1){
                            if(not(k==1 or k==n_qubits)){
                                lleft = lright;
                                auto lsv = itensor::delta(lleft,lusv);
                                lright = itensor::Index(4,"Link");
                                gate_MPO.set(site1, s*v*lsv*itensor::delta(lright));
                            }
                            else{
    
                                lleft = lright;
                                auto lsv = itensor::delta(lleft,lusv);
                                gate_MPO.set(site1, s*v*lsv);
                            }
                        }
                    }

                    c2_exec_start = std::chrono::high_resolution_clock::now();
                    network = applyMPO(gate_MPO,network);
                    network.normalize();
                    c2_exec_end = std::chrono::high_resolution_clock::now();
                    total_c2_exec += std::chrono::duration_cast<std::chrono::duration<double>>(c2_exec_end - c2_exec_start).count();
                }

                // network clean-up
                auto nl_end_time = std::chrono::high_resolution_clock::now();
                network.noPrime();
                // network.orthogonalize();
                total_c2_nl_time += std::chrono::duration_cast<std::chrono::duration<double>>(nl_end_time - nl_start_time).count();
            }
            else{
                total_c2_gate_l++; // Increment local counter
                // Local 2-qubit gate method
                //
                //std::cout<<"Local C2"<<std::endl;
                // Move orthoganility center to control qubit
                if(isOrtho(network)){
                    if(orthoCenter(network) != site0){
                        network.position(site0);
                    }
                }
                else{
                    network.position(site0);
                }
                

                //     |    |
                //    --------
                //    | gate |
                //    --------
                //     |    |
                //     O -- O
                //
                //
                auto merge_start = std::chrono::high_resolution_clock::now();
                auto contract_location = network(site0)*network(site1);
                auto merge_end = std::chrono::high_resolution_clock::now();
                total_c2_merge += std::chrono::duration_cast<std::chrono::duration<double>>(merge_end - merge_start).count();

                //     |    |
                //    --------
                //    | gate |
                //    --------
                //     |    |
                //    --------
                //   | sites  |
                //    --------
                auto c2_exec_start = std::chrono::high_resolution_clock::now();
                auto new_sites_contracted = gate*contract_location;
                auto c2_exec_end = std::chrono::high_resolution_clock::now();
                total_c2_exec += std::chrono::duration_cast<std::chrono::duration<double>>(c2_exec_end - c2_exec_start).count();
                //
                //      |       |
                //    -------------
                //    | sites*gate |
                //    -------------
  
                new_sites_contracted.noPrime();
                
                auto svd_start = std::chrono::high_resolution_clock::now();
                auto [u,s,v] = itensor::svd(new_sites_contracted ,itensor::inds(network(site0)),{"Cutoff=", Cutoff, "MaxDim=", MaxDim, "SVDMethod=", "gesdd"});    
                auto svd_end = std::chrono::high_resolution_clock::now();
                total_svd_time += std::chrono::duration_cast<std::chrono::duration<double>>(svd_end - svd_start).count();

                auto c2_tensor_set_start = std::chrono::high_resolution_clock::now();
                network.set(site0, u);
                network.set(site1, s*v);
                auto c2_tensor_set_end = std::chrono::high_resolution_clock::now();
                total_c2_tensor_set += std::chrono::duration_cast<std::chrono::duration<double>>(c2_tensor_set_end - c2_tensor_set_start).count();

                //    |      |
                //   (u) - (s*v)
                //
                // network.orthogonalize();
            }
        }

        //============== C4 Gate ================
        // Arbitrary 4-qubit gate
        virtual void C4_GATE(const ValType *gm_real, const ValType *gm_imag,
                             const IdxType qubit0, const IdxType qubit1,
                             const IdxType qubit2, const IdxType qubit3)
        {
            throw std::runtime_error("Not implemented");
        }

        virtual void M_GATE(const IdxType qubit)
        {
            throw std::runtime_error("Not implemented");
        }

        //============== MA Gate (Measure all qubits in Pauli-Z) ================
        virtual void MA_GATE(const IdxType repetition)
        {
            auto ma_start = std::chrono::high_resolution_clock::now();
            
            SAFE_FREE_HOST(results);
            SAFE_ALOC_HOST(results, sizeof(IdxType) * repetition);
            memset(results, 0, sizeof(IdxType) * repetition);
 
            // Move to right orthogonal gauge for improved scaling before sampling
            network.position(1);

            // "Perfect" Sampling Algorithm
            //  https://tensornetwork.org/mps/algorithms/sampling/
            //
            //  Overview:
            //  Make Reduced Density Matrix (over qubit i) of network
            //  Diagonal of RDM are the probabilities of obtaining 0 or 1 for qubit i
            //  Sample qubit i
            //  Contracte result of sampling as 0 or 1 vector into starting network
            //  Repeat process
            //  


	    // memoization, storing rdms for later lookup
	    std::unordered_map<std::string, itensor::ITensor> memo;

            for (IdxType i = 0; i < repetition; i++)
            {
                // Work with copy of network
                auto network_ =  network;
                IdxType res = 0;


                for(IdxType j = 1;j <= n_qubits; j++){

                    // RNG
                    ValType r = uni_dist(rng);


                    // Sampling final qubit
                    if(j==n_qubits){

     
                        auto rdm = network_(1) * prime(dag(network_(1)));
     			// Bug in iTensor which initializes single site MPS with dangling link 
			double p_si;
			if(n_qubits==1){
				p_si = std::real(eltC(rdm,1,1,1,1));
			} else{
                        	p_si = std::real(eltC(rdm,1,1));
			}
                    
                        if (r >= p_si){
                            res |= static_cast<IdxType>(1) << (j-1);
                            }
                        results[i] = res;
                        }
                    else{
                        // Sampling qubits 0 through n_qubits - 1

                        // Initial build of reduced density matrix on first location
			itensor::ITensor rdm;
			std::string key = std::to_string(res)+" "+std::to_string(j);
			if(memo.find(key) != memo.end()){ rdm = memo[key]; }
			else {
			
			rdm = prime(dag(network_(n_qubits-j+1)),"Link")*network_(n_qubits-j+1);
                      
                        for (auto k = n_qubits-j; k >= 1 ; k--){
                            
                            rdm *= network_(k);
                            if (k==1){
                                rdm *= prime(dag(network_(k)));
                            }
                            else{
                                rdm *= prime(dag(network_(k)),"Link");
                            }
                        }
			}
                     
                        auto p_si = std::real(eltC(rdm,1,1));

                        auto site = sites(j);
                        auto si = itensor::ITensor(site);

                        if (r <= p_si){

                            // si =  |0>
                            si.set(1,1.);
                            si.set(2,0.);

                            auto temp = si * network_(1);

                            temp *= network_(2);


                            auto temp_net = itensor::MPS(n_qubits-j);
                            for (int k = 1;k<=(n_qubits-j);k++){
                            
                                temp_net.ref(k) = network_(k+1);}
    
                            temp_net.set(1,temp);
                            network_ = temp_net;
                            
                            // Dont divide by 0
                            if (p_si > 0){
                                network_ /= std::sqrt(p_si);
                            }
			    if(memo.find(key) == memo.end()){ memo[key] = rdm; }

                        }
                        else{

                           // si =  |1>
                            si.set(1,0.);
                            si.set(2,1.);
      
                            auto temp = si * network_(1);
              
                            temp *= network_(2);
                 
                            auto temp_net = itensor::MPS(n_qubits-j);
              
                            for (int k = 1;k<=(n_qubits-j);k++){
                             
                                temp_net.ref(k) = network_(k+1);}
                    
                            temp_net.set(1,temp);
                          
                            network_ = temp_net;
                  
                    

                            res |= static_cast<IdxType>(1) << (j-1);
			    

                            // Dont divide by 0
                            if (p_si != 1){
                            network_ /= std::sqrt(1-p_si);
                            }
			    if(memo.find(key) == memo.end()){ memo[key] = rdm; }
                    }
                    }
		        }

            }
            auto ma_end = std::chrono::high_resolution_clock::now();
            total_ma_exec += std::chrono::duration_cast<std::chrono::duration<double>>(ma_end - ma_start).count();
        }
        virtual double EXPECT_C4_GATE(const ValType *gm_real, const ValType *gm_imag, IdxType qubit0, IdxType qubit1, IdxType qubit2, IdxType qubit3, IdxType mask)
        {
            throw std::runtime_error("Not implemented");
        }

        virtual double EXPECT_C2_GATE(const ValType *gm_real, const ValType *gm_imag, IdxType qubit0, IdxType qubit1, IdxType mask)
        {
            throw std::runtime_error("Not implemented");
        }

        virtual double Expect_C0(IdxType mask)
        {
            throw std::runtime_error("Not implemented");
        }


        virtual void EXPECT_GATE(ObservableList *o)
        {
            throw std::runtime_error("Not implemented");
        }
        //============== Reset ================
        virtual void RESET_GATE(const IdxType qubit)
        {
            // Current implementation throws error. If implemented, add actual reset logic here.
            throw std::runtime_error("Not implemented");
        }

        //============== Purity Check  ================
        void Purity_Check(SVGate g, const IdxType t)
        {
            throw std::runtime_error("Not implemented");
        }

        virtual ValType *get_real() const override { throw std::runtime_error("Not implemented"); };
        virtual ValType *get_imag() const override { throw std::runtime_error("Not implemented"); };
    };

} // namespace NWQSim
