from tqdm import trange
import numpy as np
from ROOT import TLorentzVector, TVector3

DR_CUT = 0.4
K =6
n_events = 80000
filename_temp = 'data100k_raw_combined_atlas_cut'
###TEST
#n_events=100
#filename_temp = 'data100k_raw_combined_atlas_cut_TEST'
#######

import sys
import ROOT
ROOT.gSystem.Load("libDelphes")
ROOT.gInterpreter.Declare('#include "classes/DelphesClasses.h"')
ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootTreeReader.h"')
ROOT.gInterpreter.Declare('#include "external/ExRootAnalysis/ExRootResult.h"')

def add_children(queue, part, pid):
    queue.append([part.D1, pid])
    queue.append([part.D2, pid])
    
def get_main_children_aux(main_children, branchParticle, parent_pid=6):
    """We're going into this assuming that the W was not recorded as an event
    and all three quark children of t/tb have t as a parent"""
    n_parts = branchParticle.GetEntries()
    main_children = []
    for i in range(n_parts):
        if(len(main_children) == 3):
            break
        part = branchParticle.At(i)
        pid = part.PID
        if(abs(pid)>6):
            continue
        
        parent1_pid = -1000
        parent2_pid = -1000
        if(part.M1>0):
            parent1_pid = branchParticle.At(part.M1).PID
        
        if(part.M2>0):
            parent2_pid = branchParticle.At(part.M2).PID
        
        if(pid == parent1_pid or pid == parent2_pid):
            continue
            
        if(parent1_pid == parent_pid or parent2_pid == parent_pid):
            pT = part.PT
            eta = part.Eta
            phi = part.Phi
#             print("\t\t AUX: Particle %d is a new descendant with PID %d"%(i, pid))
            main_children.append(part.P4())

    return (len(main_children)==3)

def get_main_children(main_children, queue, branchParticle):
    visited = set()
    idx = 0
    while(len(main_children)<3 and idx < len(queue)):
        curr = queue[idx]
        idx+=1
        particle_num = int(curr[0])
        father_pid = int(curr[1])
        if(particle_num < 0):
            continue    
        if(particle_num in visited):
            continue        
        visited.add(particle_num)
        part = branchParticle.At(particle_num)
        pid = part.PID
#         print("particle num %d has PID %d with parent PID %d"%(particle_num, pid, father_pid))
        add_children(queue, part, pid)

        if(abs(pid)>6):
            continue

        if(pid != father_pid):
            pT = part.PT
            eta = part.Eta
            phi = part.Phi
            #print("\tParticle %d is a new descendant with PID %d"%(particle_num, pid))
            main_children.append(part.P4())
    
    return (len(main_children)==3)

def populate_graph(graph, main_children, jet, DR_hist, offset=0):
    for tidx in range(len(main_children)):
        jet4 = jet.P4()
        DR = jet4.DeltaR(main_children[tidx])
        DR_hist.append(DR)
        if(DR < DR_CUT):
            graph[i].append(tidx+offset)
            
def try_kuhn(v, mt, used, graph):
    """For maximum bipartite matching that I really don't need to do :("""
    if(used[v]):
        return False
    used[v] = True
    for to in graph[v]:
        if(mt[to] == -1 or try_kuhn(mt[to], mt, used, graph)):
            mt[to] = v
            return True
    return False


from ROOT import TLorentzVector


n_top = 0
n_anti_top = 0
n_jet_hist = []
DR_hist = []  

inp = [] # dims (N_events, -1, N_in_features)
oup = [] # dims (N_events, -1, N_out_features)
aux_info = [] #dims (N_events, -1, 2) where each jet has P4 and Btag info
event_tag = [] #dims (N_events ,2) #event tag and "ideal event" = 2 b tag, 2 tau tag, 6 jets total

file_idx=0
while(len(inp)<n_events and file_idx<=40):
    directory= str("/data/delon/tth_hTOtata/Events/run_08_%d/tag_1_delphes_events.root"%(file_idx))
    print('currently on',directory, flush=True)
    
    n_skip = 0
    chain = ROOT.TChain("Delphes")   
    chain.Add(directory)
    treeReader = ROOT.ExRootTreeReader(chain)
    n_entries = treeReader.GetEntries()
    
    branchParticle = treeReader.UseBranch("Particle")
    branchJet = treeReader.UseBranch("Jet")
    branchMET = treeReader.UseBranch("MissingET")
    
    for entry in range(n_entries): #events
        if(len(inp)>= n_events):
            break

        #print("Event %d of %d"%(entry, n_entries))



        treeReader.ReadEntry(entry)   
        
        n_jets = branchJet.GetEntries()
        n_btag_jets = 0
        for i in range(n_jets):
            jet = branchJet.At(i)
            
            n_btag_jets += jet.BTag

        #multiplicity cut
        if(not((n_jets >= 5 and n_btag_jets>=2) or (n_jets >= 6 and n_btag_jets >= 1)) or n_jets>15):
            continue
        
        tau_jets_pT = []
        tau_jets_P4 = []
        tau_jets_eta = []

        other_jets_pT = []
        other_jets_eta = []
        for i in range(branchJet.GetEntries()):
            jet = branchJet.At(i)
            if(jet.TauTag):
                tau_jets_pT.append(jet.PT)
                tau_jets_eta.append(jet.Eta)
                tau_jets_P4.append(jet.P4())
            else:
                other_jets_pT.append(jet.PT)
                other_jets_eta.append(np.abs(jet.Eta))
                
        #pT tau had cut
        if(len(tau_jets_pT) != 2):
            continue
        if(not (max(tau_jets_pT) > 40 and min(tau_jets_pT)>30)):
            continue

        #leading jet cut
        leading_jet_idx = np.argmax(other_jets_pT)
        if(not (other_jets_pT[leading_jet_idx] > 70 and other_jets_eta[leading_jet_idx] <3.2)):
            continue
            
        #angular cuts
        dR_tautau= tau_jets_P4[0].DeltaR(tau_jets_P4[1])
        if(not (0.6 < dR_tautau and dR_tautau<2.5)):
            continue
        if(not(np.abs(tau_jets_eta[0] - tau_jets_eta[1])<1.5)):
            continue
        
        #coll app cuts

        rV = [tau.Vect() for tau in tau_jets_P4]
        zaxis = rV[0].Cross(rV[1]).Unit()
        yaxis = zaxis.Cross(rV[0]).Unit()
        xaxis = rV[0].Unit()

        temp_mags = [V.Mag() for V in rV]

        rV[0].SetXYZ(rV[0].Mag(),0,0)
        rV[1].SetXYZ(rV[1].Dot(xaxis), rV[1].Dot(yaxis), 0)

        for i in range(2):
            assert(np.abs(temp_mags[i] - rV[i].Mag()) <1e-5)

        v_MET = TVector3()
        MET_E = branchMET.At(0).MET
        MET_Phi = branchMET.At(0).Phi
        v_MET.SetPtEtaPhi(MET_E, 0,MET_Phi) #ignore eta?


        ri = [np.abs((v_MET.Y()*rV[i].X() - v_MET.X()*rV[i].Y())/(rV[0].Y()*rV[1].X()-rV[0].X()*rV[1].Y())) for i in range(2)]
        fi = [1/(1+ri[(i+1)%2]) for i in range(2)]

        if(not (0.1 < fi[0] and fi[0] < 1.4 and 0.1 < fi[1] and fi[1] < 1.4)):
            continue

        

        n_parts = branchParticle.GetEntries()
        n_jet_hist.append(n_jets)

        t_idx = -1
        found_top = False
        
        tb_idx = -1
        found_anti_top = False
        
        #queues
        top_s = []
        top_s_idx = 0
        
        anti_top_s = []
        anti_top_s_idx = 0
        
        main_top_children = []
        main_anti_top_children = []
        for i in range(n_parts):
            part = branchParticle.At(i)
            pid = part.PID
            if(not found_top and pid == 6):
                t_idx = i
                found_top = True
                add_children(top_s, part, pid)
#######                print("\tFound top at %d"%(t_idx))

            elif(not found_anti_top and pid == -6):
                tb_idx = i
                found_anti_top = True
                add_children(anti_top_s, part, pid)
#                print("\tFound anti-top at %d"%(tb_idx))
                     
        found_all_top = get_main_children(main_top_children, top_s, branchParticle)
        if(not found_all_top):
            #print("\t Doing auxillary search")
            found_all_top = get_main_children_aux(main_top_children, branchParticle, parent_pid=6)
            
        found_all_anti_top = get_main_children(main_anti_top_children, anti_top_s, branchParticle)
        if(not found_all_anti_top):
            #print("\t Doing auxillary search")
            found_all_anti_top = get_main_children_aux(main_anti_top_children, branchParticle, parent_pid=-6)

        if(((not found_all_top) or (not found_all_anti_top))):
            #print("\t skipping this event")
            continue
        
        n_jets = branchJet.GetEntries()
        jets = []
        aux_jets = []
        dead_idx = []
        jets_oup = [[0,0,1] for j in range(n_jets)]

        graph = [ [] for i in range(n_jets)]
        n_tau_tag = 0
        n_b_tag = 0
        for i in range(n_jets):
            jet = branchJet.At(i)
            jet4 = jet.P4()
            aux_jets_curr = [jet4, jet.BTag, jet.TauTag]
            aux_jets.append(aux_jets_curr)
            curr_jet = [jet.PT, jet.Eta, jet.Phi, jet.BTag]
            jets.append(curr_jet)
            if(jet.TauTag):
                dead_idx.append(i)
                n_tau_tag += 1
                continue
            
                        
            if(0==0):
                populate_graph(graph, main_top_children, jet, DR_hist)
                populate_graph(graph, main_anti_top_children, jet, DR_hist,offset = 3)
            
            n_b_tag += jet.BTag
        
        
        if(n_tau_tag!=2):
            continue

        n = n_jets
        mt = [-1 for j in range(K)]
        used1 = [False for j in range(n)]
        
        
        for v in range(n):
            for to in graph[v]:
                if(mt[to]==-1):
                    mt[to] = v
                    used1[v] = True
                    break

        for v in range(n):
            if(used1[v]):
                continue
            used = [False for j in range(n)]
            try_kuhn(v, mt, used, graph)
    
        for i in range(K):
            if(mt[i] != -1):
                if(i<3):
                    #we found a top jets
#                    #print("\tLabelling jet %d as top"%(mt[i]))
                    jets_oup[mt[i]] = [1,0,0]
                    n_top += 1
                else:
                    #we found an anti-top jet
#                    #print("\tLabelling jet %d as anti-top"%(mt[i]))
                    jets_oup[mt[i]] = [0,1,0]
                    n_anti_top += 1

#        dead_idx = sorted(dead_idx, reverse=True)
#        for idx in dead_idx:
#            #go from back to front so that index aren't shifted
#            del jets_oup[idx]
        
        ideal_event = int(n_b_tag==2 and n_tau_tag==2 and len(jets_oup) == 6) #ideal event
#        #print(n_b_tag, n_tau_tag, len(jets_oup))
        
            
        assert(len(jets_oup) == len(aux_jets))
        assert(len(jets) == len(jets_oup))
        inp.append(jets)
        oup.append(jets_oup)
        aux_info.append(aux_jets)
        event_tag.append([0, ideal_event, branchMET.At(0).MET, branchMET.At(0).Eta, branchMET.At(0).Phi])  #0 corresponds to ttH
        #print('added %d'%(entry))

    file_idx += 1

file_idx=0
    #todo chnage while loop ocndition
linp0 = len(inp)
while(len(inp)-linp0<n_events and file_idx<=95):
    directory= str("/data/delon/ttbar_tTOtaunub/Events/run_02_%d/tag_1_delphes_events.root"%(file_idx))
    print('currently on',directory, flush=True)
    
    n_skip = 0
    chain = ROOT.TChain("Delphes")   
    chain.Add(directory)
    treeReader = ROOT.ExRootTreeReader(chain)
    n_entries = treeReader.GetEntries()
    
    
    branchParticle = treeReader.UseBranch("Particle")
    branchJet = treeReader.UseBranch("Jet")
    branchMET = treeReader.UseBranch("MissingET")
    
    for entry in range(n_entries): #events
        if(len(inp)-linp0 >= n_events):
            break
        #print("Event %d of %d"%(entry, n_entries))
        treeReader.ReadEntry(entry)   
        
        n_jets = branchJet.GetEntries()
        n_btag_jets = 0
        for i in range(n_jets):
            jet = branchJet.At(i)
            
            n_btag_jets += jet.BTag

        #multiplicity cut
        if(not((n_jets >= 5 and n_btag_jets>=2) or (n_jets >= 6 and n_btag_jets >= 1)) or n_jets>15):
            continue
        
        tau_jets_pT = []
        tau_jets_P4 = []
        tau_jets_eta = []

        other_jets_pT = []
        other_jets_eta = []
        for i in range(branchJet.GetEntries()):
            jet = branchJet.At(i)
            if(jet.TauTag):
                tau_jets_pT.append(jet.PT)
                tau_jets_eta.append(jet.Eta)
                tau_jets_P4.append(jet.P4())
            else:
                other_jets_pT.append(jet.PT)
                other_jets_eta.append(np.abs(jet.Eta))
                
        #pT tau had cut
        if(len(tau_jets_pT) != 2):
            continue
        if(not (max(tau_jets_pT) > 40 and min(tau_jets_pT)>30)):
            continue

        #leading jet cut
        leading_jet_idx = np.argmax(other_jets_pT)
        if(not (other_jets_pT[leading_jet_idx] > 70 and other_jets_eta[leading_jet_idx] <3.2)):
            continue
            
        #angular cuts
        dR_tautau= tau_jets_P4[0].DeltaR(tau_jets_P4[1])
        if(not (0.6 < dR_tautau and dR_tautau<2.5)):
            continue
        if(not(np.abs(tau_jets_eta[0] - tau_jets_eta[1])<1.5)):
            continue
        
        #coll app cuts

        rV = [tau.Vect() for tau in tau_jets_P4]
        zaxis = rV[0].Cross(rV[1]).Unit()
        yaxis = zaxis.Cross(rV[0]).Unit()
        xaxis = rV[0].Unit()

        temp_mags = [V.Mag() for V in rV]

        rV[0].SetXYZ(rV[0].Mag(),0,0)
        rV[1].SetXYZ(rV[1].Dot(xaxis), rV[1].Dot(yaxis), 0)

        for i in range(2):
            assert(np.abs(temp_mags[i] - rV[i].Mag()) <1e-5)

        v_MET = TVector3()
        MET_E = branchMET.At(0).MET
        MET_Phi = branchMET.At(0).Phi
        v_MET.SetPtEtaPhi(MET_E, 0,MET_Phi) #ignore eta?

        ri = [np.abs((v_MET.Y()*rV[i].X() - v_MET.X()*rV[i].Y())/(rV[0].Y()*rV[1].X()-rV[0].X()*rV[1].Y())) for i in range(2)]
        fi = [1/(1+ri[(i+1)%2]) for i in range(2)]

        if(not (0.1 < fi[0] and fi[0] < 1.4 and 0.1 < fi[1] and fi[1] < 1.4)):
            continue

        n_parts = branchParticle.GetEntries()
        n_jet_hist.append(n_jets)

        t_idx = -1
        found_top = False
        
        tb_idx = -1
        found_anti_top = False
        
        #queues
        top_s = []
        top_s_idx = 0
        
        anti_top_s = []
        anti_top_s_idx = 0
        
        main_top_children = []
        main_anti_top_children = []
        for i in range(n_parts):
            part = branchParticle.At(i)
            pid = part.PID
            if(not found_top and pid == 6):
                t_idx = i
                found_top = True
                add_children(top_s, part, pid)
#                print("\tFound top at %d"%(t_idx))

            elif(not found_anti_top and pid == -6):
                tb_idx = i
                found_anti_top = True
                add_children(anti_top_s, part, pid)
#                print("\tFound anti-top at %d"%(tb_idx))
                     
        found_all_top = get_main_children(main_top_children, top_s, branchParticle)
        if(not found_all_top):
#            print("\t Doing auxillary search")
            found_all_top = get_main_children_aux(main_top_children, branchParticle, parent_pid=6)
            
        found_all_anti_top = get_main_children(main_anti_top_children, anti_top_s, branchParticle)
        if(not found_all_anti_top):
#            print("\t Doing auxillary search")
            found_all_anti_top = get_main_children_aux(main_anti_top_children, branchParticle, parent_pid=-6)

        n_jets = branchJet.GetEntries()
        jets = []
        aux_jets = []
        dead_idx = []
        jets_oup = [[0,0,1] for j in range(n_jets)]

        graph = [ [] for i in range(n_jets)]
        n_tau_tag = 0
        n_b_tag = 0
        for i in range(n_jets):
            jet = branchJet.At(i)
            jet4 = jet.P4()
            aux_jets_curr = [jet4, jet.BTag, jet.TauTag]
            aux_jets.append(aux_jets_curr)
            curr_jet = [jet.PT, jet.Eta, jet.Phi, jet.BTag]
            jets.append(curr_jet)

            if(jet.TauTag):
                dead_idx.append(i)
                n_tau_tag += 1
                continue
            
                        
            if(1==0):
                populate_graph(graph, main_top_children, jet, DR_hist)
                populate_graph(graph, main_anti_top_children, jet, DR_hist,offset = 3)
            
            n_b_tag += jet.BTag
        
        if(n_tau_tag!=2):
            continue
        assert(n_tau_tag==2)

        n = n_jets
        mt = [-1 for j in range(K)]
        used1 = [False for j in range(n)]
        
        
        for v in range(n):
            for to in graph[v]:
                if(mt[to]==-1):
                    mt[to] = v
                    used1[v] = True
                    break

        for v in range(n):
            if(used1[v]):
                continue
            used = [False for j in range(n)]
            try_kuhn(v, mt, used, graph)
    
        for i in range(K):
            if(mt[i] != -1):
                if(i<3):
                    #we found a top jets
#                    print("\tLabelling jet %d as top"%(mt[i]))
                    jets_oup[mt[i]] = [1,0,0]
                    n_top += 1
                else:
                    #we found an anti-top jet
#                    print("\tLabelling jet %d as anti-top"%(mt[i]))
                    jets_oup[mt[i]] = [0,1,0]
                    n_anti_top += 1

        for jet_oup in jets_oup:
            if(jet_oup != [0,0,1]):
                print('WHAT')
                assert(1==0)
#        dead_idx = sorted(dead_idx, reverse=True)
#        for idx in dead_idx:
#            #go from back to front so that index aren't shifted
#            del jets_oup[idx]
        
        ideal_event = int(n_b_tag==2 and n_tau_tag==2 and len(jets_oup) == 6) #ideal event
#        print(n_b_tag, n_tau_tag, len(jets_oup))
        
            
        assert(len(jets_oup) == len(aux_jets))
        assert(len(jets) == len(jets_oup))
        inp.append(jets)
        oup.append(jets_oup)
        aux_info.append(aux_jets)
        event_tag.append([1, ideal_event, branchMET.At(0).MET, branchMET.At(0).Eta, branchMET.At(0).Phi])  #0 corresponds to ttH

        #print('added %d'%(entry))
    file_idx += 1

print('We got %d ttH and %d ttbar events'%(linp0, len(inp)-linp0))
#From here we output the RAW JET KINEMATIC DATA without preprocessing
import pickle

outfile = open(filename_temp+'.pkl', 'wb')
pickle.dump((aux_info, oup, event_tag), outfile)
outfile.close()


#From here we output the RAW JET KINEMATIC DATA without preprocessing
import pickle

outfile = open(filename_temp+'_small.pkl', 'wb')
small_size = n_events//100
pickle.dump((aux_info[:small_size]+aux_info[-small_size:], oup[:small_size]+oup[-small_size:], event_tag[:small_size]+event_tag[-small_size:]), outfile)
outfile.close()
