from gurobipy import *
from pathlib import Path
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime
from utils import ClassificationReport, evt_rel_distribution
import numpy as np
import pickle

label_map_rel = {0:'BEFORE', 1:'AFTER', 2:'OVERLAP'}
# 'O' needs to have the o index to make the data proccessing code work                                                     
label_map_evt = {0: 'O',
                 1:'B-CLINICAL_DEPT', 2:'I-CLINICAL_DEPT',
                 3:'B-EVIDENTIAL', 4:'I-EVIDENTIAL',
                 5:'B-OCCURRENCE', 6:'I-OCCURRENCE',
                 7:'B-PROBLEM', 8:'I-PROBLEM',
                 9:'B-TEST', 10:'I-TEST',
                 11:'B-TREATMENT', 12:'I-TREATMENT'}

class Event_Inference():
    def __init__(self, prob_evts, event_constraint_param,  event_constraints):
        self.model = Model("Event Inference")
        self.prob_evts = prob_evts
        self.N, self.Nc = prob_evts.shape

        # local event prediction labels for each token                                                            
        self.pred_evt_labels = list(np.argmax(prob_evts, axis=1))
        
        # lambda
        self.event_constraint_param = event_constraint_param

        # ratio
        self.event_constraints = event_constraints
        
    def define_vars(self):
        var_table_e = []
        # event variables                                            
        for n in range(self.N):
            sample = []
            for p in range(self.Nc):
                sample.append(self.model.addVar(vtype=GRB.BINARY, name="e_%s_%s"%(n,p)))
            var_table_e.append(sample)
        return var_table_e
    
    def objective(self, samples_e, p_table_e):

        obj = 0.0
        assert len(samples_e) == self.N
        assert len(samples_e[0]) == self.Nc
        
        # event                                                                                                   
        for n in range(self.N):
            for p in range(self.Nc):
                obj += samples_e[n][p] * p_table_e[n][p]
                # add event soft constraint
                for k, r in self.event_constraints.items():
                    if n > 0 and np.argmax(p_table_e[n-1]) == k[0]:
                        ld = self.event_constraint_param[k]
                        if p == k[1]:
                            obj += ld * samples_e[n][p] * (1 - r)
                        else:
                            obj -= ld * samples_e[n][p] * r
        return obj

    def single_label(self, sample):
        return sum(sample) == 1

    def span_consistency(self, sample, n):
        return sample[n][0] + sample[n+1][2] + sample[n+1][4] + sample[n+1][6] + \
            sample[n+1][8] + sample[n+1][10] + sample[n+1][12] <= 1

    def type_consistency(self, sample, n):
        return [sample[n+1][2] - sample[n][2] - sample[n][1],
                sample[n+1][4] - sample[n][4] - sample[n][3],
                sample[n+1][6] - sample[n][6] - sample[n][5],
                sample[n+1][8] - sample[n][8] - sample[n][7],
                sample[n+1][10] - sample[n][10] - sample[n][9],
                sample[n+1][12] - sample[n][12] - sample[n][11]]

    def define_constraints(self, var_table_e):
        # Constraint 1: single label assignment                                                                   
        for n in range(self.N):
            self.model.addConstr(self.single_label(var_table_e[n]), "c1_%s" % n)

        '''
        c3_count = 0
        for n in range(self.N-1):
            # Constraint 2: span consistency O --> O / B
            self.model.addConstr(self.span_consistency(var_table_e, n), "c2_%s" % n)
            # Constraint 3: type consistency B-type --> I-type / I-type --> I-type
            for ci in self.type_consistency(var_table_e, n):
                self.model.addConstr(ci <= 0, "c3_%s" % c3_count)
                c3_count += 1
        '''
        return
        
    def run(self):
        try:
            # Define variables                                                                                    
            var_table_e = self.define_vars()
            
            # Set objective                                                                                       
            self.model.setObjective(self.objective(var_table_e,  self.prob_evts), GRB.MAXIMIZE)
            
            # Define constrains                                                                                   
            self.define_constraints(var_table_e)

            # run model                                                                                           
            self.model.setParam('OutputFlag', False)
            self.model.optimize()

        except GurobiError:
            print('Error reported')

    def predict(self):
        
        evt_count = 0
        for i, v in enumerate(self.model.getVars()):
            is_evt = True if v.varName.split('_')[0] == 'e' else False
            # sample idx                                                                                           
            s_idx = int(v.varName.split('_')[1])
            # sample class index                                                                                   
            c_idx = int(v.varName.split('_')[2])

            if v.x == 1.0 and self.pred_evt_labels[s_idx] != c_idx:
                #print(v.varName, self.pred_evt_labels[s_idx])
                self.pred_evt_labels[s_idx] = c_idx
                evt_count += 1
        return evt_count


class Global_Inference():
    
    def __init__(self, prob_evts, prob_rels, rel_evt_indices, constraint_param, constraints, event_constraint_param,
                event_constraints, evt_gold=[], label2idx={}, ew=1.0, relation_agg_gold={}, rkey=[], debug=False):
        self.model = Model("inference")

        self.debug = debug
        
        self.prob_evts = prob_evts
        self.prob_rels = prob_rels

        self.N, self.Nc = prob_evts.shape
        if prob_rels.shape[0] == 0:
            self.M = 0
            self.pred_rel_labels = []
        else:
            self.M, self.Mc = prob_rels.shape
            self.pred_rel_labels = list(np.argmax(prob_rels, axis=1))

        # local event prediction labels for each token
        self.pred_evt_labels = list(np.argmax(prob_evts, axis=1))
        
        self.ew = ew

        self.relation_agg_gold = relation_agg_gold
        self.rkey = rkey

        # lambda
        self.constraint_param = constraint_param
        self.event_constraint_param = event_constraint_param

        # ratio
        self.constraints = constraints
        self.event_constraints = event_constraints
            
        if evt_gold:
            self.evt_rel_types = [(evt_gold[idx[0]], evt_gold[idx[2]])
                                  if len(idx) == 4 else (evt_gold[idx[0]], evt_gold[idx[1]])
                                  for idx in rel_evt_indices]
        else:
            self.evt_rel_types = [(self.pred_evt_labels[idx[0]], self.pred_evt_labels[idx[2]])
                                  if len(idx) == 4
                                  else (self.pred_evt_labels[idx[0]], self.pred_evt_labels[idx[1]])
                                  for idx in rel_evt_indices]
        
        self.label2idx = label2idx
        self.idx2label = OrderedDict([(v,k) for k,v in label2idx.items()])
        
        self.pairs = rel_evt_indices
        self.idx2pair = {n: self.pairs[n] for n in range(len(self.pairs))}
        self.pair2idx = {v:k for k,v in self.idx2pair.items()}

        
    def define_vars(self):
        var_table_e, var_table_r = [], []
        
        # event variables
        for n in range(self.N):
            sample = []
            for p in range(self.Nc):
                sample.append(self.model.addVar(vtype=GRB.BINARY, name="e_%s_%s"%(n,p)))
            var_table_e.append(sample)

        # relation variables
        for m in range(self.M):
            sample = []
            for p in range(self.Mc):
                sample.append(self.model.addVar(vtype=GRB.BINARY, name="r_%s_%s"%(m,p)))
            var_table_r.append(sample)

        return var_table_e, var_table_r

    def objective(self, samples_e, samples_r, p_table_e, p_table_r):
    
        obj = 0.0

        assert len(samples_e) == self.N 
        assert len(samples_r) == self.M
        assert len(samples_e[0]) == self.Nc
        if self.M > 0:
            assert len(samples_r[0]) == self.Mc
        
        # event
        
        for n in range(self.N):
            for p in range(self.Nc):
                obj += self.ew * samples_e[n][p] * p_table_e[n][p]

                '''
                # add event soft constraint
                for k, r in self.event_constraints.items():
                    if n > 0 and np.argmax(p_table_e[n-1]) == k[0]:
                        ld = self.event_constraint_param[k]
                        
                        if p == k[1]:
                            obj -= ld * samples_e[n][p] * (1 - r)
                        else:
                            obj += ld * samples_e[n][p] * r
                '''  
        # relation
        for m in range(self.M):
            for p in range(self.Mc):
                obj += samples_r[m][p] * p_table_r[m][p]
                
                # regularized by prior event + rel distribution
                # exclude corrupted cases where event ''                
                if all(i % 2 != 0 for i in self.evt_rel_types[m]):
                    key = (self.evt_rel_types[m][0], self.evt_rel_types[m][1], p)
                    
                    if key in self.constraint_param:
                        ld = self.constraint_param[key]
                        r = self.constraints[key]
                        #print(key, ld, r)
                        for pp in range(self.Mc):
                            if pp == p:
                                obj += ld * samples_r[m][pp] * (1 - r)
                            else:
                                obj -= ld * samples_r[m][pp] * r
                                
        return obj
        
    def single_label(self, sample):
        return sum(sample) == 1

    def span_consistency(self, sample, n):
        return sample[n][0] + sample[n+1][2] + sample[n+1][4] + sample[n+1][6] + \
            sample[n+1][8] + sample[n+1][10] + sample[n+1][12] <= 1

    def type_consistency(self, sample, n):
        return [sample[n+1][2] - sample[n][2] - sample[n][1],
                sample[n+1][4] - sample[n][4] - sample[n][3],
                sample[n+1][6] - sample[n][6] - sample[n][5],
                sample[n+1][8] - sample[n][8] - sample[n][7],
                sample[n+1][10] - sample[n][10] - sample[n][9],
                sample[n+1][12] - sample[n][12] - sample[n][11]]
    
    def define_constraints(self, var_table_e, var_table_r):
        # Constraint 1: single label assignment
        for n in range(self.N):
            self.model.addConstr(self.single_label(var_table_e[n]), "c1_%s" % n)
        for m in range(self.M):
            self.model.addConstr(self.single_label(var_table_r[m]), "c1_%s" % (self.N + m))
        '''
        c3_count = 0
        for n in range(self.N-1):
            # Constraint 2: span consistency O --> O / B                                                   
            self.model.addConstr(self.span_consistency(var_table_e, n), "c2_%s" % n)
            # Constraint 3: type consistency B-type --> I-type / I-type --> I-type
            for ci in self.type_consistency(var_table_e, n):
                self.model.addConstr(ci <= 0, "c3_%s" % c3_count)
                c3_count += 1
        
        # Constraint 2: transitivity
        trans_triples = self.transitivity_list()
        print(len(trans_triples))
        t = 0
        for triple in trans_triples:
            for ci in self.transitivity_criteria(var_table_r, triple):
                self.model.addConstr(ci <= 1, "c5_%s" % t)
                t += 1
        '''
        return
    
    def transitivity_list(self):

        transitivity_samples = []
        pair2idx = self.pair2idx

        for k, (e11, e12, e21, e22) in self.idx2pair.items():
            for (re11, re12, re21, re22), i in pair2idx.items():
                if (e21, e22) == (re11, re12) and (e11, e12, re21, re22) in pair2idx.keys():
                    transitivity_samples.append((pair2idx[(e11, e12, e21, e22)],
                                                 pair2idx[(re11, re12, re21, re22)],
                                                 pair2idx[(e11, e12, re21, re22)]))
        return transitivity_samples

    def transitivity_criteria(self, samples, triplet):
        # r1  r2  Trans(r1, r2)                                                                                    
        # _____________________                                                                                    
        # b   b   b                                                                                                
        # a   a   a                                                                                                
        # b   v   b, v                                                                                             
        # a   v   a, v                                                                                             
        # v   b   b, v                                                                                             
        # v   a   a, v                                                                                             
        r1, r2, r3 = triplet
        label_dict = self.label2idx
        
        return [
            samples[r1][label_dict['BEFORE']] + samples[r2][label_dict['BEFORE']] - samples[r3][label_dict['BEFORE']],
            samples[r1][label_dict['AFTER']] + samples[r2][label_dict['AFTER']] - samples[r3][label_dict['AFTER']],
            samples[r1][label_dict['BEFORE']] + samples[r2][label_dict['OVERLAP']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['OVERLAP']], 
            samples[r1][label_dict['AFTER']] + samples[r2][label_dict['OVERLAP']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['OVERLAP']],
            samples[r1][label_dict['OVERLAP']] + samples[r2][label_dict['BEFORE']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['OVERLAP']],
            samples[r1][label_dict['OVERLAP']] + samples[r2][label_dict['AFTER']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['OVERLAP']]
        ]
    
    def run(self):
        try:
            # Define variables
            var_table_e, var_table_r = self.define_vars()

            # Set objective 
            self.model.setObjective(self.objective(var_table_e, var_table_r, self.prob_evts, 
                                                   self.prob_rels), GRB.MAXIMIZE)
            
            # Define constrains
            self.define_constraints(var_table_e, var_table_r)

            # run model
            self.model.setParam('OutputFlag', False)
            self.model.optimize()
            
        except GurobiError:
            print('Error reported')

    def predict(self):
        evt_count, rel_count = 0, 0

        for i, v in enumerate(self.model.getVars()):            
            # rel_evt indicator
            is_evt = True if v.varName.split('_')[0] == 'e' else False
            # sample idx
            s_idx = int(v.varName.split('_')[1])
            # sample class index
            c_idx = int(v.varName.split('_')[2])

            if is_evt:
                if v.x == 1.0 and self.pred_evt_labels[s_idx] != c_idx:
                    #print(v.varName, self.pred_ent_labels[s_idx])
                    self.pred_evt_labels[s_idx] = c_idx
                    evt_count += 1
            else:
                if v.x == 1.0 and self.pred_rel_labels[s_idx] != c_idx:
                    #event1 = self.evt_rel_types[s_idx][0]
                    #event2 = self.evt_rel_types[s_idx][1]
                    '''
                    if self.debug:
                        print("(%s, %s);%s;%s;%s;%s;%s;%s" % (label_map_evt[event1], label_map_evt[event2],
                                                              label_map_rel[self.relation_agg_gold[self.rkey[s_cidx]]],
                                                              label_map_rel[self.pred_rel_labels[s_idx]],
                                                              label_map_rel[c_idx],
                                                              self.prob_rels[s_idx][0],
                                                              self.prob_rels[s_idx][1],
                                                              self.prob_rels[s_idx][2]))
                    '''
                    self.pred_rel_labels[s_idx] = c_idx
                    rel_count += 1

        #print('# of global event correction: %s' % evt_count)
        #print('# of global relation correction: %s' % rel_count)
        #print('Objective Function Value:', self.model.objVal)
        
        return rel_count, evt_count
