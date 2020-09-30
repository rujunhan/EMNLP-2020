from gurobipy import *
from pathlib import Path
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime
import numpy as np
import pickle


class Relation_Inference():
    def __init__(self, probs, constraint_param, constraints, sample_event_types, pairs, label_map_rel,
                 event_heads):
        self.model = Model("Relation Inference")
        self.probs = probs
        self.M, self.Mc = probs.shape

        self.label_map_rel = label_map_rel
        
        # event pair types; maybe NONE
        self.sample_event_types = sample_event_types
        
        # lambda                                                                                                    
        self.constraint_param = constraint_param

        # constraint ratio                                                                                            
        self.constraints = constraints

        # local relation predictions
        self.pred_rel_labels = list(np.argmax(probs, axis=1))

        # prior distribution
        #self.sample_pair_text = sample_pair_text
        #assert len(sample_pair_text) == self.M
        self.event_heads = event_heads
        
        # pairs ids
        self.pairs = pairs
        #self.idx2pair = {n: self.pairs[n] for n in range(len(pairs))}
        #self.pair2idx = {v:k for k,v in self.idx2pair.items()}
        
    def define_vars(self):
        var_table_r = []
        
        # relation variables                                                                                           
        for m in range(self.M):
            sample = []
            for p in range(self.Mc):
                sample.append(self.model.addVar(vtype=GRB.BINARY, name="e_%s_%s"%(m,p)))
            var_table_r.append(sample)
        return var_table_r

    
    def objective(self, samples_r, p_table_r):
        obj = 0.0
        if self.M > 0:
            assert len(samples_r[0]) == self.Mc
            
        # relation                                                                                                  
        for m in range(self.M):
            for p in range(self.Mc):
                obj += samples_r[m][p] * p_table_r[m][p]

                if len(self.sample_event_types[m]) == 0: continue

                # LR optimization
                key = (self.sample_event_types[m][0], self.sample_event_types[m][1], p)
                if key in self.constraint_param:
                    ld = self.constraint_param[key]
                    r = self.constraints[key]
                    
                    for pp in range(self.Mc):
                        if pp == p or (pp == self.label_map_rel['NONE'] and
                                       p == self.label_map_rel['VAGUE']):
                            obj += ld * samples_r[m][pp] * (1 - r)
                        else:
                            obj -= ld * samples_r[m][pp] * r

        return obj

    
    def single_label(self, sample):
        return sum(sample) == 1

    def rel_ent_sum(self, samples_e, samples_r, e, r, c):
        # negative rel constraint
        return samples_e[e[0]][0] + samples_e[e[1]][0] - samples_r[r][c]

    def rel_left_ent(self, samples_e, samples_r, e, r, c):
        # positive rel left constraint
        return samples_e[e[0]][1] - samples_r[r][c]
        
    def rel_right_ent(self, samples_e, samples_r, e, r, c):
        # positive rel right constraint
        return samples_e[e[1]][1] - samples_r[r][c]

    
    def transitivity_list(self):

        transitivity_samples = []
        pair2idx = self.pair2idx

        for k, (e1, e2) in self.idx2pair.items():
            for (re1, re2), i in pair2idx.items():
                if e2 == re1 and (e1, re2) in pair2idx.keys():
                    transitivity_samples.append((pair2idx[(e1, e2)], pair2idx[(re1, re2)], pair2idx[(e1, re2)]))
        return transitivity_samples

    def transitivity_criteria(self, samples, triplet):
        r1, r2, r3 = triplet
        label_dict = label_map_rel
        return [
            samples[r1][label_dict['BEFORE']] + samples[r2][label_dict['BEFORE']] - samples[r3][label_dict['BEFORE']],
            samples[r1][label_dict['AFTER']] + samples[r2][label_dict['AFTER']] - samples[r3][label_dict['AFTER']],
            samples[r1][label_dict['SIMULTANEOUS']] + samples[r2][label_dict['SIMULTANEOUS']] - samples[r3][label_dict['SIMULTANEOUS']],
            #samples[r1][label_dict['VAGUE']] + samples[r2][label_dict['VAGUE']] - samples[r3][label_dict['VAGUE']],
            #samples[r1][label_dict['NONE']] + samples[r2][label_dict['NONE']] - samples[r3][label_dict['NONE']],
            samples[r1][label_dict['BEFORE']] + samples[r2][label_dict['VAGUE']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['VAGUE']], 
            samples[r1][label_dict['AFTER']] + samples[r2][label_dict['VAGUE']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['VAGUE']],
            samples[r1][label_dict['VAGUE']] + samples[r2][label_dict['BEFORE']] - samples[r3][label_dict['BEFORE']] - samples[r3][label_dict['VAGUE']],
            samples[r1][label_dict['VAGUE']] + samples[r2][label_dict['AFTER']] - samples[r3][label_dict['AFTER']] - samples[r3][label_dict['VAGUE']]
        ]

    def define_constraints(self, var_table_r):
        # Constraint 1: single label assignment
        for m in range(self.M):
            self.model.addConstr(self.single_label(var_table_r[m]), "c1_%s" % m)

        """ Not using any previous hard constraints"""
        '''
        trans_triples = self.transitivity_list()
        self.trans_triples = trans_triples
        print("Total %s triplets:" % len(trans_triples))
        t = 0
        for triple in trans_triples:
            for ci in self.transitivity_criteria(var_table_r, triple):
                self.model.addConstr(ci <= 1, "c2_%s" % t)
                t += 1
        print("Total %s transitivity constraints" % t)
        
        # Constraint 2: Positive relation requires positive event arguments
        for r, cr in enumerate(self.event_heads):
            for c in range(self.Mc-1):
                self.model.addConstr(self.rel_left_ent(var_table_e, var_table_r, cr, r, c) >= 0, "c2_%s_%s" % (r, c))
                self.model.addConstr(self.rel_right_ent(var_table_e, var_table_r, cr, r, c) >= 0, "c3_%s_%s" % (r, c))
            if c == self.Mc-1:
                self.model.addConstr(self.rel_ent_sum(var_table_e, var_table_r, cr, r, c) >= 0, "c4_%s_%s" % (r, c))
        '''     
    def run(self):
        try:
            # Define variables                                                                                      
            var_table_r = self.define_vars()

            # Set objective                                                                                         
            self.model.setObjective(self.objective(var_table_r, self.probs), GRB.MAXIMIZE)

            # Define constrains                                                                                     
            self.define_constraints(var_table_r)

            # run model                                                                                             
            self.model.setParam('OutputFlag', False)
            self.model.optimize()

        except GurobiError:
            print('Error reported')


    def predict(self):
        rel_count = 0

        for i, v in enumerate(self.model.getVars()):
            # sample idx                                                                                            
            s_idx = int(v.varName.split('_')[1])
            # sample class index                                                                                    
            c_idx = int(v.varName.split('_')[2])

            if v.x == 1.0 and self.pred_rel_labels[s_idx] != c_idx:
                self.pred_rel_labels[s_idx] = c_idx
                rel_count += 1

        return rel_count
