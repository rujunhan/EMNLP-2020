from gurobi_inference import Relation_Inference
from collections import Counter
import copy

def calculate_prob(class_counts, threshold=100):
    total_counts = sum(class_counts)
    if total_counts < threshold: return [0.0]*len(class_counts)
    else: return [x / total_counts for x in class_counts]
    
def define_prior_constraints(prior_dist, relation_dictionaries, label_map_rel, threshold=100):
    constraints = {}
    for x in prior_dist:
        if len(x) == 0: continue
        event1, event2 = x
        class_counts = [relation_dictionaries[0][(event1, event2)],                                                
                        relation_dictionaries[1][(event1, event2)],                                                
                        relation_dictionaries[2][(event1, event2)],                                                
                        relation_dictionaries[3][(event1, event2)],                                                
                        relation_dictionaries[4][(event1, event2)],                                                
                        relation_dictionaries[5][(event1, event2)]] 
        prior = calculate_prob(class_counts, threshold)
        if any(prior):
            for l, v in label_map_rel.items():
                if v == label_map_rel['NONE']: continue
                constraints[(event1, event2, l)] = prior[v]
                
    return constraints

def update_constraint_parameters(constraint_param, constraint_update, relation_pred_counter,
                                 constraints, lr, tolerance, num_meet_tolerance, label_map_rel):

    idx2label_rel = {v:k for k,v in label_map_rel.items()}
    diff_list = []
    prev_constraints = copy.deepcopy(constraint_param)
    
    for ic, ((e1, e2, r), v) in enumerate(constraint_param.items()):
        
        pair_rel_sum = sum([relation_pred_counter[(e1, e2, i)] for i in range(len(label_map_rel))])
        current_ratio = relation_pred_counter[(e1, e2, r)] / pair_rel_sum

        diff = constraints[(e1, e2, r)] - current_ratio
        diff_list.append(((e1, e2, r), diff))

        #print(e1, e2, idx2label_rel[r], relation_pred_counter[(e1, e2, r)],
        #      pair_rel_sum, current_ratio, diff)

        if abs(diff) <= tolerance and constraint_update[(e1, e2, r)]:
            num_meet_tolerance += 1
            constraint_update[(e1, e2, r)] = False

        if constraint_update[(e1, e2, r)]:
            constraint_param[(e1, e2, r)] += lr * diff

    print("Constraints", prev_constraints)
    print("Diff", {k: round(x, 4) for k, x in diff_list})
    print("num_meet_tolerance: %d" % num_meet_tolerance)
        
    return constraint_param, constraint_update, num_meet_tolerance

def LROptimization(data, sample_event_types, labels, all_pairs, event_heads, label_map_rel,
                   constraints, lr=5.0, tolerance=0.10, decay=0.9, max_iter=20):

    constraints = {(k[0], k[1], label_map_rel[k[2]]):v for k,v in constraints.items()}

    constraint_param = {k: 0 for k in constraints}
    constraint_update = {k: True for k in constraints}
    
    num_meet_tolerance = 0
    
    for itr in range(max_iter):

        print("="*30, " Iteration #%s " %itr, "="*30)

        correction_count = 0
        relation_pred_counter = Counter()
        constraint_perf = {k: [0, 0, 0, 0] for k in constraints}
        
        M, Mc = data.shape
        
        global_model = Relation_Inference(data, constraint_param, constraints, sample_event_types,
                                          all_pairs, label_map_rel, event_heads)
        
        global_model.run()

        correction = global_model.predict()
        correction_count += correction

        print("Total relation corrections: %s" % correction_count)
        preds = []
        # relation global assignment                                                                               
        for m in range(M):
            rel_pred = global_model.pred_rel_labels[m]
            preds.append(rel_pred)

            label = labels[m] 
            
            sample_event_type = sample_event_types[m]
            if len(sample_event_type) == 0: continue

            key = sample_event_types[m]
            
            # merge Vague and None for ratio computation
            if rel_pred == label_map_rel['NONE']: rel_pred = 0
            if labels[m] == label_map_rel['NONE']: label = 0
            
            relation_pred_counter[(key[0], key[1], rel_pred)] += 1

            for k in constraint_perf:
                if k ==	(key[0], key[1], label):
                    constraint_perf[k][0] += 1
                    if rel_pred == label:
                        constraint_perf[k][1] += 1
                        
                if k == (key[0], key[1], rel_pred):
                    constraint_perf[k][2] += 1
                    if rel_pred == label:
                        constraint_perf[k][3] += 1
        
        constraint_param, constraint_update, num_meet_tolerance = \
            update_constraint_parameters(constraint_param, constraint_update, relation_pred_counter,
                                         constraints, lr, tolerance, num_meet_tolerance, label_map_rel)
        #print(constraint_perf)
        for triplet, perf in constraint_perf.items():
            recall, precision, f1 = 0.0, 0.0, 0.0
            if perf[0] > 0: recall = perf[1] / perf[0]
            if perf[2] > 0: precision = perf[3] / perf[2]
            if recall + precision > 0.0: f1 = 2*recall*precision / (recall+precision)
            #print("(%s, %s, %s): recall: %.4f; precision: %.4f; f1: %.4f" % (triplet[0], triplet[1], triplet[2],
            #                                                                 recall, precision, f1))
        lr *= decay
        if num_meet_tolerance == len(constraints): break
        
    return preds, num_meet_tolerance
