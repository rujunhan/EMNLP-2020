from gurobi_inference import Event_Inference
from collections import Counter, OrderedDict

label_map_rel = {'BEFORE': 0, 'AFTER': 1, 'OVERLAP': 2}

# change label overlap --> simultaneous for TempEval metric                                                         
idx2label_rel = {0: 'BEFORE', 1: 'AFTER', 2: 'SIMULTANEOUS'}

# 'O' needs to have the o index to make the data proccessing code work                                              
label_map_evt = {'O': 0,
                 'B-CLINICAL_DEPT': 1, 'I-CLINICAL_DEPT': 2,
                 'B-EVIDENTIAL': 3, 'I-EVIDENTIAL': 4,
                 'B-OCCURRENCE': 5, 'I-OCCURRENCE': 6,
                 'B-PROBLEM': 7, 'I-PROBLEM': 8,
                 'B-TEST': 9, 'I-TEST': 10,
                 'B-TREATMENT': 11, 'I-TREATMENT': 12}
reverse_map_rel = {0: 1, 1: 0, 2: 2, 3: 3}

idx2label = {v:k for k,v in label_map_evt.items()}

def define_event_constraints():
    event_constraints = {(label_map_evt["B-OCCURRENCE"], label_map_evt["I-OCCURRENCE"]): 0.17273132333283825,
                         (label_map_evt["B-TEST"], label_map_evt["I-TEST"]): 0.33225366518922606,
                         (label_map_evt["B-PROBLEM"], label_map_evt["I-PROBLEM"]): 0.37651209677419356,
                         (label_map_evt["B-TREATMENT"], label_map_evt["I-TREATMENT"]): 0.23704435660957401,
                         (label_map_evt["B-CLINICAL_DEPT"], label_map_evt["I-CLINICAL_DEPT"]): 0.5534564190639759}
    return event_constraints

def update_constraint_parameters(constraint_param, constraint_update, pred_counter, constraints,
                                 tolerance=0.05, lr=1.0, constraint_type="event_pair"):

    diff_list = []
    for ic, (key, v) in enumerate(constraint_param.items()):
        if constraint_type == "event_pair":
            pair_sum = sum([v for k, v in pred_counter.items() if key[0] == k[0]])

        current_ratio = pred_counter[key] / pair_sum
        diff = constraints[key] - current_ratio
        #dl = diff
        # ensure lambda is always positive                                                                          
        #if diff < 0: dl = -diff                                                                                   
        #else: dl = diff                                                                                            
        diff_list.append(diff)

        if constraint_type == "event_pair":
            print(idx2label[key[0]], idx2label[key[1]], pred_counter[key],
                  pair_sum, current_ratio, diff)

        if abs(diff) <= tolerance and constraint_update[key]:
            constraint_update[key] = False

        if constraint_update[key]: constraint_param[key] += lr * diff

    return constraint_param, constraint_update


def LROptimization(data, event_constraints, lr=1.0, tolerance=0.05, max_iter=3):

    event_constraint_param = {k: 0 for k in event_constraints}
    event_constraint_update = {k:True for k in event_constraints}

    for itr in range(max_iter):

        print("="*30, " Iteration #%s " %itr, "="*30)
        evt_correction_count = 0
        event_pair_pred_counter = Counter()

        for k, prob_evt in enumerate(data):
            N, Nc = prob_evt.shape

            global_model = Event_Inference(prob_evt, event_constraint_param, event_constraints)

            global_model.run()

            evt_correction = global_model.predict()
            evt_correction_count += evt_correction

            # event global assignment                                              
            for n in range(N):
                # calculate event (consecutive) pair count                                       
                if n < N - 1:
                    for event_pair in event_constraints.keys():
                        if global_model.pred_evt_labels[n] == event_pair[0]:
                            event_pair_pred_counter[(global_model.pred_evt_labels[n]),
                                                    global_model.pred_evt_labels[n+1]] += 1
        print(event_pair_pred_counter)
        event_constraint_param, event_constraint_update = \
            update_constraint_parameters(event_constraint_param, event_constraint_update,
                                         event_pair_pred_counter, event_constraints)
        
        if all([v == False for v in event_constraint_update.values()]): break

    return event_constraint_param


def event_relation_constraints():
    distribution = OrderedDict([(('B-CLINICAL_DEPT', 'B-CLINICAL_DEPT', 'AFTER'), 0.0469),
             (('B-CLINICAL_DEPT', 'B-CLINICAL_DEPT', 'BEFORE'), 0.0939),
             (('B-CLINICAL_DEPT', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.8592),
             (('B-CLINICAL_DEPT', 'B-EVIDENTIAL', 'AFTER'), 0.1714),
             (('B-CLINICAL_DEPT', 'B-EVIDENTIAL', 'OVERLAP'), 0.8286),
             (('B-CLINICAL_DEPT', 'B-OCCURRENCE', 'AFTER'), 0.1525),
             (('B-CLINICAL_DEPT', 'B-OCCURRENCE', 'BEFORE'), 0.322),
             (('B-CLINICAL_DEPT', 'B-OCCURRENCE', 'OVERLAP'), 0.5254),
             (('B-CLINICAL_DEPT', 'B-PROBLEM', 'AFTER'), 0.4746),
             (('B-CLINICAL_DEPT', 'B-PROBLEM', 'BEFORE'), 0.029),
             (('B-CLINICAL_DEPT', 'B-PROBLEM', 'OVERLAP'), 0.4964),
             (('B-CLINICAL_DEPT', 'B-TEST', 'AFTER'), 0.0137),
             (('B-CLINICAL_DEPT', 'B-TEST', 'BEFORE'), 0.0519),
             (('B-CLINICAL_DEPT', 'B-TEST', 'OVERLAP'), 0.9344),
             (('B-CLINICAL_DEPT', 'B-TREATMENT', 'AFTER'), 0.0631),
             (('B-CLINICAL_DEPT', 'B-TREATMENT', 'BEFORE'), 0.1874),
             (('B-CLINICAL_DEPT', 'B-TREATMENT', 'OVERLAP'), 0.7495),
             (('B-EVIDENTIAL', 'B-CLINICAL_DEPT', 'BEFORE'), 0.6486),
             (('B-EVIDENTIAL', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.3514),
             (('B-EVIDENTIAL', 'B-EVIDENTIAL', 'BEFORE'), 0.0776),
             (('B-EVIDENTIAL', 'B-EVIDENTIAL', 'OVERLAP'), 0.9224),
             (('B-EVIDENTIAL', 'B-OCCURRENCE', 'AFTER'), 0.3263),
             (('B-EVIDENTIAL', 'B-OCCURRENCE', 'BEFORE'), 0.1895),
             (('B-EVIDENTIAL', 'B-OCCURRENCE', 'OVERLAP'), 0.4842),
             (('B-EVIDENTIAL', 'B-PROBLEM', 'AFTER'), 0.7703),
             (('B-EVIDENTIAL', 'B-PROBLEM', 'BEFORE'), 0.0037),
             (('B-EVIDENTIAL', 'B-PROBLEM', 'OVERLAP'), 0.226),
             (('B-EVIDENTIAL', 'B-TEST', 'AFTER'), 0.3197),
             (('B-EVIDENTIAL', 'B-TEST', 'BEFORE'), 0.2131),
             (('B-EVIDENTIAL', 'B-TEST', 'OVERLAP'), 0.4672),
             (('B-EVIDENTIAL', 'B-TREATMENT', 'AFTER'), 0.3119),
             (('B-EVIDENTIAL', 'B-TREATMENT', 'BEFORE'), 0.4037),
             (('B-EVIDENTIAL', 'B-TREATMENT', 'OVERLAP'), 0.2844),
             (('B-OCCURRENCE', 'B-CLINICAL_DEPT', 'AFTER'), 0.1037),
             (('B-OCCURRENCE', 'B-CLINICAL_DEPT', 'BEFORE'), 0.7125),
             (('B-OCCURRENCE', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.1838),
             (('B-OCCURRENCE', 'B-EVIDENTIAL', 'AFTER'), 0.0778),
             (('B-OCCURRENCE', 'B-EVIDENTIAL', 'BEFORE'), 0.2556),
             (('B-OCCURRENCE', 'B-EVIDENTIAL', 'OVERLAP'), 0.6667),
             (('B-OCCURRENCE', 'B-OCCURRENCE', 'AFTER'), 0.0905),
             (('B-OCCURRENCE', 'B-OCCURRENCE', 'BEFORE'), 0.2724),
             (('B-OCCURRENCE', 'B-OCCURRENCE', 'OVERLAP'), 0.637),
             (('B-OCCURRENCE', 'B-PROBLEM', 'AFTER'), 0.4511),
             (('B-OCCURRENCE', 'B-PROBLEM', 'BEFORE'), 0.0946),
             (('B-OCCURRENCE', 'B-PROBLEM', 'OVERLAP'), 0.4543),
             (('B-OCCURRENCE', 'B-TEST', 'AFTER'), 0.1437),
             (('B-OCCURRENCE', 'B-TEST', 'BEFORE'), 0.2552),
             (('B-OCCURRENCE', 'B-TEST', 'OVERLAP'), 0.6011),
             (('B-OCCURRENCE', 'B-TREATMENT', 'AFTER'), 0.2755),
             (('B-OCCURRENCE', 'B-TREATMENT', 'BEFORE'), 0.3686),
             (('B-OCCURRENCE', 'B-TREATMENT', 'OVERLAP'), 0.3559),
             (('B-PROBLEM', 'B-CLINICAL_DEPT', 'AFTER'), 0.0202),
             (('B-PROBLEM', 'B-CLINICAL_DEPT', 'BEFORE'), 0.404),
             (('B-PROBLEM', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.5758),
             (('B-PROBLEM', 'B-EVIDENTIAL', 'AFTER'), 0.0448),
             (('B-PROBLEM', 'B-EVIDENTIAL', 'BEFORE'), 0.4627),
             (('B-PROBLEM', 'B-EVIDENTIAL', 'OVERLAP'), 0.4925),
             (('B-PROBLEM', 'B-OCCURRENCE', 'AFTER'), 0.0774),
             (('B-PROBLEM', 'B-OCCURRENCE', 'BEFORE'), 0.4152),
             (('B-PROBLEM', 'B-OCCURRENCE', 'OVERLAP'), 0.5073),
             (('B-PROBLEM', 'B-PROBLEM', 'AFTER'), 0.065),
             (('B-PROBLEM', 'B-PROBLEM', 'BEFORE'), 0.0355),
             (('B-PROBLEM', 'B-PROBLEM', 'OVERLAP'), 0.8994),
             (('B-PROBLEM', 'B-TEST', 'AFTER'), 0.0272),
             (('B-PROBLEM', 'B-TEST', 'BEFORE'), 0.3502),
             (('B-PROBLEM', 'B-TEST', 'OVERLAP'), 0.6226),
             (('B-PROBLEM', 'B-TREATMENT', 'AFTER'), 0.1424),
             (('B-PROBLEM', 'B-TREATMENT', 'BEFORE'), 0.5093),
             (('B-PROBLEM', 'B-TREATMENT', 'OVERLAP'), 0.3483),
             (('B-TEST', 'B-CLINICAL_DEPT', 'AFTER'), 0.0221),
             (('B-TEST', 'B-CLINICAL_DEPT', 'BEFORE'), 0.1691),
             (('B-TEST', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.8088),
             (('B-TEST', 'B-EVIDENTIAL', 'BEFORE'), 0.0075),
             (('B-TEST', 'B-EVIDENTIAL', 'OVERLAP'), 0.9925),
             (('B-TEST', 'B-OCCURRENCE', 'AFTER'), 0.0883),
             (('B-TEST', 'B-OCCURRENCE', 'BEFORE'), 0.212),
             (('B-TEST', 'B-OCCURRENCE', 'OVERLAP'), 0.6996),
             (('B-TEST', 'B-PROBLEM', 'AFTER'), 0.6272),
             (('B-TEST', 'B-PROBLEM', 'BEFORE'), 0.0188),
             (('B-TEST', 'B-PROBLEM', 'OVERLAP'), 0.3541),
             (('B-TEST', 'B-TEST', 'AFTER'), 0.0271),
             (('B-TEST', 'B-TEST', 'BEFORE'), 0.0721),
             (('B-TEST', 'B-TEST', 'OVERLAP'), 0.9008),
             (('B-TEST', 'B-TREATMENT', 'AFTER'), 0.1502),
             (('B-TEST', 'B-TREATMENT', 'BEFORE'), 0.3047),
             (('B-TEST', 'B-TREATMENT', 'OVERLAP'), 0.5451),
             (('B-TREATMENT', 'B-CLINICAL_DEPT', 'AFTER'), 0.085),
             (('B-TREATMENT', 'B-CLINICAL_DEPT', 'BEFORE'), 0.21),
             (('B-TREATMENT', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.705),
             (('B-TREATMENT', 'B-EVIDENTIAL', 'AFTER'), 0.2),
             (('B-TREATMENT', 'B-EVIDENTIAL', 'BEFORE'), 0.4333),
             (('B-TREATMENT', 'B-EVIDENTIAL', 'OVERLAP'), 0.3667),
             (('B-TREATMENT', 'B-OCCURRENCE', 'AFTER'), 0.0921),
             (('B-TREATMENT', 'B-OCCURRENCE', 'BEFORE'), 0.511),
             (('B-TREATMENT', 'B-OCCURRENCE', 'OVERLAP'), 0.3969),
             (('B-TREATMENT', 'B-PROBLEM', 'AFTER'), 0.4574),
             (('B-TREATMENT', 'B-PROBLEM', 'BEFORE'), 0.1436),
             (('B-TREATMENT', 'B-PROBLEM', 'OVERLAP'), 0.399),
             (('B-TREATMENT', 'B-TEST', 'AFTER'), 0.1023),
             (('B-TREATMENT', 'B-TEST', 'BEFORE'), 0.4211),
             (('B-TREATMENT', 'B-TEST', 'OVERLAP'), 0.4766),
             (('B-TREATMENT', 'B-TREATMENT', 'AFTER'), 0.0416),
             (('B-TREATMENT', 'B-TREATMENT', 'BEFORE'), 0.1341),
             (('B-TREATMENT', 'B-TREATMENT', 'OVERLAP'), 0.8243)])
    return distribution

    include_pairs = [('B-TREATMENT', 'B-OCCURRENCE', 'OVERLAP')]
    constraints = OrderedDict()
    for k, v in distribution.items():
        new_key = (label_map_evt[k[0]], label_map_evt[k[1]], label_map_rel[k[2]])
        if k in include_pairs:
            constraints[new_key] = v
        
    return constraints
