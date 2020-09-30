from typing import Iterator, List, Mapping, Union, Optional, Set
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime
import json
from scipy import stats
import numpy as np
import pickle
from temporal_evaluation_adapted import evaluate_two_files
import re
import os
import argparse
from tlinkEvaluation import tlinkEvaluation

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


def event_partial_match(evt1, type1, evt2, type2):
    assert len(evt1) == len(type1)
    assert len(evt2) == len(type2)
    
    span_match, type_match = [0]*len(evt1), [0]*len(evt1)
    for i, idx in enumerate(evt1):
        for idx2, t2 in zip(evt2, type2):
            if partial_match(idx, idx2):
                span_match[i] = 1
                if type1[i] == t2:
                    type_match[i] = 1
    return sum(span_match), sum(type_match)
        
        
def compare_events(events1, types1, events2, types2, mapping='partial'):
    
    count = sum([len(events) for events in events1])
    
    if count == 0: return 0, 0, count

    span_match, type_match = 0, 0
    for evt1, type1, evt2, type2 in zip(events1, types1, events2, types2):
        if mapping == 'exact':
            results = event_exact_match(evt1, type1, evt2, type2)
            span_match += results[0]
            type_match += results[1]
        else:
            results = event_partial_match(evt1, type1, evt2, type2)
            span_match += results[0]
            type_match += results[1]
    return span_match, type_match, count


def construct_events_relaxed(token_pred, orig2token):
    batch_events, batch_types = [], []
    for b, o2t in enumerate(orig2token):
        events, types = [], []
        event_start, event_end = 0, 0
        for k, i in enumerate(o2t):
            if token_pred[b][i] % 2 == 1:
                # case 1-a: find a new start                                                                       
                if event_start == 0: event_start = i
                # case 1-b: find a end token (by looking a new start token)                              
                else:
                    event_end = o2t[k-1] # find previous end's head token
                    events.append((event_start, event_end))
                    types.append(token_pred[b][event_start])
                    event_start, event_end = i, 0

            elif token_pred[b][i] % 2 == 0 and token_pred[b][i] > 0:
                # case 2-a: find a new start                                                                       
                if event_start == 0: event_start = i
                # case 2-b                                                                                         
                elif not (token_pred[b][i] == token_pred[b][event_start] or
                          token_pred[b][i] == token_pred[b][event_start] + 1):
                    event_end = o2t[k-1] # find previous end's head token 
                    events.append((event_start, event_end))
                    types.append(token_pred[b][event_start])
                    event_start, event_end = i, 0
                    
            #case 3: find event ending for multi-token event
            elif event_start > 0 and token_pred[b][i] == 0:
                event_end = o2t[k-1] # find current end's head token
                events.append((event_start, event_end))
                types.append(token_pred[b][event_start])
                event_start, event_end = 0, 0
                    
        # case 4: last word or phrase is an event
        if event_start > 0:
            events.append((event_start, i))
            types.append(token_pred[b][event_start])
        batch_events.append(events)
        batch_types.append(types)
    return batch_events, batch_types

def construct_events(token_pred, orig2token):
    batch_events, batch_types = [], [] 
    for b, o2t in enumerate(orig2token):
        events, types = [], []
        event_start, event_end = 0, 0
        for k, i in enumerate(o2t):
            if token_pred[b][i] % 2 == 1:
                # case 1: find a new start                                                                         
                if event_start == 0: event_start = i
                # case 2-a: single token event                                                                     
                else:
                    event_end = o2t[k-1] # find previous end's head token                               
                    events.append((event_start, event_end))
                    types.append(token_pred[b][event_start])
                    event_start, event_end = i, 0
            #case 2-b: find event ending for multi-token event                                                 
            elif event_start > 0 and (token_pred[b][i] == 0 or
                                      token_pred[b][i] != token_pred[b][event_start] + 1):
                event_end = o2t[k-1] # find current end's head token
                events.append((event_start, event_end))
                types.append(token_pred[b][event_start])
                event_start, event_end = 0, 0
        # case 3: last word or phrase is an event                   
        if event_start > 0:
            events.append((event_start, i))
            types.append(token_pred[b][event_start])
        batch_events.append(events)
        batch_types.append(types)
        
    return batch_events, batch_types
    

def update_relation_agg(temp_preds_rel, relation_agg_gold, relation_agg_pred):
    for k, v in temp_preds_rel.items():
        # gold label has to be unqiue                                                                   
        gold_label = list(set([vv[0] for vv in v]))
        assert len(gold_label) == 1

        pred_labels = [vv[1] for vv in v]
        if k in relation_agg_gold:
            assert relation_agg_gold[k] == gold_label[0]
            relation_agg_pred[k] += pred_labels
        else:
            relation_agg_gold[k] = gold_label[0]
            relation_agg_pred[k] = pred_labels
    return relation_agg_gold, relation_agg_pred

def collect_batch_event_prob(all_prob_evt, orig2token):
    batch_prob_evt = []
    for ib, o2t in enumerate(orig2token):
        prob_evt = [all_prob_evt[ib][o][:] for o in o2t]
        batch_prob_evt.append(np.asarray(prob_evt))
    return list(batch_prob_evt)

def key2eventID(key):
    return "%s\t%s\t%s" % (key[0], '#'.join(key[1:3]), '#'.join(key[3:])) 
    
def eval_all_files(relation_agg_gold, relation_agg_pred):
    
    all_files = list(set([k[0] for k in relation_agg_gold.keys()]))

    print("total %s files to evaluate" % len(all_files))
    
    pred_cnt, gold_cnt, pred_matched, gold_matched = 0, 0, 0, 0
    for fl in all_files:
        #print("---------------- evaluate %s ------------------" % fl)
        gold_text = '\n'.join([key2eventID(k) + '\t' + idx2label_rel[v]
                             for k,v in relation_agg_gold.items()
                             if k[0] == fl])
        pred_text = '\n'.join([key2eventID(k) + '\t' + idx2label_rel[stats.mode(v)[0][0]]
                             for k,v in relation_agg_pred.items()
                             if k[0] == fl and stats.mode(v)[0][0] >= 0])
        #print(gold_text)
        #print(pred_text)
        scores = evaluate_two_files(gold_text, pred_text)
        #print(scores)
        pred_cnt += scores[0]
        gold_cnt += scores[1]
        pred_matched += scores[2]
        gold_matched += scores[3]
        
    print(pred_cnt, gold_cnt, pred_matched, gold_matched)
    
    prec = pred_matched / pred_cnt if pred_cnt > 0 else 0.0
    recall = gold_matched / gold_cnt if gold_cnt > 0 else 0.0

    if prec + recall > 0.0: return prec, recall, 2 * prec * recall / (prec + recall)
    return prec, recall, 0.0


def eval_all_files_gold(relation_agg_gold, relation_agg_pred):

    all_files = list(set([k[0] for k in relation_agg_gold.keys()]))

    relation_agg_pred = {k:[vv[0] for vv in v] for k,v in relation_agg_pred.items()}
    
    print("total %s files to evaluate" % len(all_files))

    pred_cnt, gold_cnt, pred_matched, gold_matched = 0, 0, 0, 0
    for fl in all_files:
        #print("---------------- evaluate %s ------------------" % fl)                                                    
        gold_text = '\n'.join([key2eventID(k) + '\t' + idx2label_rel[v]
                             for k,v in relation_agg_gold.items()
                             if k[0] == fl])
        pred_text = '\n'.join([key2eventID(k) + '\t' + idx2label_rel[stats.mode(v)[0][0]]
                             for k,v in relation_agg_pred.items()
                             if k[0] == fl and stats.mode(v)[0][0] >= 0])
        
        scores = evaluate_two_files(gold_text, pred_text)
        pred_cnt += scores[0]
        gold_cnt += scores[1]
        pred_matched += scores[2]
        gold_matched += scores[3]

    #print(pred_cnt, gold_cnt, pred_matched, gold_matched)

    prec = pred_matched / pred_cnt if pred_cnt > 0 else 0.0
    recall = gold_matched / gold_cnt if gold_cnt > 0 else 0.0

    if prec + recall > 0.0: return prec, recall, 2 * prec * recall / (prec + recall)
    return prec, recall, 0.0

def find_rel_evt_type(rkey, event_agg_gold):
    # input a relation key, map its arguments to idx of relation types
    
    e1t = event_agg_gold[(rkey[0], int(rkey[1].split(':')[0]), int(rkey[1].split(':')[1]))]
    e2t = event_agg_gold[(rkey[0], int(rkey[3].split(':')[0]), int(rkey[3].split(':')[1]))]
    return e1t, e2t

def reshape_to_batch(x, lengths, orig2token=[], i=0):
    # reshape flatten list to a batch of lists
    # if orig2token map has content, then map
    # tokenized idx to original idx
    
    reshaped = []
    idx = 0
    for il, l in enumerate(lengths):
        if orig2token:
            reshaped.append([(orig2token[i+il].index(xx[0]),
                              orig2token[i+il].index(xx[1]),
                              orig2token[i+il].index(xx[2]),
                              orig2token[i+il].index(xx[3]))
                             for xx in x[idx:idx+l]])
        else:
            reshaped.append(x[idx:idx+l])
        idx += l
    return reshaped

def load_data(data_dir, filename):
    print("load %s..." % filename)
    with open("%s%s.json" % (data_dir, filename), "r") as read_file:
        return json.load(read_file)

def select_field(data, field):
    # collect a list of field in data      
    # fields: 'label', 'offset', 'input_ids, 'mask_ids', 'segment_ids', 'question_id'
    return [ex[field] for ex in data]

def calculate_f1(n_corr, n_pred, n_gold):
    print("Corr: %s; Pred: %s, Gold: %s" % (n_corr, n_pred, n_gold))
    
    prec = float(n_corr) / n_pred if n_pred > 0 else 0.0
    recall = float(n_corr) / n_gold if n_gold > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0.0 else 0.0
    
    return f1

def mask_negative(rel_labels, p):
    # given a list of relation labels
    # return a list of random mask for negative class with chance of p
    mask = [np.random.choice([0, 1], size=(1), p=[p, 1-p])[0] if r > 2 else 1
            for r in rel_labels]
    assert len(mask) == len(rel_labels)
    return mask

def partial_match(indicesA, indicesB):
    if indicesA[0] > indicesB[0]:
        indicesA, indicesB = indicesB, indicesA
    if indicesA[1] >= indicesB[0]: return True
    return False

def is_correct(pair, pred, gold_indices, labels):        
    # correction prediction has to fulfil two conditions                                         
    # 1. event head match                           
    # 2. relation is correct
    for i, idx in enumerate(gold_indices):
        if pair[0] == idx[0] and pair[2] == idx[2] and pred == labels[i] < 3:
            return 1
        if pair[0] == idx[2] and pair[2] == idx[0] and pred == reverse_map_rel[labels[i]] < 3:
            return 1
    return 0

def is_correct_relaxed(pair, pred, gold_indices, labels):
    # relaxed correction prediction has to fulfil two conditions                                                      
    # 1. event partial match
    # 2. relation is correct and positive class (before, after, overlap) 
    for i, idx in enumerate(gold_indices):
        if partial_match((pair[0], pair[1]), (idx[0], idx[1])) and \
           partial_match((pair[2], pair[3]), (idx[2], idx[3])) and pred == labels[i] < 3:
            return 1
        if partial_match((pair[0], pair[1]), (idx[2], idx[3])) and \
           partial_match((pair[2], pair[3]), (idx[0], idx[1])) and pred == reverse_map_rel[labels[i]] < 3:
            return 1
    return 0

def is_partial_match(pair, gold_indices):
    for i, idx in enumerate(gold_indices):
        if partial_match((pair[0], pair[1]), (idx[0], idx[1])) and \
           partial_match((pair[2], pair[3]), (idx[2], idx[3])):
            return 1
        if partial_match((pair[0], pair[1]), (idx[2], idx[3])) and \
           partial_match((pair[2], pair[3]), (idx[0], idx[1])):
            return 1
    return 0

def eval_end2end_result(rel_pred, rel_gold, pred_indices, gold_indices):

    num_gold = sum([x < 3 for batch in rel_gold for x in batch])
    if len(rel_pred) == 0: return 0, 0, 0, num_gold
    num_pred, num_corr, num_corr_relaxed = 0, 0, 0
    for b, pairs in enumerate(pred_indices):
        for pair, pred in zip(pairs, rel_pred[b]):
            num_pred += is_partial_match(pair, gold_indices[b])
            num_corr += is_correct(pair, pred, gold_indices[b], rel_gold[b])
            num_corr_relaxed += is_correct_relaxed(pair, pred, gold_indices[b], rel_gold[b])
    return num_corr, num_corr_relaxed, num_pred, num_gold


def event_head(token_list, idx):
    first = token_list[idx[0]]
    if first > 0 and first % 2 == 0: first -= 1

    second = token_list[idx[2]]
    if second > 0 and second % 2 == 0: second -= 1

    return first, second

def eval_rel_constraint_perf(rel_pred, rel_gold, pred_indices, gold_indices,
                             evt_pred, evt_gold, constraint_perf):
    # evaluate performance per constraint
    assert len(rel_pred) == len(pred_indices)
    assert len(rel_gold) == len(gold_indices)
    # gold triplet count
    for gold, gold_idx in zip(rel_gold, gold_indices):
        # event type
        first, second = event_head(evt_gold, gold_idx)
        if (first, second, gold) in constraint_perf:
            constraint_perf[(first, second, gold)][0] += 1
            # matched triplet count
            if is_correct_relaxed(gold_idx, gold, pred_indices, rel_pred):
                constraint_perf[(first, second, gold)][1] += 1
                
    # pred triplet count
    for pred, pred_idx in zip(rel_pred, pred_indices):
        first, second = event_head(evt_pred, pred_idx)
        if (first, second, pred) in constraint_perf:
            constraint_perf[(first, second, pred)][2] += 1
            # correct pred triplet count   
            if is_correct_relaxed(pred_idx, pred, gold_indices, rel_gold):
                constraint_perf[(first, second, pred)][3] += 1
            
    return constraint_perf

def find_gold_match(pair, indices, preds):
    # given a gold pair, check it's prediction
    for i, idx in enumerate(indices):
        if partial_match((pair[0], pair[1]), (idx[0], idx[1])) and \
           partial_match((pair[2], pair[3]), (idx[2], idx[3])):
            return preds[i]
        if partial_match((pair[0], pair[1]), (idx[2], idx[3])) and \
           partial_match((pair[2], pair[3]), (idx[0], idx[1])):
            if preds[i] == 0: return 1
            if preds[i] == 1: return 0
            return preds[i]
    # couldn't find an prediction for gold
    return -1
    
def eval_end2end_gold_perf(rel_pred, rel_gold, pred_indices, gold_indices,
                           all_instance_keys, relations, batch_idx, exclude_label=[3]):
    res = defaultdict(list)
    for b, pairs in enumerate(gold_indices):
        doc = all_instance_keys[batch_idx + b].split('_')[0]
        for i, (pair, gold) in enumerate(zip(pairs, rel_gold[b])):
            pred = find_gold_match(pair, pred_indices[b], rel_pred[b])
            rkey = relations[batch_idx + b][i]['orig']
            if gold not in exclude_label:
                res[(doc, rkey[0][0], rkey[0][1], rkey[1][0], rkey[1][1])].append((gold, pred))
    return res

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def exact_match(question_ids, labels, predictions):
    em = defaultdict(list)
    for q, l, p in zip(question_ids, labels, predictions):
        em[q].append(l == p)
    print("Total %s questions" % len(em))
    return float(sum([all(v) for v in em.values()])) / float(len(em))

def rel_inputs(relations):
    # flatten relations and use batch length to map back to the original input 
    indices = [(m[r['event1_start']], m[r['event1_end']], m[r['event2_start']], m[r['event2_end']])
                for rel, m in relations for r in rel if r['label']]
    rel_labels = [label_map_rel[r['label']] for rel, m in relations for r in rel if r['label']]
    lengths = [len([r for r in rel if r['label']]) for rel, m in relations]

    return indices, rel_labels, lengths

def save_results(outfile, filename, save_dir):
    output = open(save_dir + filename, 'wb')
    pickle.dump(outfile, output)
    output.close()
    return

def output_errors(lookup, labels, preds, pred_score, pred_inference, filename):
    assert len(labels) == len(preds)
    assert len(lookup) == len(preds)

    if len(pred_inference) > 0:
        assert len(labels) == len(pred_inference)
        outfile = open("./logs/i2b2_all_train_unlabeled_inference_%s.tsv" % filename, 'w')
        outfile.write("ID\tLabel\tPrediction\tInference\tScore\n")
        for key, l, p, pi, s in zip(list(lookup.keys()), labels, preds, pred_inference, pred_score):
            outfile.write("%s\t%s\t%s\t%s\t%.4f\n" % (key, l, p, pi, s))
        outfile.close()
    else:
        outfile = open("./logs/i2b2_all_test_%s.tsv" % filename, 'w')
        outfile.write("ID\tLabel\tPrediction\tScore\n")
        for key, l, p in zip(list(lookup.keys()), labels, preds):
            outfile.write("%s\t%s\t%s\n" % (key, l, p))
        outfile.close()
    return


def convert_to_features_instance(data, tokenizer, label_map_evt, max_length=220, evaluation=False):
    # This is use only for the baseline i.e. BERT-base without fine-tuning
    label_map_evt = {'O': 0, 'B':1}
    event_types = {'O': 0,
                   'B-CLINICAL_DEPT': 1, 'I-CLINICAL_DEPT': 2,
                   'B-EVIDENTIAL': 3, 'I-EVIDENTIAL': 4,
                   'B-OCCURRENCE': 5, 'I-OCCURRENCE': 6,
                   'B-PROBLEM': 7, 'I-PROBLEM': 8,
                   'B-TEST': 9, 'I-TEST': 10,
                   'B-TREATMENT': 11, 'I-TREATMENT': 12}
    event_types = {'O': 0,
                   'B-CLINICAL_DEPT': 1, 'I-CLINICAL_DEPT': 2,
                   'B-EVIDENTIAL': 3, 'I-EVIDENTIAL': 4,
                   'B-OCCURRENCE': 5, 'I-OCCURRENCE': 6,
                   'B-PROBLEM': 7, 'I-PROBLEM': 8,
                   'B-TEST': 9, 'I-TEST': 10,
                   'B-TREATMENT': 11, 'I-TREATMENT': 12,
                   'B-DATE': 13, 'I-DATE':14,
                   'B-TIME':15, 'I-TIME':16,
                   'B-ADMISSION':17, 'I-ADMISSION':18,
                   'B-DISCHARGE':19, 'I-DISCHARGE':20,
                   'B-DURATION': 21, 'I-DURATION':22,
                   'B-FREQUENCY':23, 'I-FREQUENCY':24
    }
    
    samples = []
    counter, empty_tokens = 0, 0
    max_len_global = 0 # to show global max_len without truncating
    for k, v in data.items():
        segment_ids = []
        # the following bert tokenized context starts / end with ['SEP']        
        new_tokens = ["[CLS]"]
        orig_to_tok_map = []
        new_evt_labels = ['O']
        
        assert len(v['context']) == len(v['event_labels'])
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))

            # for some rare case, token is empty, i.e. ''                                                      
            # add a placeholder to ensure correct indexing                                                     
            if len(token) == 0:
                empty_tokens += 1
                temp_tokens = [' ']
            else:
                temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
            new_evt_labels.extend([v['event_labels'][i]]*len(temp_tokens))

        # do not predict event types
        evt_types = [event_types[t] for t in new_evt_labels]
        new_evt_labels = [0 if l[0] == 'O' else 1 for l in new_evt_labels]
        
        assert len(new_tokens) == len(new_evt_labels)

        new_tokens.append("[SEP]")
        new_evt_labels.append(0)
        evt_types.append(0)
        
        # orig_to_tok_map should only contain mapping of tokens in the original context                        
        assert len(orig_to_tok_map) == len(v['context'])

        tokenized_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]
            new_evt_labels =  new_evt_labels[:-(len(tokenized_ids) - max_length + 1)] + [0]
            evt_types =  evt_types[:-(len(tokenized_ids) - max_length + 1)] + [0]
        
        segment_ids = [0] * len(tokenized_ids)
        # mask ids                                                                                             
        mask_ids = [1] * len(tokenized_ids)
        # padding                                                                                              
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length                                                               
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            new_evt_labels += padding
            evt_types += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length
        assert len(evt_types) == len(new_evt_labels)
        # construct a sample                                                                                   
        sample = {'event_labels': new_evt_labels,
                  'event_types': evt_types,
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'sample_id': k,
                  'relations': v['relations'],
                  'orig2token': orig_to_tok_map,
                  'event_keys': v['event_label_keys']}

        samples.append(sample)
        if counter < 0:
            print(new_evt_labels)
            print(segment_ids)
            print(tokenized_ids)
            print(v['event_label_keys'])
        counter += 1
        
    print("Number empty tokens: % s" % empty_tokens)
    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples
        
def convert_to_features_roberta_instance(data, tokenizer, label_map_evt, max_length=220, evaluation=False):
    # RoBERTa: <s> + Context + </s>
    samples = []
    counter, empty_tokens = 0, 0
    max_len_global = 0 # to show global max_len without truncating 
    for k, v in data.items():
        
        segment_ids = []
        new_tokens = ["<s>"]
        orig_to_tok_map = []
        
        new_evt_labels = ['O']
        assert len(v['context']) == len(v['event_labels'])
        for i, token in enumerate(v['context']):
            orig_to_tok_map.append(len(new_tokens))
            
            # for some rare case, token is empty, i.e. ''
            # add a placeholder to ensure correct indexing
            if len(token) == 0:
                empty_tokens += 1
                temp_tokens = [' ']
            else:
                temp_tokens = tokenizer.tokenize(token)
            
            new_tokens.extend(temp_tokens)
            new_evt_labels.extend([v['event_labels'][i]]*len(temp_tokens))

        new_evt_labels = [label_map_evt[l] for l in new_evt_labels]
        assert len(new_tokens) == len(new_evt_labels)

        new_tokens.append("</s>")
        new_evt_labels.append(0)

        # orig_to_tok_map should only contain mapping of tokens in the original context
        assert len(orig_to_tok_map) == len(v['context'])

        tokenized_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)
        
        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]            
            new_evt_labels =  new_evt_labels[:-(len(tokenized_ids) - max_length + 1)] + [0]

        segment_ids = [0] * len(tokenized_ids)
        # mask ids
        mask_ids = [1] * len(tokenized_ids)
        # padding
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            new_evt_labels += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length

        # construct a sample 
        sample = {'event_labels': new_evt_labels,
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'sample_id': k,
                  'relations': v['relations'],
                  'orig2token': orig_to_tok_map,
                  'event_keys': v['event_label_keys']}

        samples.append(sample)
        if counter < 0:
            print(k)
            print(v)
            print(tokenized_ids)
        counter += 1
        #if counter > 1000:
        #    break
    print("Number empty tokens: % s" % empty_tokens)
    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples
        

def convert_to_features_roberta(data, tokenizer, label_map, max_length=215, evaluation=False):
    # RoBERTa: <s> + Context + </s>
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating 
    for k, v in data.items():
        segment_ids = []
        new_tokens = ["<s>"]
        orig_to_tok_map = []

        if len(v['context']) == 1: 
            context = v['context'][0]
            prev_sent_len = 0
        else:
            context = v['context'][0] + v['context'][1]
            prev_sent_len = len(v['context'][0])

        for i, token in enumerate(context):
            orig_to_tok_map.append(len(new_tokens))
            temp_tokens = tokenizer.tokenize(token)
            new_tokens.extend(temp_tokens)
        new_tokens.append("</s>")

        assert len(orig_to_tok_map) == sum([len(c) for c in v['context']])
        orig_to_tok_map.append(new_tokens)
        assert len(orig_to_tok_map) == sum([len(c) for c in v['context']]) + 1 # account for ending ['SEP'] 

        tokenized_ids = tokenizer.convert_tokens_to_ids(new_tokens)

        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        # truncate long sequence, but we can simply set max_length > global_max                                                                       
        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]

        segment_ids = [0] * len(tokenized_ids)
        # mask ids                                                                                                                                    
        mask_ids = [1] * len(tokenized_ids)

        # padding                                                                                                                                     
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.                                                                                                     
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length
        # construct a sample                                                                                                     
        sample = {'label': label_map[v['label']],
                  'start1': orig_to_tok_map[v['event1_start']],
                  'end1': orig_to_tok_map[v['event1_end']],
                  'start2': orig_to_tok_map[v['event2_start'] + prev_sent_len],
                  'end2': orig_to_tok_map[v['event2_end'] + prev_sent_len],
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'sample_id': k}
    
        # add these fields for qualitative analysis                                                                                                   
        if evaluation:
            sample['passage'] = v['context']
        samples.append(sample)
        # check some example data                                                                                                                     
        if counter < 0:
            print(k)
            print(v)
            print(tokenized_ids)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples

def convert_to_features(data, tokenizer, label_map, max_length=200, evaluation=False):
    # each sample will either have [CLS] + Context1 + [SEP] + Context2 + [SEP]
    # or ......................... [CLS] + Context + [SEP]
    
    samples = []
    counter = 0
    max_len_global = 0 # to show global max_len without truncating                                                                             
    for k, v in data.items():
        segment_ids = []
        # the following bert tokenized context starts / end with ['SEP']                                                                       
        new_tokens = ["[CLS]"]
        orig_to_tok_map = []

        # the pair of events are in the same sentence
        if len(v['context']) == 1:
            prev_sent_len = 0 
            for i, token in enumerate(v['context'][0]):
                orig_to_tok_map.append(len(new_tokens))
                temp_tokens = tokenizer.tokenize(token)
                new_tokens.extend(temp_tokens)
            new_tokens.append("[SEP]")
            orig_to_tok_map.append(len(new_tokens))

            # following the bert convention for calculating segment ids   
            segment_ids = [0] * len(new_tokens)
 
        # the pair of events are in two different sentences
        else:
            # offset 2nd sent idx in orig_to_tok_map
            prev_sent_len = len(v['context'][0])
            for i, token in enumerate(v['context'][0]):
                orig_to_tok_map.append(len(new_tokens))
                temp_tokens = tokenizer.tokenize(token)
                new_tokens.extend(temp_tokens)
            new_tokens.append("[SEP]")
            # following the bert convention for calculating segment ids   
            segment_ids = [0] * len(new_tokens)

            for i, token in enumerate(v['context'][1]):
                orig_to_tok_map.append(len(new_tokens))
                temp_tokens = tokenizer.tokenize(token)
                new_tokens.extend(temp_tokens)
            new_tokens.append("[SEP]")
            orig_to_tok_map.append(len(new_tokens))

            # following the bert convention for calculating segment ids                                                                                        
            segment_ids += [1] * (len(new_tokens) - len(segment_ids))
        
        assert len(orig_to_tok_map) == sum([len(c) for c in v['context']]) + 1 # account for ending ['SEP'] 
        # mask ids                                                                                                                                                
        mask_ids = [1] * len(new_tokens)
        assert len(mask_ids) == len(segment_ids)

        tokenized_ids = tokenizer.convert_tokens_to_ids(new_tokens)
        assert len(tokenized_ids) == len(segment_ids)

        if len(tokenized_ids) > max_len_global:
            max_len_global = len(tokenized_ids)

        # truncate long sequence, but we can simply set max_length > global_max                                                                
        if len(tokenized_ids) > max_length:
            ending = tokenized_ids[-1]
            tokenized_ids = tokenized_ids[:-(len(tokenized_ids) - max_length + 1)] + [ending]

        # padding                                                                                                                              
        if len(tokenized_ids) < max_length:
            # Zero-pad up to the sequence length.                                                                                              
            padding = [0] * (max_length - len(tokenized_ids))
            tokenized_ids += padding
            mask_ids += padding
            segment_ids += padding
        assert len(tokenized_ids) == max_length

        # construct a sample for each QA pair                                                                                
        sample = {'label': label_map[v['label']],
                  'start1': orig_to_tok_map[v['event1_start']],
                  'end1': orig_to_tok_map[v['event1_end']],
                  'start2': orig_to_tok_map[v['event2_start'] + prev_sent_len],
                  'end2': orig_to_tok_map[v['event2_end'] + prev_sent_len],
                  'input_ids': tokenized_ids,
                  'mask_ids': mask_ids,
                  'segment_ids': segment_ids,
                  'sample_id': k}
        # add these fields for qualitative analysis
        if evaluation:
            sample['passage'] = v['context']
        samples.append(sample)

        # check some example data                                                                                                              
        if counter < 0:
            print(k)
            print(v)
            print(tokenized_ids)
        counter += 1

    print("Maximum length after tokenization is: % s" % (max_len_global))
    return samples

class ClassificationReport:
    def __init__(self, name, true_labels: List[Union[int, str]],
                 pred_labels: List[Union[int, str]]):

        assert len(true_labels) == len(pred_labels)
        self.num_tests = len(true_labels)
        self.total_truths = Counter(true_labels)
        self.total_predictions = Counter(pred_labels)
        self.name = name
        self.labels = sorted(set(true_labels) | set(pred_labels))
        self.confusion_mat = self.confusion_matrix(true_labels, pred_labels)
        self.accuracy = sum(y == y_ for y, y_ in zip(true_labels, pred_labels)) / len(true_labels)
        self.trim_label_width = 15
        self.rel_f1 = 0.0

    @staticmethod
    def confusion_matrix(true_labels: List[str], predicted_labels: List[str]) \
            -> Mapping[str, Mapping[str, int]]:
        mat = defaultdict(lambda: defaultdict(int))
        for truth, prediction in zip(true_labels, predicted_labels):
            mat[truth][prediction] += 1
        return mat

    def __repr__(self):
        res = f'Name: {self.name}\t Created: {datetime.now().isoformat()}\t'
        res += f'Total Labels: {len(self.labels)} \t Total Tests: {self.num_tests}\n'
        display_labels = [label[:self.trim_label_width] for label in self.labels]
        label_widths = [len(l) + 1 for l in display_labels]
        max_label_width = max(label_widths)
        header = [l.ljust(w) for w, l in zip(label_widths, display_labels)]
        header.insert(0, ''.ljust(max_label_width))
        res += ''.join(header) + '\n'
        for true_label, true_disp_label in zip(self.labels, display_labels):
            predictions = self.confusion_mat[true_label]
            row = [true_disp_label.ljust(max_label_width)]
            for pred_label, width in zip(self.labels, label_widths):
                row.append(str(predictions[pred_label]).ljust(width))
            res += ''.join(row) + '\n'
        res += '\n'

        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        def num_to_str(num):
            return '0' if num == 0 else str(num) if type(num) is int else f'{num:.4f}'

        n_correct = 0
        n_true = 0
        n_pred = 0

        all_scores = []
        header = ['Total  ', 'Predictions', 'Correct', 'Precision', 'Recall  ', 'F1-Measure']
        res += ''.ljust(max_label_width + 2) + '  '.join(header) + '\n'
        head_width = [len(h) for h in header]

        for label, width, display_label in zip(self.labels, label_widths, display_labels):
            if label not in ['O']:
                total_count = self.total_truths.get(label, 0)
                pred_count = self.total_predictions.get(label, 0)

                correct_count = self.confusion_mat[label][label]
                
                if label not in ['VAGUE', 'NONE']:                                                                    
                    n_true += total_count
                    n_pred += pred_count
                    n_correct += correct_count

                precision = safe_division(correct_count, pred_count)
                recall = safe_division(correct_count, total_count)
                f1_score = safe_division(2 * precision * recall, precision + recall)
                all_scores.append((precision, recall, f1_score))

                row = [total_count, pred_count, correct_count, precision, recall, f1_score]
                row = [num_to_str(cell).ljust(w) for cell, w in zip(row, head_width)]
                row.insert(0, display_label.rjust(max_label_width))
                res += '  '.join(row) + '\n'

        # weighing by the truth label's frequency                                                        
        label_weights = [safe_division(self.total_truths.get(label, 0), self.num_tests)
                         for label in self.labels if label not in ['O']]
        weighted_scores = [(w * p, w * r, w * f) for w, (p, r, f) in zip(label_weights, all_scores)]

        assert len(label_weights) == len(weighted_scores)

        res += '\n'
        res += '  '.join(['Weighted Avg'.rjust(max_label_width),
                          ''.ljust(head_width[0]),
                          ''.ljust(head_width[1]),
                          ''.ljust(head_width[2]),
                          num_to_str(sum(p for p, _, _ in weighted_scores)).ljust(head_width[3]),
                          num_to_str(sum(r for _, r, _ in weighted_scores)).ljust(head_width[4]),
                          num_to_str(sum(f for _, _, f in weighted_scores)).ljust(head_width[5])])

        print(n_correct, n_pred, n_true)
        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2.0 * precision * recall, precision + recall)

        res += f'\n Total Examples: {self.num_tests}'
        res += f'\n ============== Evaluation Metrics (positive only) =============='
        res += f'\n Overall Precision: {num_to_str(precision)}'
        res += f'\n Overall Recall: {num_to_str(recall)}'
        res += f'\n Overall F1: {num_to_str(f1_score)} '
        self.rel_f1 = f1_score
        return res
        #return "\n"


def evt_rel_distribution():
    include_pairs  = [('PROBLEM', 'CLINICAL_DEPT'),
                      ('PROBLEM', 'EVIDENTIAL'),
                      ('TREATMENT', 'TEST'),
                      ('TREATMENT', 'TEST'),
                      ('CLINICAL_DEPT', 'OCCURRENCE'),
                      ('TEST', 'OCCURRENCE'),
                      ('EVIDENTIAL', 'TEST'),
                      ('OCCURRENCE', 'TREATMENT'),
                      ('PROBLEM', 'TEST'),
                      ('TREATMENT', 'PROBLEM'),
                      ('TEST', 'TREATMENT'),
                      ('PROBLEM', 'TREATMENT'),
                      ('PROBLEM', 'TEST'),
                      ('OCCURRENCE', 'TREATMENT'),
                      ('PROBLEM', 'TREATMENT'),
                      ('CLINICAL_DEPT', 'OCCURRENCE'),
                      ('OCCURRENCE', 'TEST'),
                      ('CLINICAL_DEPT', 'CLINICAL_DEPT'),
                      ('CLINICAL_DEPT', 'TREATMENT'),
                      ('OCCURRENCE', 'TEST')]
    
    new_ditr = OrderedDict()
    for k, v in distribution.items():
        new_key = (label_map_evt["B-%s" % k[0]], label_map_evt["B-%s" % k[1]], label_map_rel[k[2]])
        new_ditr[new_key] = v
        if (k[0], k[1])	in include_pairs:
            new_ditr[new_key] = v
        else:
            new_ditr[new_key] = 0.0
        
    return new_ditr


distribution = OrderedDict([(('B-CLINICAL_DEPT', 'B-CLINICAL_DEPT', 'AFTER'), 0.0376),
             (('B-CLINICAL_DEPT', 'B-CLINICAL_DEPT', 'BEFORE'), 0.1033),
             (('B-CLINICAL_DEPT', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.8592),
             (('B-CLINICAL_DEPT', 'B-EVIDENTIAL', 'AFTER'), 0.7059),
             (('B-CLINICAL_DEPT', 'B-EVIDENTIAL', 'OVERLAP'), 0.2941),
             (('B-CLINICAL_DEPT', 'B-OCCURRENCE', 'AFTER'), 0.7183),
             (('B-CLINICAL_DEPT', 'B-OCCURRENCE', 'BEFORE'), 0.1761),
             (('B-CLINICAL_DEPT', 'B-OCCURRENCE', 'OVERLAP'), 0.1055),
             (('B-CLINICAL_DEPT', 'B-PROBLEM', 'AFTER'), 0.0794),
             (('B-CLINICAL_DEPT', 'B-PROBLEM', 'BEFORE'), 0.0952),
             (('B-CLINICAL_DEPT', 'B-PROBLEM', 'OVERLAP'), 0.8254),
             (('B-CLINICAL_DEPT', 'B-TEST', 'AFTER'), 0.1707),
             (('B-CLINICAL_DEPT', 'B-TEST', 'BEFORE'), 0.0366),
             (('B-CLINICAL_DEPT', 'B-TEST', 'OVERLAP'), 0.7927),
             (('B-CLINICAL_DEPT', 'B-TREATMENT', 'AFTER'), 0.2164),
             (('B-CLINICAL_DEPT', 'B-TREATMENT', 'BEFORE'), 0.0896),
             (('B-CLINICAL_DEPT', 'B-TREATMENT', 'OVERLAP'), 0.694),
             (('B-EVIDENTIAL', 'B-CLINICAL_DEPT', 'BEFORE'), 0.1579),
             (('B-EVIDENTIAL', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.8421),
             (('B-EVIDENTIAL', 'B-EVIDENTIAL', 'AFTER'), 0.0517),
             (('B-EVIDENTIAL', 'B-EVIDENTIAL', 'BEFORE'), 0.0259),
             (('B-EVIDENTIAL', 'B-EVIDENTIAL', 'OVERLAP'), 0.9224),
             (('B-EVIDENTIAL', 'B-OCCURRENCE', 'AFTER'), 0.1684),
             (('B-EVIDENTIAL', 'B-OCCURRENCE', 'BEFORE'), 0.0947),
             (('B-EVIDENTIAL', 'B-OCCURRENCE', 'OVERLAP'), 0.7368),
             (('B-EVIDENTIAL', 'B-PROBLEM', 'AFTER'), 0.1258),
             (('B-EVIDENTIAL', 'B-PROBLEM', 'OVERLAP'), 0.8742),
             (('B-EVIDENTIAL', 'B-TEST', 'AFTER'), 0.0146),
             (('B-EVIDENTIAL', 'B-TEST', 'OVERLAP'), 0.9854),
             (('B-EVIDENTIAL', 'B-TREATMENT', 'AFTER'), 0.4103),
             (('B-EVIDENTIAL', 'B-TREATMENT', 'OVERLAP'), 0.5897),
             (('B-OCCURRENCE', 'B-CLINICAL_DEPT', 'AFTER'), 0.0559),
             (('B-OCCURRENCE', 'B-CLINICAL_DEPT', 'BEFORE'), 0.148),
             (('B-OCCURRENCE', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.7961),
             (('B-OCCURRENCE', 'B-EVIDENTIAL', 'AFTER'), 0.1778),
             (('B-OCCURRENCE', 'B-EVIDENTIAL', 'BEFORE'), 0.4222),
             (('B-OCCURRENCE', 'B-EVIDENTIAL', 'OVERLAP'), 0.4),
             (('B-OCCURRENCE', 'B-OCCURRENCE', 'AFTER'), 0.1715),
             (('B-OCCURRENCE', 'B-OCCURRENCE', 'BEFORE'), 0.1915),
             (('B-OCCURRENCE', 'B-OCCURRENCE', 'OVERLAP'), 0.637),
             (('B-OCCURRENCE', 'B-PROBLEM', 'AFTER'), 0.256),
             (('B-OCCURRENCE', 'B-PROBLEM', 'BEFORE'), 0.07),
             (('B-OCCURRENCE', 'B-PROBLEM', 'OVERLAP'), 0.6739),
             (('B-OCCURRENCE', 'B-TEST', 'AFTER'), 0.25),
             (('B-OCCURRENCE', 'B-TEST', 'BEFORE'), 0.1611),
             (('B-OCCURRENCE', 'B-TEST', 'OVERLAP'), 0.5889),
             (('B-OCCURRENCE', 'B-TREATMENT', 'AFTER'), 0.3766),
             (('B-OCCURRENCE', 'B-TREATMENT', 'BEFORE'), 0.1),
             (('B-OCCURRENCE', 'B-TREATMENT', 'OVERLAP'), 0.5234),
             (('B-PROBLEM', 'B-CLINICAL_DEPT', 'AFTER'), 0.0128),
             (('B-PROBLEM', 'B-CLINICAL_DEPT', 'BEFORE'), 0.5321),
             (('B-PROBLEM', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.4551),
             (('B-PROBLEM', 'B-EVIDENTIAL', 'AFTER'), 0.0071),
             (('B-PROBLEM', 'B-EVIDENTIAL', 'BEFORE'), 0.848),
             (('B-PROBLEM', 'B-EVIDENTIAL', 'OVERLAP'), 0.1449),
             (('B-PROBLEM', 'B-OCCURRENCE', 'AFTER'), 0.0924),
             (('B-PROBLEM', 'B-OCCURRENCE', 'BEFORE'), 0.494),
             (('B-PROBLEM', 'B-OCCURRENCE', 'OVERLAP'), 0.4135),
             (('B-PROBLEM', 'B-PROBLEM', 'AFTER'), 0.0284),
             (('B-PROBLEM', 'B-PROBLEM', 'BEFORE'), 0.0722),
             (('B-PROBLEM', 'B-PROBLEM', 'OVERLAP'), 0.8994),
             (('B-PROBLEM', 'B-TEST', 'AFTER'), 0.022),
             (('B-PROBLEM', 'B-TEST', 'BEFORE'), 0.675),
             (('B-PROBLEM', 'B-TEST', 'OVERLAP'), 0.303),
             (('B-PROBLEM', 'B-TREATMENT', 'AFTER'), 0.1358),
             (('B-PROBLEM', 'B-TREATMENT', 'BEFORE'), 0.5926),
             (('B-PROBLEM', 'B-TREATMENT', 'OVERLAP'), 0.2716),
             (('B-TEST', 'B-CLINICAL_DEPT', 'AFTER'), 0.0452),
             (('B-TEST', 'B-CLINICAL_DEPT', 'BEFORE'), 0.0333),
             (('B-TEST', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.9214),
             (('B-TEST', 'B-EVIDENTIAL', 'AFTER'), 0.0509),
             (('B-TEST', 'B-EVIDENTIAL', 'BEFORE'), 0.0763),
             (('B-TEST', 'B-EVIDENTIAL', 'OVERLAP'), 0.8728),
             (('B-TEST', 'B-OCCURRENCE', 'AFTER'), 0.1728),
             (('B-TEST', 'B-OCCURRENCE', 'BEFORE'), 0.1442),
             (('B-TEST', 'B-OCCURRENCE', 'OVERLAP'), 0.683),
             (('B-TEST', 'B-PROBLEM', 'AFTER'), 0.1519),
             (('B-TEST', 'B-PROBLEM', 'BEFORE'), 0.0181),
             (('B-TEST', 'B-PROBLEM', 'OVERLAP'), 0.8299),
             (('B-TEST', 'B-TEST', 'AFTER'), 0.0588),
             (('B-TEST', 'B-TEST', 'BEFORE'), 0.0404),
             (('B-TEST', 'B-TEST', 'OVERLAP'), 0.9008),
             (('B-TEST', 'B-TREATMENT', 'AFTER'), 0.2723),
             (('B-TEST', 'B-TREATMENT', 'BEFORE'), 0.1881),
             (('B-TEST', 'B-TREATMENT', 'OVERLAP'), 0.5396),
             (('B-TREATMENT', 'B-CLINICAL_DEPT', 'AFTER'), 0.1752),
             (('B-TREATMENT', 'B-CLINICAL_DEPT', 'BEFORE'), 0.0777),
             (('B-TREATMENT', 'B-CLINICAL_DEPT', 'OVERLAP'), 0.7471),
             (('B-TREATMENT', 'B-EVIDENTIAL', 'AFTER'), 0.5),
             (('B-TREATMENT', 'B-EVIDENTIAL', 'BEFORE'), 0.31),
             (('B-TREATMENT', 'B-EVIDENTIAL', 'OVERLAP'), 0.19),
             (('B-TREATMENT', 'B-OCCURRENCE', 'AFTER'), 0.3328),
             (('B-TREATMENT', 'B-OCCURRENCE', 'BEFORE'), 0.3771),
             (('B-TREATMENT', 'B-OCCURRENCE', 'OVERLAP'), 0.2902),
             (('B-TREATMENT', 'B-PROBLEM', 'AFTER'), 0.2841),
             (('B-TREATMENT', 'B-PROBLEM', 'BEFORE'), 0.1562),
             (('B-TREATMENT', 'B-PROBLEM', 'OVERLAP'), 0.5597),
             (('B-TREATMENT', 'B-TEST', 'AFTER'), 0.25),
             (('B-TREATMENT', 'B-TEST', 'BEFORE'), 0.2574),
             (('B-TREATMENT', 'B-TEST', 'OVERLAP'), 0.4926),
             (('B-TREATMENT', 'B-TREATMENT', 'AFTER'), 0.0914),
             (('B-TREATMENT', 'B-TREATMENT', 'BEFORE'), 0.0843),
             (('B-TREATMENT', 'B-TREATMENT', 'OVERLAP'), 0.8243)])

def constraint_dict():
    '''
    include_pairs  = [('PROBLEM', 'CLINICAL_DEPT', 'OVERLAP'),
                      ('OCCURRENCE', 'TREATMENT', 'OVERLAP'),
                      ('TEST', 'TREATMENT', 'OVERLAP'),
                      ('TREATMENT', 'TEST', 'OVERLAP'),
                      ('PROBLEM', 'CLINICAL_DEPT', 'BEFORE'),
                      ('TREATMENT', 'PROBLEM', 'OVERLAP'),
                      ('OCCURRENCE', 'TREATMENT', 'AFTER'),
                      ('EVIDENTIAL', 'TEST', 'OVERLAP'),
                      ('TREATMENT', 'TEST', 'AFTER'),
                      ('TEST', 'OCCURRENCE', 'OVERLAP'),
                      ('TEST', 'TREATMENT', 'AFTER'),
                      ('CLINICAL_DEPT', 'TREATMENT', 'OVERLAP'),
                      ('PROBLEM', 'EVIDENTIAL', 'BEFORE'),
                      ('PROBLEM', 'EVIDENTIAL', 'OVERLAP'),
                      ('CLINICAL_DEPT', 'CLINICAL_DEPT', 'OVERLAP'),
                      ('TEST', 'OCCURRENCE', 'AFTER'),
                      ('PROBLEM', 'TEST', 'BEFORE'),
                      ('OCCURRENCE', 'TEST', 'OVERLAP'),
                      ('PROBLEM', 'TREATMENT', 'BEFORE'),
                      ('PROBLEM', 'TEST', 'OVERLAP'),
                      ('PROBLEM', 'TREATMENT', 'AFTER'),
                      ('CLINICAL_DEPT', 'OCCURRENCE', 'BEFORE'),
                      ('CLINICAL_DEPT', 'OCCURRENCE', 'AFTER'),
                      ('OCCURRENCE', 'TEST', 'AFTER')]

    include_pairs = [('OCCURRENCE', 'OCCURRENCE', 'OVERLAP'),
                     ('PROBLEM', 'EVIDENTIAL', 'BEFORE'),
                     ('TEST', 'EVIDENTIAL', 'OVERLAP'),
                     -('TREATMENT', 'TEST', 'OVERLAP'),
                     ('TREATMENT', 'OCCURRENCE', 'BEFORE'),
                     -('TEST', 'OCCURRENCE', 'OVERLAP'),
                     -('PROBLEM', 'TEST', 'OVERLAP'),
                     -('TREATMENT', 'PROBLEM', 'OVERLAP'),
                     ('TEST', 'TREATMENT', 'OVERLAP'),
                     ('PROBLEM', 'OCCURRENCE', 'OVERLAP'),
                     ('TREATMENT', 'OCCURRENCE', 'OVERLAP'),
                     ('PROBLEM', 'OCCURRENCE', 'BEFORE'),
                     ('OCCURRENCE', 'CLINICAL_DEPT', 'OVERLAP'),
                     -('PROBLEM', 'TEST', 'BEFORE'),
                     ('OCCURRENCE', 'PROBLEM', 'OVERLAP'),
    ]

    include_pairs = [('CLINICAL_DEPT', 'OCCURRENCE', 'AFTER'),
                     ('PROBLEM', 'OCCURRENCE', 'BEFORE'),
                     ('PROBLEM', 'OCCURRENCE', 'OVERLAP'),
                     #('PROBLEM', 'TREATMENT', 'OVERLAP'),
                     ('TEST', 'EVIDENTIAL', 'OVERLAP'),
                     #('TREATMENT', 'OCCURRENCE', 'AFTER'),
                     #('TREATMENT', 'OCCURRENCE', 'OVERLAP'),
                     #('TREATMENT', 'PROBLEM', 'OVERLAP'),
                     ('TREATMENT', 'TREATMENT', 'OVERLAP')]
    '''
    include_pairs = [('B-EVIDENTIAL', 'B-PROBLEM', 'AFTER')]
    #('TEST', 'OCCURRENCE', 'OVERLAP')]
    #('TREATMENT', 'TEST', 'OVERLAP')]
    
    constraints = OrderedDict()
    for k, v in distribution.items():
        new_key = (label_map_evt[k[0]], label_map_evt[k[1]], label_map_rel[k[2]])
        #if k in include_pairs:
        constraints[new_key] = v
            
    return constraints
    

    
