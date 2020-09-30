# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import pickle
import numpy as np
import json
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from collections import Counter, OrderedDict
from transformers import *
from models_end2end import *
from optimization import *
from collections import defaultdict
from utils import *
from utils import constraint_dict
import sys
import copy
from scipy import stats
from scipy.special import softmax
from gurobi_inference import Global_Inference
from LROptimization import event_relation_constraints

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

label_map_rel = {'BEFORE': 0, 'AFTER': 1, 'OVERLAP': 2}
# 'O' needs to have the o index to make the data proccessing code work                                            
label_map_evt = {'O': 0,
                 'B-CLINICAL_DEPT': 1, 'I-CLINICAL_DEPT': 2,
                 'B-EVIDENTIAL': 3, 'I-EVIDENTIAL': 4,
                 'B-OCCURRENCE': 5, 'I-OCCURRENCE': 6,
                 'B-PROBLEM': 7, 'I-PROBLEM': 8,
                 'B-TEST': 9, 'I-TEST': 10,
                 'B-TREATMENT': 11, 'I-TREATMENT': 12}

def main(constraints):
    parser = argparse.ArgumentParser()
    ## Required parameters                                                                                         
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .json files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="pre-trained model selected in the list: roberta-base, "
                             "roberta-large, bert-base, bert-large. ")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--model_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The directory where the trained model are saved")
    parser.add_argument("--lr",
                        default=0.1,
                        type=float,
                        required=True,
                        help="learning rate for inference algorithm")
    parser.add_argument("--max_iter",
                        default=10,
                        type=int,
                        required=True,
                        help="max iterations for inference algorithm")
    parser.add_argument("--tolerance",
                        default=0.1,
                        type=float,
                        required=True,
                        help="tolerance level for inference algorithm")
    parser.add_argument("--split",
                        type=str,
                        required=True,
                        help="train/dev/test")
    ## Other parameters
    parser.add_argument("--use_gold_event",
                        action='store_true')
    parser.add_argument("--decay",
                        default=0.99,
                        type=float,
                        help="learning rate decay rate for inference algorithm")                                                                                            
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--mlp_hid_size",
                        default=64,
                        type=int,
                        help="hid dimension for MLP layer.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=7,
                        help="random seed for initialization")
    parser.add_argument('--device_number',
                        type=int,
                        default=0)
    parser.add_argument('--triplet',
                        type=str,
                        default="")
    args = parser.parse_args()

    triplet = args.triplet.split('_')
    include_pairs = [('_'.join(triplet[0:-2]), triplet[-2], triplet[-1])]
    print(include_pairs)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_number)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    logger.info("current task is " + str(task_name))
    tokenizer = RobertaTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    
    # load trained model
    # created as placeholder                                                                           
    weights = torch.tensor([1. for x in label_map_evt]).to(device)
    relation_weights = torch.tensor([1.0 for x in label_map_rel]).to(device)
    print(relation_weights)
    
    model_state_dict = torch.load(args.model_dir + "pytorch_model.bin")
    model = RobertaEnd2endClassifier.from_pretrained(args.model, state_dict=model_state_dict,
                                                     cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_-1',
                                                     num_class=len(label_map_evt),
                                                     mlp_hid=args.mlp_hid_size,
                                                     class_weights=weights, rel_class_weights=relation_weights)
    model.to(device)
    data = load_data(args.data_dir, "%s_events_rels" % args.split)
    
    eval_features = convert_to_features_roberta_instance(data, tokenizer, label_map_evt, 
                                                         max_length=args.max_seq_length, evaluation=True)
    eval_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    eval_input_mask = torch.tensor(select_field(eval_features, 'mask_ids'), dtype=torch.long)
    eval_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    eval_labels  = torch.tensor(select_field(eval_features, 'event_labels'), dtype=torch.long)
    eval_orig2token = select_field(eval_features, 'orig2token')
    all_instance_keys = select_field(eval_features, 'sample_id')
    all_label_keys =  select_field(eval_features, 'event_keys')
    eval_relations = select_field(eval_features, 'relations')
    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_segment_ids, eval_labels)

    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    idx = 0
    n_corr, n_corr_relaxed, n_pred, n_gold = 0, 0, 0, 0
    event_agg_gold, event_agg_pred = defaultdict(int), defaultdict(list)
    relation_pred_counter = Counter()
    relation_agg_gold, relation_agg_pred, relation_agg_pred_local =  OrderedDict(), OrderedDict(), OrderedDict()
    eval_gold = True if "gold" in args.task_name else False
    all_data = []
    
    model.eval()
    for input_ids, input_masks, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_masks = input_masks.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        indices, rel_labels, lengths = rel_inputs([(eval_relations[idx+i], eval_orig2token[idx+i])
                                                   for i in range(args.eval_batch_size)
                                                   if idx+i < len(eval_relations)])
        batch_orig2token = [eval_orig2token[idx+i] for i in range(args.eval_batch_size)
                            if idx+i < len(eval_relations)]
         
        rel_labels = torch.tensor(rel_labels).to(device)
        with torch.no_grad():
            logits, logits_rel, token_predictions, pred_indices, pred_lengths, \
                _, _ = model(input_ids, indices, lengths, batch_orig2token, train=False,
                             rel_labels=rel_labels, labels=label_ids, use_gold=eval_gold,
                             attention_mask=input_masks, token_type_ids=segment_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            
            all_prob_evt = softmax(logits, axis=2)
            batch_prob_evt, batch_ekey, batch_gold = [], [], []
            assert all_prob_evt.shape[1] == args.max_seq_length
            for ib in range(args.eval_batch_size):
                if idx + ib < len(all_instance_keys):
                    # idx + ib identifies unique context
                    # the first element is document id
                    doc = all_instance_keys[idx+ib].split('_')[0]
                    # collect event labels + preds
                    prob_evt, key_evt, golds = [], [], []
                    for k, i in enumerate(eval_orig2token[ib+idx]):
                        gold = label_ids[ib][i]
                        golds.append(gold)
                        prob_evt.append(all_prob_evt[ib][i][:])
                        ekey = (doc, all_label_keys[idx+ib][k][0], all_label_keys[idx+ib][k][1])
                        key_evt.append(ekey)
                        if ekey in event_agg_gold: assert event_agg_gold[ekey] == gold
                        else: event_agg_gold[ekey] = gold

                    batch_prob_evt.append(np.asarray(prob_evt))
                    batch_ekey.append(key_evt)
                    batch_gold.append(golds)
                    
            gold_indices = reshape_to_batch(indices, lengths, eval_orig2token, idx)
            groundtruth = reshape_to_batch(rel_labels.detach().cpu().numpy(), lengths)
            batch_rel_evt_indices = []

            # collect relation labels + preds
            if logits_rel.nelement() > 0:
                logits_rel = logits_rel.detach().cpu().numpy()
                pred_indices = reshape_to_batch(pred_indices, pred_lengths, eval_orig2token, idx)
                predictions = reshape_to_batch(np.argmax(logits_rel, axis=1), pred_lengths)

                batch_prob_rel = reshape_to_batch(softmax(logits_rel, axis=1), pred_lengths)
                assert len(pred_indices) == len(predictions)
                assert len(pred_indices) == len(batch_prob_rel)
                
            else:
                pred_indices = [[] for x in range(len(batch_ekey))]
                predictions = [[] for x in range(len(batch_ekey))]  
                batch_prob_rel = [np.asarray([]) for x in range(len(batch_ekey))]

                assert len(pred_indices) == len(predictions)
                assert len(pred_indices) == len(batch_prob_rel)
                
            if sum(len(x) for x in batch_prob_rel) > 0:
                assert sum([x.shape[0] for x in batch_prob_rel]) == logits_rel.shape[0]

            assert len(batch_prob_evt) == len(batch_prob_rel)
            assert len(batch_ekey) == len(batch_gold)
            assert len(batch_prob_evt) == len(gold_indices)

            # collect data for inference
            all_data.extend(list(zip(batch_prob_evt, batch_prob_rel, batch_ekey, batch_gold,
                                     gold_indices, pred_indices, groundtruth, predictions)))
            idx += args.eval_batch_size
            
    ############################# Inference Steps ################################
    
    best_perf = 0
    lr = args.lr
    
    # relation constraints
    best_constraints = {}
    distribution = event_relation_constraints()
    include_pairs = [('B-OCCURRENCE', 'B-PROBLEM', 'OVERLAP'),
                     ('B-OCCURRENCE', 'B-TREATMENT', 'OVERLAP'),
                     ('B-TREATMENT', 'B-OCCURRENCE', 'OVERLAP'),
                     ('B-TREATMENT', 'B-PROBLEM', 'OVERLAP')]

                     
    #outfile = open("inference_logs/%s_%s_%s_inference_results.txt" % include_pairs[0], "a+")
    constraints = OrderedDict()

    for k, v in distribution.items():
        new_key = (label_map_evt[k[0]], label_map_evt[k[1]], label_map_rel[k[2]])
        if k in include_pairs:
            constraints[new_key] = v
    
    constraint_update = {k:True for k in constraints}
    #constraint_param = {(5, 7, 2): -0.7144222553897182,
    #                    (5, 11, 2): -0.8219485714285716,
    #                    (11, 5, 2): -0.7535470447761197,
    #                    (11, 7, 2): -0.6694790419161676}
    constraint_param = {k: 0 for k in constraints}

    # event constraints
    event_constraints = {}
    
    event_constraint_param = {k: 0 for k in event_constraints}
    event_constraint_update = {k:True for k in event_constraints}
    num_meet_tolerance = 0
    
    for itr in range(args.max_iter):
        print("="*30, " Iteration #%s " %itr, "="*30)
        rel_correction_count, evt_correction_count = 0, 0
        constraint_perf = {k: [0, 0, 0, 0] for k in constraints}
        
        relation_pred_counter = Counter()
        event_pair_pred_counter = Counter()
        
        n_corr, n_corr_relaxed, n_pred, n_gold = 0, 0, 0, 0
        relation_agg_pred, event_agg_pred = defaultdict(list), defaultdict(list)

        for k, (prob_evt, prob_rel, ekey, egold,
                indices, pred_indices, rgold, rpred) in enumerate(all_data):
            
            N, Nc = prob_evt.shape
            evt_idx = np.zeros((N, Nc), dtype=int)

            # only evaluate on predictions with gold match
            prob_rel_pos, pred_indices_pos = [], []
            for i_pair, pred_pair in enumerate(pred_indices):
                if is_partial_match(pred_pair, indices):
                    pred_indices_pos.append(pred_pair)
                    prob_rel_pos.append(prob_rel[i_pair])
                    
            if len(prob_rel_pos) == 0:
                prob_rel_pos = np.array([])
                M = 0
                rel_idx = []
                rel_evt_types = []
            else:
                prob_rel_pos = np.stack(prob_rel_pos)
                M, Mc = prob_rel_pos.shape
                rel_idx = np.zeros((M, Mc), dtype=int)

            if args.use_gold_event:
                global_model = Global_Inference(prob_evt, prob_rel, indices, constraint_param,
                                                constraints, evt_gold=egold, label2idx=label_map_rel,
                                                relation_agg_gold=relation_agg_gold, rkey=rkey, debug=True)
            else:
                global_model = Global_Inference(prob_evt, prob_rel_pos, pred_indices_pos, constraint_param,
                                                constraints, event_constraint_param, event_constraints,
                                                label2idx=label_map_rel)
            global_model.run()
            rel_correction, evt_correction = global_model.predict()
            evt_correction_count += evt_correction
            rel_correction_count += rel_correction
            
            # event global assignment 
            for n in range(N):
                evt_idx[n, global_model.pred_evt_labels[n]] = 1
                if ekey[n] in event_agg_pred:
                    event_agg_pred[ekey[n]] += [global_model.pred_evt_labels[n]]
                else:
                    event_agg_pred[ekey[n]] = [global_model.pred_evt_labels[n]]
                    
                # calculate event (consecutive) pair count
                if n < N - 1:
                    for event_pair in event_constraints.keys():
                        if global_model.pred_evt_labels[n] == event_pair[0]:
                            event_pair_pred_counter[(global_model.pred_evt_labels[n]),
                                                    global_model.pred_evt_labels[n+1]] += 1
                            
            # relation global assignment
            assert len(global_model.pred_rel_labels) == len(pred_indices_pos)
            n_c, n_cr, n_p, n_g = eval_end2end_result([global_model.pred_rel_labels],
                                                      [rgold], [pred_indices_pos], [indices])

            temp_preds_rel = eval_end2end_gold_perf([global_model.pred_rel_labels], [rgold],
                                                    [pred_indices_pos], [indices], all_instance_keys,
                                                    eval_relations, k)
            relation_agg_gold, relation_agg_pred = update_relation_agg(temp_preds_rel, relation_agg_gold,
                                                                       relation_agg_pred)

            n_corr += n_c
            n_corr_relaxed += n_cr
            n_pred += n_p
            n_gold += n_g

            assert len(global_model.pred_rel_labels) == len(pred_indices_pos)
            for m in range(M):
                if global_model.pred_rel_labels[m] <= 2:
                    rel_idx[m, global_model.pred_rel_labels[m]] = 1
                    if args.use_gold_event:
                        relation_pred_counter[(egold[pred_indices[m][0]], egold[pred_indices[m][2]],
                                               global_model.pred_rel_labels[m])] += 1
                    else:
                        # manually fix I-Type --> B-Type
                        first = global_model.pred_evt_labels[pred_indices_pos[m][0]]
                        if first > 0 and first % 2 == 0: first -= 1
                        second = global_model.pred_evt_labels[pred_indices_pos[m][2]]
                        if second > 0 and second % 2 == 0: second -= 1
                        relation_pred_counter[(first, second,
                                               global_model.pred_rel_labels[m])] += 1

            constraint_perf = eval_rel_constraint_perf(global_model.pred_rel_labels, rgold, pred_indices_pos, indices,
                                                       global_model.pred_evt_labels, egold, constraint_perf)
            
        for triplet, perf in constraint_perf.items():
            recall, precision = 0.0, 0.0
            if perf[0] > 0: recall = perf[1] / perf[0]
            if perf[2] > 0: precision = perf[3] / perf [2]
            print("(%s, %s, %s): recall: %.4f; precision: %.4f" % (triplet[0], triplet[1], triplet[2],
                                                                 recall, precision))
            
        #for evt, e_count in event_pred_counter.items():
        #    print(evt, e_count, e_count / sum(event_pred_counter.values()))
        #save_results(relation_pred_counter, "%s_triplet_pred.pkl" % args.split, "./results/")
        # calculate event pair constraint parameter
        event_diff_list = []
        prev_event_constraints = copy.deepcopy(event_constraint_param)
        idx2label = {v:k for k,v in label_map_evt.items()}

        for ic, ((e1, e2), v) in enumerate(event_constraint_param.items()):
            event_pair_sum = sum([v for k, v in event_pair_pred_counter.items() if e1 == k[0]])
            current_ratio = event_pair_pred_counter[(e1, e2)] / event_pair_sum

            diff = current_ratio - event_constraints[(e1, e2)]

            dl = diff
            # ensure lambda is always positive
            #if diff < 0: dl = -diff
            #else: dl = diff
            
            event_diff_list.append(((e1, e2), diff))

            print(idx2label[e1], idx2label[e2], event_pair_pred_counter[(e1, e2)],
                  event_pair_sum, current_ratio, diff)

            if abs(diff) <= args.tolerance and event_constraint_update[(e1, e2)]:
                num_meet_tolerance += 1
                event_constraint_update[(e1, e2)] = False
                
            if event_constraint_update[(e1, e2)]: event_constraint_param[(e1, e2)] += args.lr * dl

        #print("Event Constraints", prev_event_constraints)
        #print("Event_Diff", {k: round(x, 4) for k, x in event_diff_list})
        #print("Total Event Correction is: %d" % (evt_correction_count))
        
        diff_list = []
        prev_constraints = copy.deepcopy(constraint_param)

        idx2label_rel = {v:k for k,v in label_map_rel.items()}
        for ic, ((e1, e2, r), v) in enumerate(constraint_param.items()):
                
            pair_rel_sum = sum([relation_pred_counter[(e1, e2, i)] for i in range(len(label_map_rel))])
            current_ratio = relation_pred_counter[(e1, e2, r)] / pair_rel_sum

            diff = constraints[(e1, e2, r)] - current_ratio
            diff_list.append(((e1, e2, r), diff))

            print(idx2label[e1], idx2label[e2], idx2label_rel[r], relation_pred_counter[(e1, e2, r)],
                  pair_rel_sum, current_ratio, diff)

            if abs(diff) <= args.tolerance and constraint_update[(e1, e2, r)]:
                num_meet_tolerance += 1
                constraint_update[(e1, e2, r)] = False

            if constraint_update[(e1, e2, r)]:
                constraint_param[(e1, e2, r)] += args.lr * diff
            
        print("Constraints", prev_constraints)
        print("Diff", {k: round(x, 4) for k, x in diff_list})
        print("num_meet_tolerance: %d" % num_meet_tolerance)
        
        args.lr *= args.decay
        print("Total Relation Correction is: %d" % (rel_correction_count))

        assert len(relation_agg_gold) == len(relation_agg_pred)
        assert len(event_agg_gold) == len(event_agg_pred)

        pred_names = [idx2label[stats.mode(v)[0][0]] for v in event_agg_pred.values()]
        label_names = [idx2label[v] for v in event_agg_gold.values()]
        #print(ClassificationReport(args.task_name, label_names, pred_names))

        idx2label_rel[-1] = 'NONE'
        pred_names = [idx2label_rel[stats.mode(v)[0][0]] for v in relation_agg_pred.values()]
        label_names = [idx2label_rel[v] for v in relation_agg_gold.values()]
        res = ClassificationReport(args.task_name, label_names, pred_names)
        print(res)
        #print("================= Final Eval F1 Scores ===================")
        #rel_f1 = calculate_f1(n_corr, n_pred, n_gold)
        rel_f1_relaxed = calculate_f1(n_corr_relaxed, n_pred, n_gold)
        #print("current positive end-to-end f1 score is %.4f" % rel_f1)
        #print("current positive end-to-end f1 score (relaxed) is %.4f" % rel_f1_relaxed)
        
        perf = res.rel_f1
        if perf > best_perf:
            best_perf = perf
            best_constraints = prev_constraints
        if num_meet_tolerance == len(constraints): break

        #pred_names_local = [idx2label_rel[stats.mode([vv[0] for vv in v])[0][0]]
    #                    for v in relation_agg_pred_local.values()]
    #pred_score = [np.mean([vv[1] for vv in v]) for v in relation_agg_pred_local.values()]
    #output_errors(relation_agg_gold, label_names, pred_names_local, pred_score, pred_names) 
    print(best_constraints)
    #output_errors(relation_agg_gold, label_names, pred_names, "global_best")
    
    #outfile.write("%d;%.2f;%.2f;%.2f\t%.4f\t%.4f\t%d\n" % (args.max_iter, lr, args.decay,
    #                                                       args.tolerance, best_perf, perf,
    #                                                       num_meet_tolerance))
    #outfile.close()
    tempEval_scores = eval_all_files(relation_agg_gold, relation_agg_pred)
    print("current TempEval end-to-end precision: %.4f; recall: %.4f; f1: %.4f" % tempEval_scores)
if __name__ == "__main__":
    constraints = constraint_dict()
    main(constraints)
