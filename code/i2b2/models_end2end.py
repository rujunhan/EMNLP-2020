from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open
from itertools import combinations
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import *
import numpy as np
from utils import mask_negative, reshape_to_batch, label_map_rel, label_map_evt, reverse_map_rel, partial_match
from gurobi_inference import Event_Inference
from scipy.special import softmax
import copy

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
}



class BertEnd2endClassifier(BertPreTrainedModel):
    def __init__(self, config, class_weights, rel_class_weights, num_class, num_class_rel=3,
                 mlp_hid=16, dropout=0.1, hid_size=100):
        super(BertEnd2endClassifier, self).__init__(config)
        self.bert = BertModel(config)

        for name, param in self.bert.named_parameters():
            param.requires_grad = False
            
        self.dropout = nn.Dropout(dropout)
        # lstm is shared for both relation and entity
        self.hid_size = hid_size
        self.lstm = nn.LSTM(768, hid_size, bias=False, bidirectional=True, batch_first=True)
        
        # event                                                                                                
        self.num_labels = num_class
        self.linear1 = nn.Linear(2*hid_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.class_weights = class_weights

        # relation
        self.num_labels_rel = num_class_rel
        self.linear1_rel = nn.Linear(4*hid_size, mlp_hid)
        self.linear2_rel = nn.Linear(mlp_hid, self.num_labels_rel)
        self.rel_class_weights = rel_class_weights

        self.act = nn.Tanh()
        self.init_weights()

    def construct_events_relaxed(self, token_pred, orig2token):
        batch_events = []
        for b, o2t in enumerate(orig2token):
            events = []
            event_start, event_end = 0, 0
            for k, i in enumerate(o2t):
                if token_pred[b][i] % 2 == 1:
                    # case 1-a: find a new start                                                                             
                    if event_start == 0: event_start = i
                    # case 1-b: find a end token (by looking a new start token)                                              
                    else:
                        event_end = o2t[k-1] # find previous end's head token                                               
                        events.append((event_start, event_end))
                        event_start, event_end = i, 0

                elif token_pred[b][i] % 2 == 0 and token_pred[b][i] > 0:
                    # case 2-a: find a new start                                                                             
                    if event_start == 0: event_start = i
                    # case 2-b                                                                                               
                    elif not (token_pred[b][i] == token_pred[b][event_start] or
                          token_pred[b][i] == token_pred[b][event_start] + 1):
                        event_end = o2t[k-1] # find previous end's head token                                               
                        events.append((event_start, event_end))
                        event_start, event_end = i, 0

                #case 3: find event ending for multi-token event                                              
                elif event_start > 0 and token_pred[b][i] == 0:
                    event_end = o2t[k-1] # find current end's head token
                    events.append((event_start, event_end))
                    event_start, event_end = 0, 0

            # case 4: last word or phrase is an event                                                 
            if event_start > 0:
                events.append((event_start, i))
            batch_events.append(events)
        return batch_events

    ## Eventually we want to handle the symmetric label issue!!!                            
    def label_lookup(self, pairs, gold_indices, labels):
        new_labels, new_pairs = [], []
        for pair in pairs:
            # one pair should have exactly one match                                                                         
            for i, idx in enumerate(gold_indices):
                if partial_match((pair[0], pair[1]), (idx[0], idx[1])) and \
                   partial_match((pair[2], pair[3]), (idx[2], idx[3])):
                    new_labels.append(labels[i])
                    new_pairs.append(pair)
                    break
                if partial_match((pair[0], pair[1]), (idx[2], idx[3])) and \
                   partial_match((pair[2], pair[3]), (idx[0], idx[1])):
                    # quick trick to handle reverse label                                                                    
                    new_labels.append(reverse_map_rel[labels[i]])
                    new_pairs.append(pair)
                    break
        assert len(new_labels) <= len(pairs)
        assert len(new_labels) == len(new_pairs)
        return new_labels, new_pairs

    
    def construct_relations(self, events, indices, lengths, rel_labels, train=True):
        # search gold relations (with partial matched) if they exist                                                         

        idx = 0
        new_indices, new_lengths, new_labels = [], [], []
        for b, event_list in enumerate(events):
            idx_end = idx + lengths[b]

            all_pairs = [(x[0], x[1], y[0], y[1]) for x, y in combinations(event_list, 2)]

            # using gold matched label for training                                                                          
            if train:
                matched_labels, new_pairs = self.label_lookup(all_pairs, indices[idx:idx_end],
                                                              rel_labels[idx:idx_end].cpu().numpy())
                new_labels.extend(matched_labels)
                new_indices.extend(new_pairs)
                new_lengths.append(len(new_pairs))

            # assign placeholder labels (2: overlap) for inference                                                           
            else:
                new_labels.extend([2]*len(all_pairs))
                new_indices.extend(all_pairs)
                new_lengths.append(len(all_pairs))

            idx += lengths[b]

        assert len(new_indices) == sum(new_lengths)
        assert len(new_indices) == len(new_labels)
        return new_indices, new_lengths, torch.tensor(new_labels).cuda()

    
    def forward(self, input_ids, indices, lengths, orig2token, rel_labels=None, labels=None,
                attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                use_gold=True, inference=False, downsample_prob=0.0, train=True):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        outputs = self.dropout(outputs[0])
        outputs, _ = self.lstm(outputs)
        loss, loss_rel, logits_rel = 0.0, 0.0, torch.tensor([])

        # Event MLP layer                                                                           
        logits = self.linear2(self.act(self.linear1(outputs)))

        # Event and Relation Construction                                                                                   
        if not use_gold:
            token_predictions = np.argmax(logits.detach().cpu().numpy(), axis=2)
            events = self.construct_events_relaxed(token_predictions, orig2token)
            # new labels never returned out of this function                                                  
            # so we still evaluation against original labels                             
            indices, lengths, rel_labels = self.construct_relations(events, indices, lengths,
                                                                    rel_labels, train=train)
        else:
            token_predictions = np.argmax(logits.detach().cpu().numpy(), axis=2)
        
        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none', weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss.reshape(attention_mask.size()) * attention_mask
            loss = loss.reshape(1, -1).sum() / attention_mask.reshape(1, -1).sum()

             
        # Relation MLP                                                                                                 
        idx = 0
        rel_vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                l1, l2, r1, r2 = indices[idx]
                rel_vector = torch.cat([outputs[b, l1, :self.hid_size], outputs[b, l2, self.hid_size:],
                                        outputs[b, r1, :self.hid_size], outputs[b, r2, self.hid_size:]])
                rel_vectors.append(rel_vector.unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        if rel_vectors:
            rel_vectors = torch.cat(rel_vectors, dim=0)
            outputs_rel = self.act(self.linear1_rel(rel_vectors))
            logits_rel = self.linear2_rel(outputs_rel)

            if rel_labels is not None:
                loss_fct_rel = CrossEntropyLoss(reduction='none', weight=self.rel_class_weights)
                loss_rel = loss_fct_rel(logits_rel.view(-1, self.num_labels_rel), rel_labels)

                # random mask out negative pairs, this is equivalent to down-sample                                         
                rel_mask = mask_negative(rel_labels.detach().cpu().tolist(), downsample_prob)
                rel_mask = torch.tensor(rel_mask).cuda()

                loss_rel = loss_rel * rel_mask
                if rel_mask.sum() == 0:
                    loss_rel = loss_rel.sum()
                else:
                    loss_rel = loss_rel.sum() / rel_mask.sum()

        assert len(indices) == sum(lengths)

        return  logits, logits_rel, token_predictions, indices, lengths, loss, loss_rel


    
class RobertaEnd2endClassifier(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, class_weights, rel_class_weights, num_class, num_class_rel=3,
                 mlp_hid=16):
        super(RobertaEnd2endClassifier, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # event                                                                                                  
        self.num_labels = num_class
        self.linear1 = nn.Linear(config.hidden_size, mlp_hid)
        self.linear2 = nn.Linear(mlp_hid, self.num_labels)
        self.class_weights = class_weights
        # relation                                                                                               
        self.num_labels_rel = num_class_rel
        self.linear1_rel = nn.Linear(4*config.hidden_size, mlp_hid)
        self.linear2_rel = nn.Linear(mlp_hid, self.num_labels_rel)
        self.rel_class_weights = rel_class_weights

        self.act = nn.Tanh()
        self.init_weights()

    def construct_events_relaxed(self, token_pred, orig2token):
        batch_events = []
        for b, o2t in enumerate(orig2token):
            events = []
            event_start, event_end = 0, 0
            for k, i in enumerate(o2t):
                if token_pred[b][i] % 2 == 1:
                    # case 1-a: find a new start                                                                   
                    if event_start == 0: event_start = i
                    # case 1-b: find a end token (by looking a new start token)
                    else:
                        event_end = o2t[k-1] # find previous end's head token                                               
                        events.append((event_start, event_end))
                        event_start, event_end = i, 0
                        
                elif token_pred[b][i] % 2 == 0 and token_pred[b][i] > 0:
                    # case 2-a: find a new start                                                       
                    if event_start == 0: event_start = i
                    # case 2-b
                    elif not (token_pred[b][i] == token_pred[b][event_start] or
                          token_pred[b][i] == token_pred[b][event_start] + 1):
                        event_end = o2t[k-1] # find previous end's head token                                               
                        events.append((event_start, event_end))
                        event_start, event_end = i, 0
                    
                #case 3: find event ending for multi-token event
                elif event_start > 0 and token_pred[b][i] == 0:
                    event_end = o2t[k-1] # find current end's head token                                                    
                    events.append((event_start, event_end))
                    event_start, event_end = 0, 0
                    
            # case 4: last word or phrase is an event                                
            if event_start > 0:
                events.append((event_start, i))
            batch_events.append(events)
        return batch_events

    
    def construct_events(self, token_pred, orig2token):
        batch_events = []
        for b, o2t in enumerate(orig2token):
            events = []
            event_start, event_end = 0, 0
            for k, i in enumerate(o2t):
                if token_pred[b][i] % 2 == 1:
                    # case 1: find a new start                                                                   
                    if event_start == 0: event_start = i
                    # case 2-a: single token event                                                               
                    else:
                        event_end = o2t[k-1] # find previous end's head token                                    
                        events.append((event_start, event_end))
                        event_start, event_end = i, 0
                #case 2-b: find event ending for multi-token event                                               
                elif event_start > 0 and (token_pred[b][i] == 0 or
                                          token_pred[b][i] != token_pred[b][event_start] + 1):
                    event_end = o2t[k-1] # find current end's head token                                         
                    events.append((event_start, event_end))
                    event_start, event_end = 0, 0
            # case 3: last word or phrase is an event                                                            
            if event_start > 0:
                events.append((event_start, i))
            batch_events.append(events)
        return batch_events

    def check_violation(self, preds):
        for n in range(len(preds) - 1):
            if preds[n] == 0 and preds[n+1] % 2 == 0 and preds[n+1] > 0:
                #print(preds)                                                                                    
                return False
            if preds[n] % 2 == 1 and preds[n+1] > 0 and preds[n+1] % 2 == 0 and abs(preds[n+1] - preds[n]) > 1:
                #print(preds)                                                                                    
                return False
            if preds[n] > 0 and preds[n] % 2 == 0 and preds[n+1] > 0 and preds[n+1] % 2 == 0 and preds[n] != preds[n+1]:
                #print(preds)                                                                                    
                return False
        return True

    def event_inference(self, evt_prob, orig2token, event_constraint_param, event_constraints):
        
        token_predictions = np.argmax(evt_prob, axis=2)
        for b, o2t in enumerate(orig2token):
            # get event head probabilities                                                                       
            prob_evt = np.array([evt_prob[b][o] for o in o2t])
            inference_model = Event_Inference(prob_evt, event_constraint_param, event_constraints)
            prior = copy.deepcopy(inference_model.pred_evt_labels)

            inference_model.run()
            count = inference_model.predict()
            #if prior != inference_model.pred_evt_labels:
            #    assert self.check_violation(inference_model.pred_evt_labels)
            assert prob_evt.shape[0] == len(inference_model.pred_evt_labels)

            # correct local token predictions                                                                    
            for k,v in enumerate(inference_model.pred_evt_labels):
                token_predictions[b][orig2token[b][k]] = v

        return token_predictions

    ## Eventually we want to handle the symmetric label issue!!!                                                 
    def label_lookup(self, pairs, gold_indices, labels):
        new_labels, new_pairs = [], []
        for pair in pairs:
            # one pair should have exactly one match                                                             
            for i, idx in enumerate(gold_indices):
                if partial_match((pair[0], pair[1]), (idx[0], idx[1])) and \
                   partial_match((pair[2], pair[3]), (idx[2], idx[3])):
                    new_labels.append(labels[i])
                    new_pairs.append(pair)
                    break
                if partial_match((pair[0], pair[1]), (idx[2], idx[3])) and \
                   partial_match((pair[2], pair[3]), (idx[0], idx[1])):
                    # quick trick to handle reverse label                                                        
                    new_labels.append(reverse_map_rel[labels[i]])
                    new_pairs.append(pair)
                    break
        assert len(new_labels) <= len(pairs)
        assert len(new_labels) == len(new_pairs)
        return new_labels, new_pairs


    def construct_relations(self, events, indices, lengths, rel_labels, train=True):
        # search gold relations (with partial matched) if they exist                                              

        idx = 0
        new_indices, new_lengths, new_labels = [], [], []
        for b, event_list in enumerate(events):
            idx_end = idx + lengths[b]
            
            all_pairs = [(x[0], x[1], y[0], y[1]) for x, y in combinations(event_list, 2)]

            # using gold matched label for training
            if train:
                matched_labels, new_pairs = self.label_lookup(all_pairs, indices[idx:idx_end],
                                                              rel_labels[idx:idx_end].cpu().numpy())
                new_labels.extend(matched_labels)
                new_indices.extend(new_pairs)
                new_lengths.append(len(new_pairs))
                
            # assign placeholder labels (2: overlap) for inference
            else:
                new_labels.extend([2]*len(all_pairs))
                new_indices.extend(all_pairs)
                new_lengths.append(len(all_pairs))
                
            idx += lengths[b]

        assert len(new_indices) == sum(new_lengths)
        assert len(new_indices) == len(new_labels)
        return new_indices, new_lengths, torch.tensor(new_labels).cuda()

    def forward(self, input_ids, indices, lengths, orig2token, rel_labels=None, labels=None,
                attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                use_gold=True, inference=False, downsample_prob=0.0, train=True,
                event_constraint_param={},  event_constraints={}):

        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)


        outputs = self.dropout(outputs[0])

        loss, loss_rel, logits_rel = 0.0, 0.0, torch.tensor([])

        # Event MLP layer                                                                                        
        logits = self.linear2(self.act(self.linear1(outputs)))
        
        # Event and Relation Construction                                                                        
        if not use_gold:
            if inference:
                token_predictions = self.event_inference(softmax(logits.detach().cpu().numpy(), axis=2),
                                                         orig2token, event_constraint_param, event_constraints)
            else:
                token_predictions = np.argmax(logits.detach().cpu().numpy(), axis=2)

            events = self.construct_events_relaxed(token_predictions, orig2token)
            # new labels never returned out of this function                                                     
            # so we still evaluation against original labels                                                     
            indices, lengths, rel_labels = self.construct_relations(events, indices, lengths,
                                                                    rel_labels, train=train)
        else:
            token_predictions = np.argmax(logits.detach().cpu().numpy(), axis=2)

        if labels is not None:
            loss_fct = CrossEntropyLoss(reduction='none', weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss.reshape(attention_mask.size()) * attention_mask
            loss = loss.reshape(1, -1).sum() / attention_mask.reshape(1, -1).sum()

        # Relation MLP                                                                                           
        idx = 0
        rel_vectors = []
        for b, l in enumerate(lengths):
            for i in range(l):
                l1, l2, r1, r2 = indices[idx]
                rel_vector = torch.cat([outputs[b, l1, :], outputs[b, l2, :],
                                        outputs[b, r1, :], outputs[b, r2, :]])
                rel_vectors.append(rel_vector.unsqueeze(0))
                idx += 1
        assert idx == sum(lengths)

        if rel_vectors:
            rel_vectors = torch.cat(rel_vectors, dim=0)
            outputs_rel = self.act(self.linear1_rel(rel_vectors))
            logits_rel = self.linear2_rel(outputs_rel)

            if rel_labels is not None:
                loss_fct_rel = CrossEntropyLoss(reduction='none', weight=self.rel_class_weights)
                loss_rel = loss_fct_rel(logits_rel.view(-1, self.num_labels_rel), rel_labels)

                # random mask out negative pairs, this is equivalent to down-sample
                rel_mask = mask_negative(rel_labels.detach().cpu().tolist(), downsample_prob)
                rel_mask = torch.tensor(rel_mask).cuda()

                loss_rel = loss_rel * rel_mask
                if rel_mask.sum() == 0:
                    loss_rel = loss_rel.sum()
                else:
                    loss_rel = loss_rel.sum() / rel_mask.sum()

        assert len(indices) == sum(lengths)

        return logits, logits_rel, token_predictions, indices, lengths, loss, loss_rel
