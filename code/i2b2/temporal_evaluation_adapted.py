#!/usr/bin/python 

'''
This TLINK evaluation script is written by Naushad UzZaman for TempEvel 3 evaluation

It is adapted by Weiyi Sun to fit the i2b2 xml format and annotation guidelines.

The changes include:
    get_relation(): modified to fit the i2b2 format
    get_relation_from_dictionary(): added to unify extent ids in the system/gold xmls
    evaluate_two_files(): modified accordingly


'''
# this program evaluates systems that extract temporal information from text 
# tlink -> temporal links

#foreach f (24-a-gold-tlinks/data/ABC19980108.1830.0711.tml); do
#python evaluation-relations/code/temporal_evaluation.py $f $(echo $f | p 's/24-a-gold-tlinks/30-b-trips-relations/g')                             
#done

# DURING relations are changed to SIMULTANEOUS



import time 
import sys
import re 
import os
'''
def get_arg (index):
    #for arg in sys.argv:
    return sys.argv[index]
'''
global_prec_matched = 0 
global_rec_matched = 0 
global_system_total = 0 
global_gold_total = 0 
'''
basedir = re.sub('relation_to_timegraph.py', '', get_arg(0)) 
#debug = int(get_arg(1))
debug=1
cmd_folder = os.path.dirname(basedir)
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder)
    
'''
debug=0

import relation_to_timegraph

def extract_name(filename):
    parts = re.split('/', filename)
    length = len(parts)
    return parts[length-1]

def get_directory_path(path): 
    name = extract_name(path)
    dir = re.sub(name, '', path) 
    if dir == '': 
        dir = './'
    return dir 


def get_entity_val(word, line): 
    if re.search(word+'="[^"]*"', line): 
        entity = re.findall(word+'="[^"]*"', line)[0]
        entity = re.sub(word+'=', '', entity) 
        entity = re.sub('"', '', entity) 
        return entity 
    return word 
        
def change_DURING_relation(filetext): 
    newtext = '' 
    for line in filetext.split('\n'): 
        foo = '' 
        words = line.split('\t') 
        for i in range(0, len(words)): 
            if i == 3 and words[i] == 'DURING': 
                foo += re.sub('DURING', 'SIMULTANEOUS', words[i]) + '\t'
            else:
                foo += words[i] + '\t' 
        newtext += foo.strip() + '\n' 
    return newtext 

def get_relations(tlink_xml,dicsys,dic): 
    '''
    text = open(file).read()
    
    newtext = '' 
    name = extract_name(file) 
    relations = re.findall('<TLINK[^>]*>', text) 
    for each in relations: 
        core = '' 
        ref = '' 
        relType = '' 
        if re.search('eventInstanceID', each): 
            core = get_entity_val('eventInstanceID', each) 
        if re.search('timeID', each): 
            core = get_entity_val('timeID', each) 
        if re.search('relatedToEventInstance', each): 
            ref = get_entity_val('relatedToEventInstance', each) 
        if re.search('relatedToTime', each): 
            ref = get_entity_val('relatedToTime', each) 
        if re.search('relType', each): 
            relType = get_entity_val('relType', each) 
        if core == '' or ref == '' or relType == '': 
            print 'MISSING core, ref or relation', each 
    '''
    # read our xml file instead
    tlinklines=open(tlink_xml).readlines()
    newtext=''
    existing_ids={}
    for tlinkline in tlinklines:
        if re.search('<TLINK', tlinkline):
            re_exp = 'id=\"([^"]*)\"\s+fromID=\"([^"]*)\"\s+fromText=\"([^"]*)\"\s+toID=\"([^"]*)\"\s+toText=\"([^"]*)\"\s+type=\"([^"]*)\"\s+'
            m = re.search(re_exp, tlinkline)
            if m:
                id, core, fromtext, ref, totext, relType = m.groups()
            else:
                raise Exception("Malformed EVENT tag: %s" % (tlinkline))
            dicsys['Admission']='Admission'
            dicsys['Discharge']='Discharge'
            
            if core!='' and ref!='' and relType!='':
                if len(dicsys)<3:
                    test_core=core
                    test_ref=ref

                else:
                    try:
                        test_core_tuple=dicsys[core]
                        test_core=test_core_tuple.split('#@#')[0]
                    except KeyError:
                        test_core=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % core)

                    try:
                        test_ref_tuple=dicsys[ref]
                        test_ref=test_ref_tuple.split('#@#')[0]      
                    except KeyError:
                        test_ref=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % ref)

                if core!='' and ref!='': 
                    try:
                        core=existing_ids[test_core]
                    except KeyError:
                        try:
                            core=dic[test_core].split('#@#')[0]
                            existing_ids[test_core]=core   
                        except KeyError:
                            pass

                    try:
                        ref=existing_ids[test_ref]
                    except KeyError:
                        try:
                            ref=dic[test_ref].split('#@#')[0]
                            existing_ids[test_ref]=ref
                        except KeyError:
                            pass 

                    relType=relType.replace('OVERLAP','SIMULTANEOUS')
                    foo = tlink_xml+'\t'+core+'\t'+ref+'\t'+relType+'\n'
                    if debug >= 3: 
                        print(foo) 
                    newtext += foo 

    #newtext = change_DURING_relation(newtext)
    return newtext

def get_relations_from_dictionary(tlink_xml,dic):
    tlinklines=open(tlink_xml).readlines()
    newtext=''
    existing_ids={}
    count=0
    for tlinkline in tlinklines:
        if re.search('<TLINK', tlinkline):
            re_exp = 'id=\"([^"]*)\"\s+fromID=\"([^"]*)\"\s+fromText=\"([^"]*)\"\s+toID=\"([^"]*)\"\s+toText=\"([^"]*)\"\s+type=\"([^"]*)\"\s+'
            m = re.search(re_exp, tlinkline)
            if m:
                id, core, fromtext, ref, totext, relType = m.groups()
            else:
                raise Exception("Malformed EVENT tag: %s" % (tlinkline))
            count+=1
            
            if core!='' and ref!='':
                if len(dic)<3:
                    gold_core=core
                    gold_ref=ref
                else:
                    try:
                        gold_core_tuple=dic[core]
                        gold_core=gold_core_tuple.split('#@#')[0]
                    except KeyError:
                        gold_core=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % core)
                    try:
                        gore_ref_tuple=dic[ref]   
                        gold_ref=gore_ref_tuple.split('#@#')[0]
                    except KeyError:
                        gold_ref=''
                        print("\n%s" % "Error: Unknown EVENT or TIMEX id in TLINK: %s" % ref)
                if gold_core!='' and gold_ref!='' and relType!='':
                    try:
                        gold_core=existing_ids[core]
                    except KeyError:
                        existing_ids[core]=gold_core
                    try:
                        gold_ref=existing_ids[ref]
                    except KeyError:
                        existing_ids[ref]=gold_ref
                    relType=relType.replace('OVERLAP','SIMULTANEOUS')
                    foo = tlink_xml+'\t'+gold_core+'\t'+gold_ref+'\t'+relType+'\n'
                    if debug >= 3: 
                        print(foo) 
                    newtext += foo 
    #newtext = change_DURING_relation(newtext)
    return newtext

def reverse_relation(rel): 
    rel = re.sub('"', '', rel) 
    if rel.upper() == 'BEFORE': 
        return 'AFTER'
    if rel.upper() == 'AFTER': 
        return 'BEFORE' 
    if rel.upper() == 'IBEFORE': 
        return 'IAFTER' 
    if rel.upper() == 'IAFTER': 
        return 'IBEFORE' 
    if rel.upper() == 'DURING': 
        return 'DURING_BY' 
    if rel.upper() == 'BEGINS': 
        return 'BEGUN_BY' 
    if rel.upper() == 'BEGUN_BY': 
        return 'BEGINS'
    if rel.upper() == 'ENDS': 
        return 'ENDED_BY' 
    if rel.upper() == 'ENDED_BY': 
        return 'ENDS' 
    if rel.upper() == 'INCLUDES': 
        return 'IS_INCLUDED' 
    if rel.upper() == 'IS_INCLUDED': 
        return 'INCLUDES' 
    return rel.upper() 


def get_triples(tlink_file): 
    tlinks = tlink_file # open(tlink_file).read() # tlink_file # 
    relations = '' 
    for line in tlinks.split('\n'): 
        if line.strip() == '': 
            continue 
        words = line.split('\t') 
        relations += words[0]+'\t'+words[1]+'\t'+words[2]+'\t'+words[3]+'\n'
        relations += words[0]+'\t'+words[2]+'\t'+words[1]+'\t'+reverse_relation(words[3]) +'\n'        
    return relations 
        
def get_timegraphs(gold, system): 
    gold_text = gold # open(gold).read() # gold #
    system_text = system # open(system).read() # system # 

    tg_gold = relation_to_timegraph.Timegraph() 
    tg_gold = relation_to_timegraph.create_timegraph_from_weight_sorted_relations(gold_text, tg_gold) 
    tg_gold.final_relations = tg_gold.final_relations + tg_gold.violated_relations
    tg_system = relation_to_timegraph.Timegraph() 
    tg_system = relation_to_timegraph.create_timegraph_from_weight_sorted_relations(system_text, tg_system) 
    tg_system.final_relations = tg_system.final_relations + tg_system.violated_relations
    return tg_gold, tg_system 

  
# extract entities and relation from tlink line 
def get_x_y_rel(tlinks): 
    words = tlinks.split('\t')
    x = words[1]
    y = words[2]
    rel = words[3]
    return x, y, rel 

def get_entity_rel(tlink): 
    words = tlink.split('\t') 
    if len(words) == 3: 
        return words[0]+'\t'+words[1]+'\t'+words[2] 
    return words[1]+'\t'+words[2]+'\t'+words[3] 

def total_relation_matched(A_tlinks, B_tlinks, B_relations, B_tg): 
    count = 0 
    for tlink in A_tlinks.split('\n'): 
        if tlink.strip() == '': 
            continue 
        if debug >= 2: 
            print(tlink)
        x, y, rel = get_x_y_rel(tlink) 
        foo = relation_to_timegraph.interval_rel_X_Y(x, y, B_tg, rel, 'evaluation')
        if re.search(get_entity_rel(tlink.strip()), B_relations): 
            count += 1 
            if debug >= 2: 
                print('True')
            continue 
        if debug >= 2: 
            print(x, y, rel, foo[1])
        if re.search('true', foo[1]):
            count += 1 
    return count 
           
def total_implicit_matched(system_reduced, gold_reduced, gold_tg): 
    count = 0 
    for tlink in system_reduced.split('\n'): 
        if tlink.strip() == '': 
            continue 
        if debug >= 2: 
            print(tlink)
        if re.search(tlink, gold_reduced): 
            continue 

        x, y, rel = get_x_y_rel(tlink) 
        foo = relation_to_timegraph.interval_rel_X_Y(x, y, gold_tg, rel, 'evaluation')
        if debug >= 2: 
            print(x, y, rel, foo[1])
        if re.search('true', foo[1]):
            count += 1 
    return count 
    
 
def get_entities(relations): 
    included = '' 
    for each in relations.split('\n'): 
        if each.strip() == '': 
            continue 
        words = each.split('\t')
        if not re.search('#'+words[1]+'#', included):
            included += '#'+words[1]+'#\n'
        if not re.search('#'+words[2]+'#', included):
            included += '#'+words[2]+'#\n'
    return included

def get_n(relations): 
    included = get_entities(relations) 
    return (len(included.split('\n'))-1)

def get_common_n(gold_relations, system_relations): 
    gold_entities = get_entities(gold_relations) 
    system_entities = get_entities(system_relations) 
    common = '' 
    for each in gold_entities.split('\n'): 
        if each.strip() == '':
            continue 
        if re.search(each, system_entities): 
            common += each + '\n' 
    if debug >= 3: 
        print(len(gold_entities.split('\n')), len(system_entities.split('\n')), len(common.split('\n')))
        print(common.split('\n'))
        print(gold_entities.split('\n'))
    return (len(common.split('\n'))-1)

def get_ref_minus(gold_relation, system_relations): 
    system_entities = get_entities(system_relations)
    count = 0 
    for each in gold_relation.split('\n'): 
        if each.strip() == '': 
            continue 
        words = each.split('\t')
        if re.search('#'+words[1]+'#', system_entities) and re.search('#'+words[2]+'#', system_entities):
            count += 1 
    return count 

def evaluate_two_files(gold_annotation, system_annotation):

    global global_prec_matched
    global global_rec_matched
    global global_system_total
    global global_gold_total

    #if debug >= 1: 
    #    print('\n\n Evaluate', arg1, arg2)
    #gold_annotation= get_relations(arg1,dicsys,dic)
    #system_annotation = get_relations_from_dictionary(arg2,dic) 
    
    tg_gold, tg_system = get_timegraphs(gold_annotation, system_annotation) 
    gold_relations = get_triples(gold_annotation) 
    system_relations = get_triples(system_annotation) 
    #for precision
    gold_count=len(tg_gold.final_relations.split('\n'))-1
    sys_count=len(tg_system.final_relations.split('\n'))-1
    if debug >= 2: 
        print('\nchecking precision')
    prec_matched = total_relation_matched(tg_system.final_relations, tg_gold.final_relations, gold_relations, tg_gold) 
    # for recall 
    if debug >= 2: 
        print('\nchecking recall')
    rec_matched = total_relation_matched(tg_gold.final_relations, tg_system.final_relations, system_relations, tg_system) 
    rec_implicit_matched = total_implicit_matched(tg_system.final_relations, tg_gold.final_relations, tg_gold) 
    n = get_common_n(tg_gold.final_relations, tg_system.final_relations) 
    
##    n = get_n(tg_gold.final_relations)
    ref_plus = 0.5*n*(n-1)
##    ref_minus = len(tg_gold.final_relations.split('\n'))-1
    ref_minus = rec_matched ## get_ref_minus(tg_gold.final_relations, tg_system.final_relations) 
    w = 0.99/(1+ref_plus-ref_minus) # ref_minus #
    if debug >= 2: 
        print('n =', n)
        print('rec_implicit_matched', rec_implicit_matched)
        print('n, ref_plus, ref_minus', n , ref_plus , ref_minus)
        print('w', w)
        print('rec_matched', rec_matched)
        print('total', gold_count)

        print('w*rec_implicit_matched', w*rec_implicit_matched)

    if debug >= 2: 
        print('precision', prec_matched, sys_count)
    if gold_count <= 0: 
        precision = 0 
    else: 
        precision = prec_matched*1.0/gold_count

    if debug >= 2: 
        print('recall', rec_matched, len(tg_gold.final_relations.split('\n'))-1)
    if len(tg_gold.final_relations.split('\n')) <= 1: 
        recall = 0 
    else:
        recall2 = (rec_matched)*1.0/gold_count
        recall = (rec_matched+w*rec_implicit_matched)*1.0/gold_count
        if debug >= 2: 
            print('recall2', recall2)
            print('recall', recall)
    
    if debug >= 1: 
        print(precision, recall, get_fscore(precision, recall))
    global_prec_matched += prec_matched
    global_rec_matched += rec_matched+w*rec_implicit_matched
    global_system_total += sys_count 
    global_gold_total += len(tg_gold.final_relations.split('\n'))-1

    #return tg_system
    return sys_count, len(tg_gold.final_relations.split('\n'))-1, prec_matched, rec_matched


"""
count_relation = 0 
count_node = 0 
count_chains = 0 
count_time = 0 
"""

def get_fscore(p, r): 
    if p+r == 0: 
        return 0 
    return 2.0*p*r/(p+r) 


def final_score(): 
    global global_prec_matched
    global global_rec_matched
    global global_system_total
    global global_gold_total 

    if global_system_total == 0: 
        precision = 0 
    else: 
        precision = global_prec_matched*1.0/global_system_total
    if global_gold_total == 0: 
        recall = 0
    else: 
        recall = global_rec_matched*1.0/global_gold_total
    
    if precision == 0 and recall == 0: 
        fscore = 0 
    else: 
        fscore = get_fscore(precision, recall) 
    print('Overall\tP\tR\tF1')
    print('\t'+str(100*round(precision, 6))+'\t'+str(100*round(recall, 6))+'\t'+str(100*round(fscore, 6)))

