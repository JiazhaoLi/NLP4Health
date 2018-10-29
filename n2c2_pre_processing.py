import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import isfile, join
import nltk
import string
from tqdm import tqdm
import pdb

max_length_all = 342

def ann_helper(term_filename):
    """
        for each .ann file, for each line: 
            1. T -> term_dict 
            2. R -> relation_dict
    """
    # generate the Term dictionary()
    term_examples = list(open(term_filename, 'r').readlines())
    term_dict = {} # dict for T: word
    pos_type = {}  # dict pos: type
    term_pos = {}  #  dict term and postion 
    index_ann = {} # dict pos relationship
    index_rela = {}
    relation_pair = {}
    # get the term dict
    for line in term_examples:
        line = line.strip().split('\t')
        line[1] = line[1].split(' ')
        if line[0][0] == 'T':   # this is the term 
            if len(line[1]) == 3:
                pos_range = list(range(int(line[1][1]), int(line[1][2])))
                for pos in pos_range:
                    pos_type[str(pos)] = line[1][0]
                term_dict[line[0]] = line[2]
                term_pos[line[0]] = [int(line[1][1]), int(line[1][2])]
            if len(line[1]) == 4:
                line[1][2] = line[1][2].split(';') 
                pos_range = list(range(int(line[1][1]), int(line[1][2][0])))
                for pos in pos_range:
                    pos_type[str(pos)] = line[1][0] 
                pos_range = list(range(int(line[1][2][1]), int(line[1][3])))
                for pos in pos_range:
                    pos_type[str(pos)] = line[1][0]
                term_dict[line[0]] = line[2]
                term_pos[line[0]] = [int(line[1][1]), int(line[1][3])]
            if len(line[1]) == 5:
                line[1][2] = line[1][2].split(';') 
                line[1][3] = line[1][3].split(';') 
                pos_range = list(range(int(line[1][1]), int(line[1][2][0])))
                for pos in pos_range:
                    pos_type[str(pos)] = line[1][0] 
                pos_range = list(range(int(line[1][2][1]), int(line[1][3][0])))
                for pos in pos_range:
                    pos_type[str(pos)] = line[1][0]
                pos_range = list(range(int(line[1][3][1]), int(line[1][4])))
                for pos in pos_range:
                    pos_type[str(pos)] = line[1][0]
                term_dict[line[0]] = line[2]
                term_pos[line[0]] = [int(line[1][1]), int(line[1][4])]

    
    # get the relation term
    relation_index = 0 
    for line in term_examples:
        line = line.strip().split('\t')
        line[1] = line[1].split(' ')
        if line[0][0] == 'R':   # this is the term
            
            relation_index += 1
            relation_type = line[1][0]
            term1_ann = line[1][1].split(':')[1]
            term1_pos = list(range(term_pos[term1_ann][0], term_pos[term1_ann][1]))
            for pos in term1_pos:
                index_ann[pos] = term1_ann
                if pos not in index_rela:
                    index_rela[pos] = line[0]
                else:
                    relation = index_rela[pos] + ':' + line[0]
                    index_rela[pos] = relation

            term2_ann = line[1][2].split(':')[1]
            
            term2_pos = list(range(term_pos[term2_ann][0], term_pos[term2_ann][1]))
            for pos in term2_pos:
                index_ann[pos] = term2_ann
                if pos not in index_rela:
                    index_rela[pos] = line[0]
                else:
                    relation = index_rela[pos] + ':' + line[0]
                    index_rela[pos] = relation

            relation_pair[line[0]] = [relation_type, term1_ann, term2_ann]
            # relation_pos[] = 'R'+str(relation_index)
    # print(relation_pair)
    return term_dict, pos_type, term_pos, index_ann, index_rela, relation_pair


def text_helper(term_filename, text_filename,relation_fre):
    term_dict, pos_type, term_pos, index_ann, index_rela, relation_pair = ann_helper(term_filename)
    # find the postion in the text
    text_examples = list(open(text_filename, 'r').readlines())
    text = ''
    for line in text_examples:
        text = text + line
    token_map_list = []
    token_list = nltk.word_tokenize(text)
    pos_tag = nltk.pos_tag(token_list)
    start = 0 
    for index in range(len(token_list)):
        if str(token_list[index]) == '\'\'' or str(token_list[index]) == '``':
            offset_s = 0
            offset_e = len(token_list[index])
        else:
            offset_s = text[start:].index(token_list[index])
            offset_e = offset_s + len(token_list[index])
        # get the NER tag
        if str(start+offset_s) in pos_type:
            NER_type = pos_type[str(start+offset_s)]
        elif str(start+offset_e) in pos_type:
            NER_type = pos_type[str(start+offset_e)]
        else:
            NER_type = 'O'
        
        # get the relationship 
        if(start + offset_s) in index_ann:
            ann_type = index_ann[(start + offset_s)]
            relation = index_rela[(start + offset_s)]
        else:
            ann_type = 'None'
            relation = 'None'
        token_map_list.append([index, token_list[index], start + offset_s, start+ offset_e, pos_tag[index][1], NER_type, ann_type, str(relation)])
        start = offset_s + start
    for term in token_map_list:
        start_off = term[2]
        end_off = term[3]
        
    # generate the embedding of 
    output_name = ('./output/{}.tsv'.format(term_filename.split('/')[5].split('.')[0]))
    with open(output_name, 'w') as f:
        f.write('inedex\tword\tstart\tend\tPOS\tNER\tNotation\tRelation'+'\n')
        index = 0
        for pair in token_map_list:
            f.write(str(index)+'\t'+pair[1]+'\t'+str(pair[2])+'\t'+str(pair[3])+'\t'+pair[4]+'\t'+pair[5]+'\t'+pair[6]+'\t'+pair[7]+'\n')
            index += 1
    # get the range of each relationship 
    relation_list, train_exmaple_indexrange = generate_train_sample(token_map_list)
    
    train_all_sentence = []
    for i in range(len(relation_list)):
        train_sentence_feature = [] 
        # loop for all realation ship in this note
        r_index = relation_list[i]
        relation_label = relation_pair[r_index]
        
        if relation_label[0] not in relation_fre:
            relation_fre[relation_label[0]] = 1
        else:
            relation_fre[relation_label[0]] += 1
        # print(relation_label)
        start = int(train_exmaple_indexrange[i][0])
        end = int(train_exmaple_indexrange[i][1])
        sentence = (token_map_list[start:end]) # this is one realtionship
        # print(term_filename)
        # print(sentence)
        # get the position of phase position
        # print((token_map_list[start:end])) # this is one realtionship)
       
       
    #     relative_pos1 = []
    #     relative_pos2 = []
    #     index = 0
    #     len_sentence = (len(sentence))
    #     # print(sentence)
    #     # print(relation_label[2])
    #     for sen in sentence:
    #         if sen[6] == str(relation_label[1]):
    #             relative_pos1.append(index)
    #         if sen[6] == str(relation_label[2]):
    #             relative_pos2.append(index)
    #         index += 1
    #     # print(relative_pos2)
    #     # get two relative position list
    #     relativepos_list1 = []
        
    #     for j in range(len(sentence)):
    #         if j  <  relative_pos1[0]: 
    #             relativepos_list1.append( j - relative_pos1[0])
    #         elif j in relative_pos1:
    #             relativepos_list1.append(0)
    #         else:
    #             relativepos_list1.append( j - relative_pos1[-1])
        
    #     relativepos_list2 = []
    #     for j in range(len(sentence)):
    #         # print(relative_pos2)
    #         if j  <  relative_pos2[0] : 
    #             relativepos_list2.append( j - relative_pos2[0])
    #         elif j in relative_pos2:
    #             relativepos_list2.append(0)
    #         else:
    #             relativepos_list2.append( j - relative_pos2[-1])
    #     for index in range(len_sentence):
    #         train_sentence_feature.append([sentence[index][1],relativepos_list1[index],relativepos_list2[index],sentence[index][4],sentence[index][5]])
    #     train_all_sentence.append(train_sentence_feature) 
    # # print(len(train_all_sentence))

    return token_map_list


def generate_train_sample(token_map_list):
    relation_range = {}
    relation_fre = {}
    for token in token_map_list:
        index = token[0]
        Relation = token[7]
        
        if Relation != 'None':
            Relation = Relation.split(':')
            for rela in Relation:
                if rela not in relation_range:
                    min_index = index
                    max_index = index
                    relation_range[rela] = str(min_index) + ':' +  str(max_index)
                else:
                    min_index = min(int(index), int(relation_range[rela].split(':')[0]))
                    max_index = max(int(index), int(relation_range[rela].split(':')[1]))
                    relation_range[rela] = str(min_index) + ':' +  str(max_index)   
                
    # maxlength = 0    
    # for r, index_range in relation_range.items():
    #     index_range = index_range.split(':')
    #     maxlength = max(maxlength,int(index_range[1]) - int(index_range[0]))
    
    train_exmaple = []
    relation_list = []
    for r, index_range in relation_range.items():
        index_range = index_range.split(':')
        context_length = 1/2 * (max_length_all -  (int(index_range[1]) - int(index_range[0])))
        relation_list.append(r)
        train_exmaple.append([np.floor(int(index_range[0]) - context_length), np.ceil(int(index_range[1]) + context_length)])
        # print(context_length)
        # print(r)
        # print(train_exmaple)
        # print(index_range)
    #     print(np.floor(int(index_range[0]) - context_length)- np.floor(int(index_range[1]) + context_length))
    # exit()
    return  relation_list, train_exmaple

def load_data(train_datapath1,train_datapath2):
    max_length_all = 0
    relation_fre = {}
    file_list = [f for f in listdir(train_datapath1) if isfile(join(train_datapath1, f))]
    for index in tqdm(range(len(file_list))):
        file = file_list[int(index)]
        if file.endswith('.ann'):  
            term_filename = train_datapath1 + file
            text_filename = train_datapath1 + file.split('.')[0]+'.txt'
            token_map_list = text_helper(term_filename,text_filename,relation_fre)
            # print(token_map_list)

            relation_list, train_exmaple = generate_train_sample(token_map_list)
            # genretate the training sample 
    print(relation_fre)
           
    
                

            