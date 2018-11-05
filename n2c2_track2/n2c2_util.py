import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import isfile, join
import nltk
import os
import string
from tqdm import tqdm
import pdb
from mimic_pre_process import *
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import warnings
from mimic_pre_process import *
import warnings
import random
from sklearn.preprocessing import OneHotEncoder
import pickle
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
import tensorflow as tf



warnings.simplefilter(action='ignore', category=FutureWarning)

max_length_all = 50

def train_ann_helper(term_filename):
    """
        for each .ann file, for each line: 
            1. T -> nota_word 
            2. R -> relation_dict
    """
    # generate the Term dictionary()
    term_examples = list(open(term_filename, 'r').readlines())
    nota_word = {} # dict for T: word
    pos_nota = {}  # dict pos: type
    nota_range = {}  #  dict term and postion 
    index_ann = {} # dict pos relationship
    index_rela = {}
    relation_pair = {}
    # get the Notation information
    for line in term_examples:
        line = line.strip().split('\t')  # line = [[T61],[Strength 8758 8762],[40mg]]
        line[1] = line[1].split(' ')         #  line[1]= [[strength],[8758].[8762]]
        if line[0][0] == 'T':   # this is the term 

            if len(line[1]) == 3:           #line[1] = [[strength], [8758], [8762]]
                # for each pos we name a entity name 
                pos_range = list(range(int(line[1][1]), int(line[1][2])))
                for pos in pos_range:
                    pos_nota[str(pos)] = line[1][0]  # this is the entity label 
                # this is the notation_word    delete?
                nota_word[line[0]] = line[2]         # this is the word 
                # get the range of each notation
                nota_range[line[0]] = [int(line[1][1]), int(line[1][2])] # this is the temr range

            if len(line[1]) == 4:           #line[1] = [[strength], [8758], [8762:222],[454]]
                # for each pos we name a entity name 
                line[1][2] = line[1][2].split(';') 
                pos_range = list(range(int(line[1][1]), int(line[1][2][0])))
                for pos in pos_range:        
                    pos_nota[str(pos)] = line[1][0] 
                pos_range = list(range(int(line[1][2][1]), int(line[1][3])))
                for pos in pos_range:
                    pos_nota[str(pos)] = line[1][0]
                # this is the notation_word    delete?
                nota_word[line[0]] = line[2]
                # get the range of each notation
                nota_range[line[0]] = [int(line[1][1]), int(line[1][3])]

            if len(line[1]) == 5:
                # for each pos we name a entity name 
                line[1][2] = line[1][2].split(';') 
                line[1][3] = line[1][3].split(';') 
                pos_range = list(range(int(line[1][1]), int(line[1][2][0])))
                for pos in pos_range:
                    pos_nota[str(pos)] = line[1][0] 
                pos_range = list(range(int(line[1][2][1]), int(line[1][3][0])))
                for pos in pos_range:
                    pos_nota[str(pos)] = line[1][0]
                pos_range = list(range(int(line[1][3][1]), int(line[1][4])))
                for pos in pos_range:
                    pos_nota[str(pos)] = line[1][0]
                # this is the notation_word    delete?
                nota_word[line[0]] = line[2]
                 # get the range of each notation
                nota_range[line[0]] = [int(line[1][1]), int(line[1][4])]

    # get the relation term
    for line in term_examples:
        line = line.strip().split('\t')
        line[1] = line[1].split(' ')
        if line[0][0] == 'R':   #   [[R1] [Strength-Drug Arg1:T5 Arg2:T6]]
            relation_label = line[1][0]  # line[1] = [Strength-Drug Arg1:T5 Arg2:T6]]

            term1_ann = line[1][1].split(':')[1]
            term1_pos = list(range(nota_range[term1_ann][0], nota_range[term1_ann][1]))
            for pos in term1_pos:
                if pos not in index_ann:
                    index_ann[pos] = term1_ann
                elif term1_ann not in index_ann[pos]:
                    ann = index_ann[pos] + ':' + term1_ann
                    index_ann[pos] = ann

                if pos not in index_rela:
                    index_rela[pos] = line[0]
                elif line[0] not in index_rela[pos]:
                    relation = index_rela[pos] + ':' + line[0]
                    index_rela[pos] = relation

            term2_ann = line[1][2].split(':')[1]
            term2_pos = list(range(nota_range[term2_ann][0], nota_range[term2_ann][1]))
            for pos in term2_pos:
                if pos not in index_ann:
                    index_ann[pos] = term2_ann
                elif term2_ann not in index_ann[pos]:
                    ann = index_ann[pos] + ':' + term2_ann
                    index_ann[pos] = ann

                if pos not in index_rela:
                    index_rela[pos] = line[0]
                elif line[0] not in index_rela[pos]:
                    relation = index_rela[pos] + ':' + line[0]
                    index_rela[pos] = relation
            relation_pair[line[0]] = [relation_label, term1_ann, term2_ann]

    return nota_word, pos_nota, nota_range, index_ann, index_rela, relation_pair


def test_ann_helper(term_filename):
    """
        for each .ann file, for each line: 
            1. T -> nota_word 
            2. R -> relation_dict
    """
    # generate the Term dictionary()
    term_examples = list(open(term_filename, 'r').readlines())
    nota_word = {} # dict for T: word
    pos_nota = {}  # dict pos: type
    nota_range = {}  #  dict term and postion 
    pos_T= {}
    # get the Notation information
    for line in term_examples:
        line = line.strip().split('\t')  # line = [[T61],[Strength 8758 8762],[40mg]]
        line[1] = line[1].split(' ')         #  line[1]= [[strength],[8758].[8762]]
        if line[0][0] == 'T':   # this is the term 

            if len(line[1]) == 3:           #line[1] = [[strength], [8758], [8762]]
                # for each pos we name a entity name 
                pos_range = list(range(int(line[1][1]), int(line[1][2])))
                for pos in pos_range:
                    pos_nota[str(pos)] = line[1][0]  # this is the entity label 
                    pos_T[str(pos)] = line[0]
                # this is the notation_word    delete?
                nota_word[line[0]] = line[2]         # this is the word 
                # get the range of each notation
                nota_range[line[0]] = [int(line[1][1]), int(line[1][2])] # this is the temr range

            if len(line[1]) == 4:           #line[1] = [[strength], [8758], [8762:222],[454]]
                # for each pos we name a entity name 
                line[1][2] = line[1][2].split(';') 
                pos_range = list(range(int(line[1][1]), int(line[1][2][0])))
                for pos in pos_range:   
                    pos_T[str(pos)] = line[0]     
                    pos_nota[str(pos)] = line[1][0] 
                pos_range = list(range(int(line[1][2][1]), int(line[1][3])))
                for pos in pos_range:
                    pos_T[str(pos)] = line[0]
                    pos_nota[str(pos)] = line[1][0]
                # this is the notation_word    delete?
                nota_word[line[0]] = line[2]
                # get the range of each notation
                nota_range[line[0]] = [int(line[1][1]), int(line[1][3])]

            if len(line[1]) == 5:
                # for each pos we name a entity name 
                line[1][2] = line[1][2].split(';') 
                line[1][3] = line[1][3].split(';') 
                pos_range = list(range(int(line[1][1]), int(line[1][2][0])))
                for pos in pos_range:
                    pos_T[str(pos)] = line[0]
                    pos_nota[str(pos)] = line[1][0] 
                pos_range = list(range(int(line[1][2][1]), int(line[1][3][0])))
                for pos in pos_range:
                    pos_T[str(pos)] = line[0]
                    pos_nota[str(pos)] = line[1][0]
                pos_range = list(range(int(line[1][3][1]), int(line[1][4])))
                for pos in pos_range:
                    pos_T[str(pos)] = line[0]
                    pos_nota[str(pos)] = line[1][0]
                # this is the notation_word    delete?
                nota_word[line[0]] = line[2]
                 # get the range of each notation
                nota_range[line[0]] = [int(line[1][1]), int(line[1][4])]
    # print(pos_T)
    return nota_word, pos_nota, nota_range,pos_T


def train_text_helper(term_filename, text_filename):
    nota_word, pos_nota, nota_range, index_ann, index_rela, relation_pair = train_ann_helper(term_filename)
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
        word_range = list(range(start+offset_s, start + offset_e))

        # get the NER tag
        NER_type = []
        for pos in word_range:
            if str(pos) in pos_nota:
                NER_type.append(str(pos_nota[str(pos)]))
            else:
                NER_type.append('O')
        
        # get the relationship 
        ann_type = []
        relation = []
        for pos in word_range:
            if pos in index_ann:
                ann_type.append(str(index_ann[pos]))
                relation.append(str(index_rela[pos]))
            else:
                ann_type.append('None')
                relation.append('None')
        
        # filter 
        if len(list(set(NER_type))) > 1 and 'O' in NER_type:
            NER_type.remove('O')
        NER_list = ':'.join(x for x in list(set(NER_type)))

        if len(list(set(ann_type))) > 1 and 'None' in ann_type:
            ann_type.remove('None')
        ann_type_list = ':'.join(x for x in list(set(ann_type)))

        if len(list(set(relation))) > 1 and 'None' in relation:
            relation.remove('None')
        relation_list = ':'.join(x for x in list(set(relation)))

        token_map_list.append([index, token_list[index], start + offset_s, start+ offset_e, pos_tag[index][1], \
                                 NER_list,ann_type_list,relation_list])                          
        start = offset_s + start
    for term in token_map_list:
        start_off = term[2]
        end_off = term[3]
    # generate the embedding of 
    output_name = ('./train_output/{}.tsv'.format(term_filename.split('/')[5].split('.')[0]))
    if os.path.exists(output_name):
        pass
    else:
        with open(output_name, 'w') as f:
            f.write('inedex\tword\tstart\tend\tPOS\tNER\tNotation\tRelation'+'\n')
            index = 0
            for pair in token_map_list:
                f.write(str(index)+'\t'+pair[1]+'\t'+str(pair[2])+'\t'+str(pair[3])+'\t'+pair[4]+'\t'+pair[5]+'\t'+pair[6]+'\t'+pair[7]+'\n')
                index += 1
    
    return token_map_list,relation_pair


def test_text_helper(term_filename, text_filename):
    nota_word, pos_nota, nota_range,pos_T = test_ann_helper(term_filename)
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
        word_range = list(range(start+offset_s, start + offset_e))

        # get the NER tag
        NER_type = []
        for pos in word_range:
            if str(pos) in pos_nota:
                NER_type.append(str(pos_nota[str(pos)]))
            else:
                NER_type.append('O')

        ann_type = []
        for pos in word_range:
            if str(pos) in pos_T:
                ann_type.append(str(pos_T[str(pos)]))
            else:
                ann_type.append('None')
        
        # filter 
        if len(list(set(NER_type))) > 1 and 'O' in NER_type:
            NER_type.remove('O')
        NER_list = ':'.join(x for x in list(set(NER_type)))

        if len(list(set(ann_type))) > 1 and 'None' in ann_type:
            ann_type.remove('None')
        ann_type_list = ':'.join(x for x in list(set(ann_type)))

        token_map_list.append([index, token_list[index], start + offset_s, start+ offset_e, pos_tag[index][1], \
                                 NER_list,ann_type_list])          
        start = offset_s + start

    for term in token_map_list:
        start_off = term[2]
        end_off = term[3]
    # generate the embedding of 
    output_name = ('./test_output/{}.tsv'.format(term_filename.split('/')[5].split('.')[0]))
    if os.path.exists(output_name):
        pass
    else:
        with open(output_name, 'w') as f:
            f.write('inedex\tword\tstart\tend\tPOS\tNER\tT'+'\n')
            index = 0
            for pair in token_map_list:
                f.write(str(index)+'\t'+pair[1]+'\t'+str(pair[2])+'\t'+str(pair[3])+'\t'+pair[4]+'\t'+pair[5]+'\t'+pair[6]+'\t'+'\n')
                index += 1
    
    return token_map_list
    

def generate_train_possiblepair_sample(token_map_list,relation_pair):
    global max_length_all
    relation_range = {}
    Stren_Drug_trainset = []
    sentence_index = 0
    """
        find all entities position split drug and others entities out
    """
    drug_pos = {}
    entities_pos = {}
    for sentence in token_map_list:
        #  0     1        2      3       4     5          6      7
        # 3099	dialy	15652	15657	VB	Frequency	T233	R145
        # split Drug and others get the posistion range of each entities (with : split)
        if sentence[6] != 'None':   # for all entities 
            if sentence[5] == 'Drug':  # for drug  
                if sentence[6] not in drug_pos:
                    drug_pos[sentence[6]] = str(sentence[0])
                else:
                    start = drug_pos[sentence[6]]
                    drug_pos[sentence[6]] = start+':'+ str(sentence[0])
            else:   # for all other entities 
                if sentence[6] not in entities_pos:
                    entities_pos[sentence[6]] = str(sentence[0])
                else:
                    start = entities_pos[sentence[6]]
                    entities_pos[sentence[6]] = start+':'+str(sentence[0])
    """
        for each drug find possible pairs: 
    """
    # streng_list = strength_pos.keys()
    entities_list = entities_pos.keys() 
    drug_pair = {}
    drug_pair_range = {}
    for drug_t, drug_pos_ in drug_pos.items():
        possible_pair = []
        # the length of 50:
        if ':' in drug_pos_:
            drug_pos_list = drug_pos_.split(':')
            drug_start = int(drug_pos_list[0])
            drug_end = int(drug_pos_list[-1])
        else:
            drug_start = int(drug_pos_)
            drug_end = int(drug_pos_)
        context_length =  1.0 /2 * (max_length_all - (drug_end - drug_start))
        possible_range = range(int(np.floor(drug_start - context_length)), int(np.ceil(drug_end) + context_length))
       
       # for each drug and its possible range get possible pair
        for entities in entities_list:
            if ':' in entities_pos[entities]:
                pos_list = entities_pos[entities].split(':')
                en_start = int(pos_list[0])
                en_end = int(pos_list[-1])
            else:
                en_start = entities_pos[entities]
                en_end = entities_pos[entities]
            if  int(en_start) in possible_range and  int(en_end) in possible_range:
                possible_pair.append(entities)
        drug_pair[drug_t] = possible_pair

    """
        filter ground truth with label and other as negative label
        get the ground truth and genearte positiv and negetive sample
        train_feature_sample  ['T1:T2']
        train_label_sample      ['Reason-Drug'] ['None']
    """
    ground_truth_dict = {}  
    #relation_pair
    #[relation_label, term1_ann, term2_ann]
    # all grounde truth 
    for k, pair in relation_pair.items():
        ground_truth_dict[':'.join(pair[1:])] = pair[0]
        #   ['T1:T2'] = Reason-Drug

    train_feature_sample = []
    train_label_sample = []
    for drug, possible in drug_pair.items():
        for piss in possible:    # loop all possible      # if in relation                
            if ':'.join([piss,drug]) in ground_truth_dict:      # if is lable
                if [piss,drug] not in train_feature_sample:
                    train_feature_sample.append([piss,drug])
                    train_label_sample.append(ground_truth_dict[':'.join([piss,drug])])
            else:                 # if not lable but a relation
                if [piss,drug] not in train_feature_sample:
                    train_feature_sample.append([piss,drug])
                    train_label_sample.append('None')
    
    '''
        get range of each sample
    '''
    train_context_range = []
    for sample in train_feature_sample:
        # loop for each instance
        if ':' not in entities_pos[sample[0]]:
            min_e_index = int(entities_pos[sample[0]])
            max_e_index = int(entities_pos[sample[0]])
        else:
            entities_pos_l = entities_pos[sample[0]].split(':')
            min_e_index = min([int(x) for x in entities_pos_l])
            max_e_index = max([int(x) for x in entities_pos_l])
            
        if ':' not in drug_pos[sample[1]]:
            min_d_index = int(drug_pos[sample[1]])
            max_d_index = int(drug_pos[sample[1]])
        else:
            drug_pos_l = drug_pos[sample[1]].split(':')
            min_d_index = min([int(x) for x in drug_pos_l])
            max_d_index = max([int(x) for x in drug_pos_l]) 
        
        min_index = min(min_e_index,min_d_index)
        max_index = max(max_e_index,max_d_index)

        context_length =  1.0 /2 * (max_length_all - (max_index - min_index))
        possible_range = [int(np.floor(min_index - context_length)), int(np.ceil(max_index + context_length))]
        train_context_range.append([[sample[0],sample[1]],possible_range])
        final_train_feature = add_relative_features(train_context_range,token_map_list)

    return final_train_feature,train_label_sample



def add_relative_features(train_context_range,token_map_list):
    train_all_sentence = []
    for sample in train_context_range:
        relation_label = sample[0]
        start = max(0,int(sample[1][0]))
        end = min(len(token_map_list),sample[1][1])
        sentence = (token_map_list[start:end]) # this is one realtionship
        
        relative_pos1 = []
        relative_pos2 = []
        len_sentence = (len(sentence))
        index = 0
        for sen in sentence:
            if str(relation_label[0]) in sen[6]:
                relative_pos1.append(index)
            if str(relation_label[1]) in sen[6]:
                relative_pos2.append(index)
            index += 1
        # get two relative position list
        relativepos_list1 = []
        for j in range(len(sentence)):
            if j  <  relative_pos1[0]: 
                relativepos_list1.append( j - relative_pos1[0])
            elif j in relative_pos1:
                relativepos_list1.append(0)
            else:
                relativepos_list1.append( j - relative_pos1[-1])
        
        relativepos_list2 = []
        for j in range(len(sentence)):
            # print(relative_pos2)
            if j  <  relative_pos2[0]: 
                relativepos_list2.append( j - relative_pos2[0])
            elif j in relative_pos2:
                relativepos_list2.append(0)
            else:
                relativepos_list2.append( j - relative_pos2[-1])

        train_sentence_feature = []
        for index in range(len(sentence)):
            train_sentence_feature.append([sentence[index][1],relativepos_list1[index],relativepos_list2[index],sentence[index][4],sentence[index][5]])
        train_all_sentence.append(train_sentence_feature) 
        
    return train_all_sentence

def embedding_NER_POS(Streng_Drug_allsample_feature):
    # 
    possible_entity = ['Strength', 'Frequency', 'Reason', 'Route', 'Dosage', 'Form', 'ADE', 'Duration']

    NER_dict = {}
    ner_index = 1
    """ 
        get the NER_index dict
    """
    for instance  in Streng_Drug_allsample_feature: # for each stence 
        #['Cholecalciferol', 21, 17, 'NNP', 'Drug']
        sample_patch = np.array(instance)
        for word in sample_patch[:, 4]:
            if ":" in word: # more than one NER
                word = word.split(':')
                if 'O' in word:
                    word.remove('O')   # remove O
                if 'Drug' in word:
                    word = 'Drug'
                # if sub_label in word:
                #     word = sub_label
                if type(word)==list:
                    word = word[0]
            if word not in NER_dict:
                NER_dict[word] = ner_index
                ner_index +=1
    for key,pair in NER_dict.items():
        array_ve = np.zeros([1,len(NER_dict)])
        array_ve[0,pair-1] = 1
        NER_dict[key] = array_ve
    
    """
        get the POS_index dict
    """
    POS_dict = {}
    pos_index = 1
    for instance  in Streng_Drug_allsample_feature: # for each stence 
        sample_patch = np.array(instance)
        for word in sample_patch[:,3]:
            if word not in POS_dict:
                POS_dict[word] = pos_index
                pos_index +=1
    for key,pair in POS_dict.items():
        array_ve = np.zeros([1,len(POS_dict)])
        array_ve[0,pair-1] = 1
        POS_dict[key] = array_ve

    with open('train_pair/NER_POS_embed.pickle', 'wb') as f:
        pickle.dump([NER_dict, POS_dict], f, pickle.HIGHEST_PROTOCOL)

    
def embedding_features(Streng_Drug_allsample_feature,label):
    # 
    sub_label = label.split('-')[0]
    model = Word2Vec.load('word2vec_50000notes_1008')
    
    with open('train_pair/NER_POS_embed.pickle', 'rb') as f:
        [NER_dict, POS_dict] = pickle.load(f)
    
    """
        emebedding this sentences
    """
    train_instance_patched = []
    for i in tqdm(range(len(Streng_Drug_allsample_feature))):
        patch_embedding = [] # this is one sentence 
        for word in Streng_Drug_allsample_feature[i]:
            patch_embedding_list = []
            try:
                vector = model.wv[word[0].lower()].reshape((1,100))
            except:
                vector = np.random.rand(100,1).reshape((1,100))
            patch_embedding_list.extend(vector.tolist()[0])
            patch_embedding_list.append(word[1])
            patch_embedding_list.append(word[2])
            patch_embedding_list.extend(POS_dict[word[3]].tolist()[0])
            if 'Drug' in word[4]:
                word[4] = 'Drug'
            elif sub_label in word[4]:
                word[4] = sub_label
            elif ":" in word[4]:
                word[4] = word[4].split(':')[0]
            patch_embedding_list.extend(NER_dict[word[4]].tolist()[0])
            patch_embedding.append(patch_embedding_list)
        train_instance_patched.append(patch_embedding)
    """
        store the embedding
    """
    # with open('train_pair/embeding_{}.pickle'.format(label), 'wb') as f:
    #     pickle.dump(train_instance_patched, f, pickle.HIGHEST_PROTOCOL)


    return train_instance_patched



def split_labels(Streng_Drug_allsample_feature, Streng_Drug_allsample_label,label):
    label_instance_dict = {}
    for index in tqdm(range(len(Streng_Drug_allsample_label))):
        if Streng_Drug_allsample_label[index] not in label_instance_dict:
            label_instance = []
            label_instance.append(Streng_Drug_allsample_feature[index])
            label_instance_dict[Streng_Drug_allsample_label[index]] = label_instance
        else:
            label_instance = label_instance_dict[Streng_Drug_allsample_label[index]]
            label_instance.append(Streng_Drug_allsample_feature[index])
            label_instance_dict[Streng_Drug_allsample_label[index]] = label_instance
    return label_instance_dict

def load_train_data(train_datapath):
    global max_length_all
    Streng_Drug_allsample_feature = []
    Streng_Drug_allsample_label = []
    file_list = [f for f in listdir(train_datapath) if isfile(join(train_datapath, f))]
    for index in tqdm(range(len(file_list))):
        file = file_list[int(index)]
        if file.endswith('.ann'):  
            term_filename = train_datapath + file
            text_filename = train_datapath + file.split('.')[0]+'.txt'
            token_map_list,relation_pair = train_text_helper(term_filename,text_filename)
            file_train_feature,file_train_label_sample = generate_train_possiblepair_sample(token_map_list,relation_pair)
            Streng_Drug_allsample_feature.extend(file_train_feature)
            Streng_Drug_allsample_label.extend(file_train_label_sample)
    with open('./train_pair/train_pair.pickle','wb') as f:
        pickle.dump([Streng_Drug_allsample_feature, Streng_Drug_allsample_label], f, pickle.HIGHEST_PROTOCOL)



def train_word2vec():
    separate_data()
    print('start load data....')
    sentense_list = load_data_mimic()

    word_dict = get_word_dict(sentense_list)
    print('num of word in corpus = ' + str(len(word_dict)))

    # train word2vec 
    print('Start Word2Vec training...')
    model = Word2Vec(sentense_list, size=100, window=5, min_count=1, workers=4)
    model.save('word2vec_50000notes_1008') 
    #           


def train_model(train_instance_patched,Streng_Drug_allsample_label):
    
    
    embedding_dim = 156 # 156
    sequence_length = 50 # 
    filter_sizes = [1,2,3,4,5]
    num_filters = 200
    batch_size = 30
    drop = 0.5
    epochs = 5
    num_instance = len(train_instance_patched)
    split_rate = 0.3
    train_num = int(num_instance*0.7)
    input_length = sequence_length * embedding_dim
    
    
    # print(np.shape(train_instance_patched[0]))

    a = np.zeros((num_instance, sequence_length, embedding_dim))
    for i in range(num_instance):
        x = min(50,np.shape(train_instance_patched[i])[0])
        a[i,:x,:] = np.array(train_instance_patched[i])[:x,:]

    b = np.zeros((num_instance,sequence_length * embedding_dim))
    for i in range(np.shape(a)[0]):
        b[i,:] = a[i,:,:].reshape(1,sequence_length*embedding_dim)
    

    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    # print(np.shape(b))
    a = np.array(a)
    b = np.array(b)
    X_train = b[:train_num, :]
    X_test = b[train_num:, :]
    print(np.shape(X_train))
    print(np.shape(X_test))

    c = np.zeros((len(Streng_Drug_allsample_label),2))
    for i in range(len(Streng_Drug_allsample_label)):
        if Streng_Drug_allsample_label[i] == 0:
            c[i,:] = [0,1]
        else:
            c[i,:] = [1,0]
    Y_train = c[:train_num,:]
    Y_test = c[train_num:,:]
    print(np.shape(Y_train))
    print(np.shape(Y_test))

   
    print("Creating Model...")
    inputs = Input(shape=(input_length,), dtype='float32') #  ? *80
    # print(inputs)
    reshape = Reshape((sequence_length,embedding_dim,1))(inputs)
    # print(reshape)
    conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
    conv_4 = Conv2D(num_filters, kernel_size=(filter_sizes[4], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')(conv_2)
    maxpool_3 = MaxPool2D(pool_size=(sequence_length - filter_sizes[3] + 1, 1), strides=(1,1), padding='valid')(conv_3)
    maxpool_4 = MaxPool2D(pool_size=(sequence_length - filter_sizes[4] + 1, 1), strides=(1,1), padding='valid')(conv_4)
    # # print(np.shape(X_train))
    # exit()
    # sequence_length = np.shape(X_train)[1]
    # print(sequence_length)
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    output = Dense(units=2, activation='softmax')(dropout)
    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)

    checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    print("Traning Model...")
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_test, Y_test))  # starts training
    
    