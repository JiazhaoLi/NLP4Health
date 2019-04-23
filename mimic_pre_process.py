import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import isfile, join
import nltk
import string
import re
import os
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import pickle as pkl
## some utils setting
data_path = '/Users/jiazhaoli/MIMIC/NOTEEVENTS.csv'
split_path = '/Users/jiazhaoli/MIMIC/Note_split/'
ann_path = '/Users/jiazhaoli/MIMIC/Note_split_ann/'
num_notes = 60
SPLIT_CSV = False  # you need do separating first, only once ,set it False later


class Preprocessing():
    def __init__(self,num_notes):
        self._num_notes = num_notes
        self.all_sentence = []
    def separateNotes(self,SPLIT_CSV): 
        if SPLIT_CSV:
            # seperating csv files into json file
            print("start load data ...")
            df = pd.read_csv(data_path,low_memory=False)
            # the total data is 208,318 rows 
            print('start separate data ...')
            for index in tqdm(range(self._num_notes)):
                with open(split_path + str(index+1)+ '.txt' , 'w') as f:
                    note_txt = df.iloc[index+1]['TEXT']
                    f.write(note_txt)
        
    def splitSection(self, note, file_name):
        self.num_dot = re.compile('\d.  ')
        self.section_split = re.compile('[A-Za-z]*:')
        self.subsection_split = re.compile('^[0-9].')
        self._file_name = file_name
        """
            step1 is to split section into a dict format:   {range:sectext}
        """
        section_text_addup  = ''
        section_text_list = []
        section_nnn_list = note.split('\n\n\n')  # seperation by note itself
        for section_nnn in section_nnn_list:
            section_nn_list = section_nnn.split('\n\n')    # seperation by possible section split
            for section_nn in section_nn_list :
                line = section_nn.split('\n') # for each original seperation
                # for those who has ':' is section seperation.
                if  len(self.section_split.findall(line[0])) != 0 and len(self.subsection_split.findall(line[0]))==0 : # 有：，没数字标识
                    # p#rint('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    section_text_list.append(section_text_addup)
                    section_text_addup = ''
                    for re_line in line:
                        section_text_addup += re_line
                        section_text_addup += '\n'
                else:
                    for re_line in line:
                        section_text_addup += re_line
                        section_text_addup += '\n'
                section_text_addup += '\n'
                # print('_____________________________________________________')  # this is possible section seperation
            #print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++') 
            section_text_addup += '\n'
            section_text_list.append(section_text_addup)
            section_text_addup = ''
        self.range_sectiontext_map = dict()
        for sec in section_text_list:
            if len(sec) <= 4:
                pass
            else:
                if note.find(sec) != -1:
                    self.range_sectiontext_map[str(note.find(sec))+','+str(note.find(sec)+ len(sec))] = sec
        # DONE for range_sectiontext map
        """
            step2 split each section if there are some index sign in each section. merge some sentence splitted by '\n'
                1. split paragraphs
                2. for those who has index, split based on index
                                    doesn't , split based on period.
        """
        self.range_paragraphs_map = dict()
        for range,section_text in self.range_sectiontext_map.items():   
            start = int(range.split(',')[0])
            parag_list = section_text.split('\n\n')
            for parag in parag_list:
                if len(parag)<= 4:
                    pass
                else:
                    self.range_paragraphs_map[ str( start + section_text.find( parag)) + ',' + str( start + section_text.find( parag) + len(parag))] = parag
        """ 
            for each line, if it begin with index: then then don;t addup
        """
        self.range_sentence_map = dict()
        self.subsection_split = re.compile('^[0-9]+\.+' '|^#|^\?{2,3}|^[-+]')
        for par_range, par_txt in self.range_paragraphs_map.items():
            par_range_start = int(par_range.split(',')[0])
            line_list = par_txt.split('\n')
            content = ''
            for line in line_list:
                index = self.subsection_split.findall(line)
                if index == []:  # do addup
                    content += line
                    content += '\n'
                else:
                    if len(content) != 0:
                        self.range_sentence_map[str(max(par_range_start + par_txt.find(content),0))+ ','+  str(par_range_start + par_txt.find(content)+ len(content))] = content
                    content = ''
                    content += line
                    content += '\n'
            if content != '':
                if len(content) != 0:
                    self.range_sentence_map[str(max(par_range_start + par_txt.find(content),0))+ ','+ str(par_range_start + par_txt.find(content)+ len(content))] = content

            for k,v in self.range_sentence_map.items():
                self.range_sentence_map[k] = v.replace('\n',' ')
        # DONE for sentence 
        # Next step is to do sentence split for period
        self.range_singlesent_map = dict()
        for range_sent,senteces_text in self.range_sentence_map.items():
            sent_list = nltk.sent_tokenize(senteces_text)
            start = int(range_sent.split(',')[0])
            for sen in sent_list:
                if len(sen) >= 3:
                    self.range_singlesent_map[ str(start + senteces_text.find(sen))+ ','+ str(start + senteces_text.find(sen)+len(sen))] = sen
        self.range_token_map = dict()
        for range_sen, sent_txt in self.range_singlesent_map.items():
            start = int(range_sen.split(',')[0])
            token_list = nltk.word_tokenize(sent_txt)
            for token in token_list:
                self.range_token_map[ str(start + sent_txt.find(token))+ ','+ str(start + sent_txt.find(token)+len(token))] = token
        self.all_sentence.append(self.range_singlesent_map)
        

    def preProcessing(self):
        # self.subsection_split_2 = re.compile('^')
        ##  for each files, do processing. 
        sentence_split_file = './sentence.pkl'
        exists = os.path.isfile(sentence_split_file)
        if exists == 0:
            # 1 step is to split paragraphs : followed by '\n\n\n' and '\n\n'
            file_list = [f for f in listdir(split_path) if isfile(join(split_path, f))]
            for i in tqdm(range(len(file_list))):
                ann_file_name = (ann_path + file_list[i]).replace('.txt','.ann')
                exists = os.path.isfile(ann_file_name)
                if exists == 0:
                    with open(split_path + file_list[i],errors='ignore') as f:
                        text = f.read()
                        self.splitSection(text,file_list[i]) # will get all map range:text
                        with open (ann_file_name, 'w') as f:
                            for k,v in self.range_token_map.items():
                                f.write(k + '\t' + v + '\n')
            with open('./sentence.pkl', 'wb') as f:
                pkl.dump(self.all_sentence, f)
        
    def word_embedding(self):
        train_model_name = './MIMICIII_60000notes'
        exist = os.path.isfile(train_model_name)
        if exist == 0:
            sentence_file_name = 'sentence_list.pkl'
            exist = os.path.isfile(train_model_name)
            if exist == 0:
                with open('./sentence.pkl', 'rb') as f:
                    self.all_sentence = pkl.load(f)
                sentense_list = []
                print(len(self.all_sentence))
                for sen in self.all_sentence:
                    for k,v in sen.items():
                        v = v.strip()
                        token_list = nltk.word_tokenize(v)
                        sentense_list.append(token_list) 
                with open('sentence_list.pkl','wb') as f:
                    sentense_list = pkl.dump(f)
            else:
                with open('sentence_list.pkl','rb') as f:
                    sentense_list = pkl.load(f)
            # train word2vec 
            print('Start Word2Vec training...')
            model = Word2Vec(sentense_list, size=100, window=5, min_count=1, workers=4)
            model.save('./MIMICIII_60000notes')
        else:
            model = Word2Vec.load('./MIMICIII_60000notes')
            w1= 'man'
            print(model.wv.most_similar(w1))
    # def clean(self,s):
    #     for w in s:
    #         if w ==None:
    #             s.remove(w)
    #     punctuation = string.punctuation
    #     s =  [w.strip(punctuation) for w in s]
    #     for w in s:
    #         if '' in s:
    #             s.remove('')
    #     return s

    # def lower(self,s):
    #     return [w.lower() for w in s]
 
    # def get_word_dict(self):
    #     for sentence in sentences:
    #         for word in sentence:
    #             if word not in word_dict:
    #                 word_dict[word] = 1
    #             else:
    #                 word_dict[word] += 1
    #     return word_dict

    
             
 
        
    












                
        
    

    
