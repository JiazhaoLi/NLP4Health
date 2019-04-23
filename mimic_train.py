from mimic_pre_process import *
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


#TODO

PrePro = Preprocessing(num_notes)
PrePro.separateNotes(SPLIT_CSV) # split the whole csv into single notes
PrePro.preProcessing()   # for each note: 1. get multi-level mapping 2. generate ann. file for token level
PrePro.word_embedding()
# word_model = WordEmbedding(PrePro)


# sentense_list = load_data()

# word_dict = get_word_dict(sentense_list)
# print('num of word in corpus = ' + str(len(word_dict)))

# # train word2vec 
# print('Start Word2Vec training...')
# model = Word2Vec(sentense_list, size=100, window=5, min_count=1, workers=4)
# model.save('word2vec_50000notes_1008')
# model = Word2Vec.load('word2vec_50000notes_1008')
#


