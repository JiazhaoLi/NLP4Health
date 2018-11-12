from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Dropout, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
import tensorflow as tf
from n2c2_util import * 
import pickle
import os 
from pre_file import *


upsample_ratio = 4
max_length_all = 50
possible_entity = ['Strength-Drug', 'Frequency-Drug', 'Reason-Drug', 'Route-Drug', 'Dosage-Drug', 'Form-Drug', 'ADE-Drug', 'Duration-Drug']

label = 'Strength-Drug'
"""
    pretrain and pretest
"""
train_embedding_label_dict = pre_train(label)



# for label in possible_entity:


train_positive_feature = train_embedding_label_dict[label] 
train_positive_lable = np.ones(len(train_positive_feature))
print("train_len of positive lable:"+str(len(train_positive_lable)))

train_negative_feature = []
for neg_label in possible_entity:
    if neg_label != label:
        train_negative_feature.extend(train_embedding_label_dict[neg_label])
        train_negative_label = np.zeros(len(train_negative_feature))

train_feature = train_positive_feature + train_negative_feature
train_label = np.zeros(np.shape(train_feature)[0])
train_label[:len(train_positive_lable.reshape(-1,1))] = train_positive_lable

print("len of lable:"+str(len(train_label)))



"""
    test
"""
test_embedding_label_dict = pre_test(label)

test_positive_feature = test_embedding_label_dict[label]
test_positive_lable = np.ones(len(test_positive_feature))
print("train_len of positive lable:"+str(len(test_positive_lable)))

test_negative_feature = []
for neg_label in possible_entity:
    if neg_label != label:
        test_negative_feature.extend(test_embedding_label_dict[neg_label])
        test_negative_label = np.zeros(len(test_negative_feature))

test_feature =  test_positive_feature+test_negative_feature
test_label = np.zeros((len(test_positive_lable.reshape(-1,1))+len(test_negative_label.reshape(-1,1))))
test_label[:len(test_positive_lable.reshape(-1,1))] = test_positive_lable

# test_label = np.ones((len(test_positive_lable.reshape(-1,1))))
# test_label[:len(test_positive_lable.reshape(-1,1))] = positive_lable

print("len of lable:"+str(len(test_label)))


train_model(train_feature,train_label,test_feature,test_label,label)


