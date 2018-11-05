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



embedding_label_dict = pre_train()


possible_entity = ['Strength-Drug', 'Frequency-Drug', 'Reason-Drug', 'Route-Drug', 'Dosage-Drug', 'Form-Drug', 'ADE-Drug', 'Duration-Drug']

# for label in possible_entity:
label = 'ADE-Drug'
positive_feature = embedding_label_dict[label]
positive_lable = np.ones(len(positive_feature))
negative_feature = []
for neg_label in possible_entity:
    if neg_label != label:
        negative_feature.extend(embedding_label_dict[neg_label])
        negative_label = np.zeros(len(negative_feature))
train_feature = positive_feature+negative_feature
train_label = np.zeros((len(positive_lable.reshape(-1,1))+len(negative_label.reshape(-1,1))))
train_label[:len(positive_lable.reshape(-1,1))] = positive_lable
train_model(train_feature,train_label)

