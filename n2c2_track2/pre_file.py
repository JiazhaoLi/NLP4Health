from n2c2_util import *
import pickle
"""
    this script is to generate the embedding for 9 tasks 
    each of them is a binary classification

"""
def pre_train():
    train_datapath = './data/n2c2_2018/dataset/track2-training_data_2/'
    test_datapath = './data/n2c2_2018/dataset/test_data_Task2/'

    possible_entity = ['Strength-Drug', 'Frequency-Drug', 'Reason-Drug', 'Route-Drug', 'Dosage-Drug', 'Form-Drug', 'ADE-Drug', 'Duration-Drug']

# if there is not sample, generate all possible pathc
    if os.path.exists('train_pair/train_pair.pickle'):
        print('Train pair patch file exists')
    else:
        print("Start preprocessing the train_data for each file:")
        load_train_data(train_datapath)


    # split all difference samples 
    with open('train_pair/train_pair.pickle', 'rb') as f:
        [Streng_Drug_allsample_feature, Streng_Drug_allsample_label] = pickle.load(f)
        embedding_NER_POS(Streng_Drug_allsample_feature)

        print('Split instance based on different lable:')
        label_instance_dict = split_labels(Streng_Drug_allsample_feature, Streng_Drug_allsample_label,possible_entity)
        for label, instance_list in label_instance_dict.items():
            print(label,len(instance_list))

    # get the 

    # embedding each label
    embedding_label_dict = {}
    for label, instance_list in label_instance_dict.items():
        # if os.path.exists('train_pair/embeding_{}.pickle'.format(label)):
        #     with open('train_pair/embeding_{}.pickle'.format(label), 'rb') as f:
        #         train_instance_patched  =  pickle.load(f)
        # embedding_label_dict[label] = train_instance_patched
        # else:
        if label !='None':
            print("Start embedding_{}:".format(label))
            train_instance_patched = embedding_features(instance_list, label)
            embedding_label_dict[label] = train_instance_patched
        else:
            print("Start embedding_{}:".format(label))
            train_instance_patched = embedding_features(instance_list[:5000], label)
            embedding_label_dict[label] = train_instance_patched


    return embedding_label_dict
    # prepare training and test for different label 

    # for label in possible_entity:
    #     if os.path.exists('train_pair/neg_pos_embed_{}.pickle'.format(label)):
    #         pass
    #     else:
    #         print('Prepaire_{}_pos_neg_data'.format(label))
    #         positive_feature = embedding_label_dict[label]
    #         positive_lable = np.ones(len(positive_feature))
    #         negative_feature = []
    #         for neg_label in possible_entity:
    #             if neg_label != label:
    #                 negative_feature.extend(embedding_label_dict[neg_label])
    #                 negative_label = np.zeros(len(negative_feature))
    #         train_feature = positive_feature+negative_feature
    #         train_label = np.zeros((len(positive_lable.reshape(-1,1))+len(negative_label.reshape(-1,1))))
    #         train_label[:len(positive_lable.reshape(-1,1))] = positive_lable

    #         with open('train_pair/neg_pos_embed_{}.pickle'.format(label), 'wb') as f:
    #             pickle.dump([train_feature,train_label], f, pickle.HIGHEST_PROTOCOL)
        



        

