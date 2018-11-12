from n2c2_util import *
import pickle
"""
    this script is to generate the embedding for 9 tasks 
    each of them is a binary classification

"""
def pre_train(goal):
    train_datapath = './data/n2c2_2018/dataset/track2-training_data_2/'
    """
        For train, generate possible train instance and correspoding label
    """
    possible_entity = ['Strength-Drug', 'Frequency-Drug', 'Reason-Drug', 'Route-Drug', 'Dosage-Drug', 'Form-Drug', 'ADE-Drug', 'Duration-Drug']
    # if there is not sample, generate all possible pathc
    if os.path.exists('train_pair/train_pair.pickle'):
        print('Train pair patch file exists')
    else:
        print("Start preprocessing the train_data for each file:")
        load_train_data(train_datapath)

    # # if there is not sample, generate all possible pathc
    # if os.path.exists('train_pair/test_pair.pickle'):
    #     print('Test pair patch file exists')
    # else:
    #     print("Start preprocessing the test_data for each file:")
    #     load_test_data(test_datapath)
    # if there is not sample, generate all possible pathc
    """ 
        For training ,embeddding label_dict
    """
    # split all difference samples 
    with open('train_pair/train_pair.pickle', 'rb') as f:
        [Streng_Drug_allsample_feature, Streng_Drug_allsample_label] = pickle.load(f)
        embedding_NER_POS(Streng_Drug_allsample_feature)

        print('Split instance based on different lable:')
        label_instance_dict = split_labels(Streng_Drug_allsample_feature, Streng_Drug_allsample_label,possible_entity)
        for label, instance_list in label_instance_dict.items():
            print(label,len(instance_list))

    
    # embedding each label
    embedding_label_dict = {}
    for label, instance_list in label_instance_dict.items():
        if label !='None':
            if label != goal:
                print("Start embedding_{}:".format(label))
                train_instance_patched = embedding_features(instance_list[:3000], label)
                embedding_label_dict[label] = train_instance_patched
            else:
                print("Start embedding_{}:".format(label))
                train_instance_patched = embedding_features(instance_list, label)
                embedding_label_dict[label] = train_instance_patched
        else:
            print("Start embedding_{}:".format(label))
            train_instance_patched = embedding_features(instance_list[:2000], label)
            embedding_label_dict[label] = train_instance_patched

    a = embedding_label_dict[goal]
    embedding_label_dict[goal] = a + a + a 
    return embedding_label_dict

def pre_test(goal):
    possible_entity = ['Strength-Drug', 'Frequency-Drug', 'Reason-Drug', 'Route-Drug', 'Dosage-Drug', 'Form-Drug', 'ADE-Drug', 'Duration-Drug']
    golden_state = './data/n2c2_2018/dataset/gold_standard_test/'

    if os.path.exists('train_pair/test_pair.pickle'):
        print('test lable file exists')
    else:
        print("Start preprocessing the test for each file:")
        load_test_data(golden_state)

    with open('train_pair/test_pair.pickle', 'rb') as f:
        [Streng_Drug_allsample_feature, Streng_Drug_allsample_label] = pickle.load(f)
        print('Split instance based on different lable:')
        label_instance_dict = split_labels(Streng_Drug_allsample_feature, Streng_Drug_allsample_label,possible_entity)
        for label, instance_list in label_instance_dict.items():
            print(label,len(instance_list))

    # embedding each label
    embedding_label_dict = {}
    for label, instance_list in label_instance_dict.items():
        if label !='None':
            if label != goal:
                print("Start embedding_{}:".format(label))
                train_instance_patched = embedding_features(instance_list[:1000], label)
                embedding_label_dict[label] = train_instance_patched
            else:
                print("Start embedding_{}:".format(label))
                train_instance_patched = embedding_features(instance_list, label)
                embedding_label_dict[label] = train_instance_patched
        else:
            print("Start embedding_{}:".format(label))
            train_instance_patched = embedding_features(instance_list[:1000], label)
            embedding_label_dict[label] = train_instance_patched
    a = embedding_label_dict[goal]
    embedding_label_dict[goal] = a + a 

    return embedding_label_dict
