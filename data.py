import os,json
import pandas as pd
import numpy as np
from itertools import chain 

from config import cfg
from to_testdata import TestCorpus

UNK = 'UNK'
PAD = 'PAD'
class DataLoader(object):
    def __init__(self,mode = 'train'):
        self.x_train = []
        self.y_train = []
        self.x_valid = []
        self.y_valid = []
        self.x_test = []
        self.y_test = []
        self.mode = mode
        self.tag_dict = ['O','B-ENT','I-ENT','E-ENT','S-ENT']
        self.label_size = len(self.tag_dict)

        def x_index(words):
            # convert words to indexs
            indexs = []
            for w in words:
                if w in self.set_words:
                    indexs.append(self.word2id[w])
                else:
                    indexs.append(self.word2id[UNK])
            return indexs
        def y_index(tags):
            indexs = list(self.tag2id[tags])
            return indexs

        # if cfg.is_training:
        if self.mode == 'train':
        # read data from json format file
            with open(os.path.join(cfg.data_path,'train.json'),'r') as f:
                data = json.load(f)
            words = data['words']
            tags = data['tags']
        # build vocab
            data = pd.DataFrame({'words':words,'tags':tags},index = range(len(words)))
            self.all_words = list(chain(*data['words'].values)) # chain() return all items in multiple iterable objects
            sr_allwords = pd.Series(self.all_words).value_counts() 
            # pandas object, include index and values; .value_counts() return all unique items and their counts
            set_words = [PAD,UNK] + list(sr_allwords.index) # all unique words
            self.set_words = [w for w in set_words if w != ' ']
            self.vocab_size = len(self.set_words)
        # word vocab dict and tag vocab dict
            self.word2id = pd.Series(range(len(self.set_words)),index = self.set_words)
            self.id2word = pd.Series(self.set_words, index = range(len(self.set_words)))
        # save word vocab to file
            self.id2word.to_csv(cfg.vocab_path,sep = '\t',encoding = 'utf-8',index = False)
            self.tag2id = pd.Series(range(len(self.tag_dict)), index = self.tag_dict)
            self.id2tag = pd.Series(self.tag_dict, index = range(len(self.tag_dict)))
            print('word vocab size: {}, tag vocab size: {}'.format(len(self.word2id),len(self.tag2id)))
        # convert words to indexs ,convert to ndarray
            x_train = data['words'].apply(x_index)
            y_train = data['tags'].apply(y_index)
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
        # split to train dataset and valid dataset
            idx = int(len(x_train) / 10)
            self.x_train = x_train[:-idx]
            self.y_train = y_train[:-idx]
            self.x_train,self.y_train = self.random_sort(x_train,y_train)
            self.x_valid = x_train[-idx:]
            self.y_valid = y_train[-idx:]
            # print(self.x_valid)
        elif self.mode == 'infer':
            testcorpus = TestCorpus()
        # load vocab
            word_vocab = pd.read_csv(cfg.vocab_path,sep = '\t',header = None)
            self.set_words = word_vocab[0].values
            self.vocab_size = len(self.set_words)
            self.word2id = pd.Series(range(len(self.set_words)),index = self.set_words)
            self.id2word = pd.Series(self.set_words,index = range(len(self.set_words)))
            self.tag2id = pd.Series(range(len(self.tag_dict)),index = self.tag_dict)
            self.id2tag = pd.Series(self.tag_dict,index = range(len(self.tag_dict)))
            print('word vocab size: {}, tag vocab size: {}'.format(len(self.word2id),len(self.tag2id)))
        # load test words
            test_data = pd.DataFrame({'words':testcorpus.test_words},index = range(len(testcorpus.test_words)))
        # convert words to index
            x_test = test_data['words'].apply(x_index)
            self.x_test = np.asarray(x_test)
            # print(self.x_test)

    def random_sort(self,inputs,labels):
        indexs = list(range(len(inputs)))
        np.random.shuffle(indexs)
        inputs_shf,labels_shf = [],[]
        for i in indexs:
            inputs_shf.append(inputs[i])
            labels_shf.append(labels[i])
        return inputs_shf,labels_shf 

    @property
    def train(self):
        return self.x_train,self.y_train

    @property
    def valid(self):
        return self.x_valid,self.y_valid

    @property
    def test(self):
        return self.x_test,self.y_test