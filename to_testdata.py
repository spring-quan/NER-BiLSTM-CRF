import os

from config import cfg

class TestCorpus(object):
    def __init__(self):
        self.entity_vocab = self.get_entity_vocab()
        self.test_words = self.get_test_words()

    def get_entity_vocab(self):
        # read all entities from file
        entity_vocab = []
        with open(cfg.entity_path,'r') as f:
            for line in f.readlines():
                entity_vocab.extend(line.strip().split(','))
        return entity_vocab

    def get_test_words(self):
        # sentence to chars list, if entity in vocab, entity is a word
        test_words = []
        with open(cfg.test_path,'r') as f:
            sent = f.readline().strip()
            while sent:
                ent_locs = self.find_entity(sent)
                sent_list = list(sent)
                if ent_locs != []:
                    for ent_loc in ent_locs:
                        ent,s_idx,e_idx = ent_loc
                        sent_list[s_idx] = ent 
                        sent_list[s_idx + 1: e_idx + 1] = [' '] * (len(ent) - 1)
                words = [char for char in sent_list if char != ' ']
                test_words.append(words)
                # break
                sent = f.readline().strip()
        return test_words

    def find_entity(self,sent):
        # if entity in vocab, save entity and start index, end index. [ent,s_idx,e_idx]
        all_ent_loc = []
        location = []
        while True:
            find_num = 0
            for ent in self.entity_vocab:
                try:
                    if all_ent_loc == []:
                        location.append(sent.index(ent))
                    else:
                        location.append(end_idx + 1 + sent[end_idx + 1:].index(ent))
                except: # if entity not in sent, find_num + 1
                    location.append(len(sent))
                    find_num += 1
            if find_num == len(self.entity_vocab):
                break
            else: # have entity
                minloc,entity = self.minloc_entityfind(location)
                end_idx = minloc + len(entity) - 1
                all_ent_loc.append([entity,minloc,end_idx])
                location = []
        return all_ent_loc

    def minloc_entityfind(self,location):
        # location: [vocab_size]
        # find minloc and max length entity 
        idx = -1
        entity_idx = []
        minloc = min(location)
        # find all minloc entity index
        for i in location:
            idx += 1 
            if  i == minloc:
                entity_idx.append(idx)
        # find max length entity in all minloc entities
        maxlen_ent = entity_idx[0]
        if len(entity_idx) > 1:
            for j in entity_idx:
                if len(self.entity_vocab[j]) > len(self.entity_vocab[maxlen_ent]):
                    maxlen_ent = j 
        return minloc,self.entity_vocab[maxlen_ent]