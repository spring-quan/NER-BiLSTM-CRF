# read data from json file: sentence,[[ent1,start_index,end_index],[ent2,start_index,end_index], ...]
# save data to json file；  words/chars list, tags list
import os,json
from tokenizer import segment


def sent2tag(sent,ent_locs,entity_vocab):
    # segment sentence to words
    words = segment(sent,cut_type = 'char')
    tags = ['O'] * len(words)

    # given entity and locs, get tags
    for ent_loc in ent_locs:
        ent,s_idx,e_idx = ent_loc
        diff = e_idx - s_idx
        if ent not in entity_vocab: # if entity length is 1 or entity in entity vocab, its tag is 'S-ENT'
            if diff == 0:
                tags[s_idx] = 'S-ENT'
            elif diff == 1:
                tags[s_idx] = 'B-ENT'
                tags[e_idx] = 'E-ENT'
            elif diff >= 2:
                tags[s_idx] = 'B-ENT'
                tags[e_idx] = 'E-ENT'
                tags[s_idx + 1: e_idx] = ['I-ENT'] * (diff - 1)
        elif ent in entity_vocab:
            if diff == 0:
                tags[s_idx] = 'S-ENT'
            elif diff == 1:
                tags[s_idx] = 'S-ENT'
                tags[e_idx] = '*'
            elif diff >= 2:
                tags[s_idx] = 'S-ENT'
                tags[s_idx + 1:e_idx +1] = ['*'] * diff
    # given tags to process words and tags
    content,labels = [],[]
    t_idx = 0
    ent_in_dict = 0
    for tag in tags:
        if tag != "*":
            labels.append(tag)
            if ent_in_dict == 0:
                content.append(words[t_idx])
            elif ent_in_dict == 1:
                content[-1] = "".join(words[begin : t_idx]) # not include words[t_idx]
                content.append(words[t_idx]) 
                ent_in_dict = 0  # unlock to tackle next entity in dict 
        elif tag == '*' and ent_in_dict == 0:
            begin = t_idx - 1
            ent_in_dict = 1 # record the begin index of entity in dict, and lock off
        t_idx += 1 
    # print(content)
    # print(labels)
    assert len(content) == len(labels)
    return content,labels
# read sentences and ent_locs from json format file 
with open('./data/sent_ent_loc.json','r') as f:
    data = json.load(f)
    sents = data['sent']
    ent_locs = data['ent_loc']

# read entity vocab from file
entity_vocab = []
with open('./data/entity_vocab.txt','r') as f:
    for line in f.readlines():
        entity_vocab.extend(line.strip().split(','))

# get content and labels, and save to json format file
data = {}
data['sent'],data['ent_loc'],data['words'],data['tags'] = [],[],[],[]

for sent,ent_loc in zip(sents,ent_locs):
    content,labels = sent2tag(sent,ent_loc,entity_vocab)
    data['sent'].append(sent)
    data['ent_loc'].append(ent_loc)
    data['words'].append(content)
    data['tags'].append(labels)

# save data to json format file
with open('./data/train.json','w') as f:
    json.dump(data,f)
print('save train data as train.json, size: ',len(data['sent']))

print('example data:')
for key in data:
    print(data[key][0],'\n')
