# given raw data file:       sentence,ent1,ent2 ...
# save to json format file:  sentence,[[ent1,start_index,end_index],[ent2,start_index,end_index]] 

import os,json

def minloc_entityfind(location,entity):
    # given entities and begin index: [ent1,ent2, ...]; [idx1,idx2, ...]
    # return: min_loc, and max_length entity
    minloc = location[0] # the begin index of entity
    n = -1
    num = []
    # get min location
    for i in location:
        if i < minloc:
            minloc = i 
    # get all entity with min loc
    for j in location:
        n += 1
        if j == minloc:
            num.append(n)
    # choose max len entity in all entities with min loc
    maxlen_ent = num[0]
    if len(num) > 1:
        for k in num:
            if len(entity[k]) > len(entity[maxlen_ent]):
                maxlen_ent = k
    return minloc,entity[maxlen_ent]


def ent_loc(sent_ents):
    # given sentence and entities: sentence,[ent1,ent2,...]
    # return all entity's begin index and end index: sentence,[[ent1,start_index,end_index], ...]
    data = {}
    data['sent'],data['ent_loc'] = [],[]
    for i in range(len(sent_ents)):
        sent,ents = sent_ents[i]
        location = []
        all_ent_loc = []
        if sent[0] == "":
            continue
        while True:
            find_num = 0 
            for ent in ents:
                try:
                    if all_ent_loc == []:
                        location.append(sent[0].index(ent))     
                    else:
                        location.append((end_idx + 1 + sent[0][end_idx + 1:].index(ent)))   
                except:
                    location.append(len(sent[0]))
                    find_num += 1
            # traversing all entity 
            if find_num == len(ents):
                break
            else:
                begin_idx,loc_entity = minloc_entityfind(location,ents)
                location = []
                end_idx = begin_idx + len(loc_entity) - 1
                all_ent_loc.append([loc_entity,begin_idx,end_idx])
        # print(all_ent_loc)
        data['sent'].append(sent[0])
        data['ent_loc'].append(all_ent_loc)
        assert len(data['sent']) == len(data['ent_loc'])
    with open('./data/sent_ent_loc.json','w') as f:
        json.dump(data,f)
    print('get sentences with all entity and indexes, size: ',len(data['sent']))

# read raw data from file
with open('./data/raw_data.csv','r') as f:
    data = f.readlines()
# get [sent,entities]
sent_ents = []
for i in range(1,len(data)):
    items = data[i].strip().split(',')
    if items != []:
        assert len(items) > 2
        sent_ents.append([[items[1]],items[2:]])
# print(len(data))
# print(len(sent_ents))
# print(sent_ents[0])

ent_loc(sent_ents)