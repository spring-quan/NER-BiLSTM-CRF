import tensorflow as tf 
import numpy as np 

from config import cfg
from data import DataLoader
from batch import BatchGenerator
from model import Model
from to_testdata import TestCorpus

def extract_entity(words,labels):
    # input: words and labels list
    # return: [entity,label,idx] or [entity,labels,start_index,end_index]
    words,labels = list(words),list(labels)
    result,index_correct,temp = [],[],[]
    # extracht regular entity
    i = 0
    while i < len(words):
        if labels[i][0] == 'S':
            result.append([words[i],labels[i],i])
            index_correct.append(i)
            i += 1
        elif labels[i][0] == 'B' and (i + 1) < len(labels):
            k = i + 1
            temp = [labels[i][0],labels[k][0]]
            while labels[k][0] == 'I':
                k += 1
                if k < len(labels):
                    temp += labels[k][0]
                else:
                    break
            if temp == ['B'] + ['I'] * (len(temp) - 2) + ['E']:
                result.append([words[i: k + 1],labels[i: k+1],i,k])
                index_correct.extend(range(i,k+1))
            temp = []
            if k >= len(labels):
                break
            if labels[k][0] in ['S','B']:
                i = k
            else:
                i = k + 1
        else:
            i += 1
    # print(result)
    return result
def to_entity(ent_locs):
    all_entity = []
    for ent_loc in ent_locs:
        if len(ent_loc[0]) > 1:
            all_entity.append(''.join(ent_loc[0]))
        else:
            all_entity.append(ent_loc[0])
    return all_entity


corpus = DataLoader('train')
x_valid,y_valid = corpus.valid
corpus = DataLoader('infer')
x_test,y_test = corpus.test
valid_feed = BatchGenerator(x_valid,y_valid,'valid')

tf.reset_default_graph() 
model = Model(corpus)
saver = tf.train.Saver()
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init_op)  # initialize all parameters
# load model
    checkpoint = tf.train.latest_checkpoint(cfg.model_path)
    saver.restore(sess,checkpoint)
    if cfg.is_testing: # labeling
        f_out = open('./data/test_entity.txt','w')
        for words in x_test:
            test_words = np.asarray([words] * cfg.batch_size) # to copy data to construct batch fiting model
        # viterbi decode
            logits,transition_params = model.forward(sess,test_words,mode = 'infer')
            # logits: [batch,length,label_size]
            # transition_params: [label_size,label_size]
            for scores in logits:
                viterbi_sequence,_ = tf.contrib.crf.viterbi_decode(scores,transition_params)
                break
        # convert idx to words and labels list
            words = corpus.id2word[words].values
            tags = corpus.id2tag[viterbi_sequence].values
        # get pred entity
            ent_locs = extract_entity(words,tags)
            pred_entity = to_entity(ent_locs)
        # get entity using pre-defined entity vocab
            dic_entity = [w for w in words if len(w) > 1]
            entity = list(set(pred_entity + dic_entity))
            f_out.write(''.join(words) + '\t' + ','.join(entity) + '\n')
            break
        f_out.flush()
        f_out.close()
        print('labeling over !!!')
    else: # valid precision and recall
        print('begin valid')
        valid_feed.epoch_init()
        batch = valid_feed.next_batch()
        len_correct,len_error,len_label = 0,0,0
        while batch is not None:
            [valid_words,lengths,valid_labels] = batch
            _,logits,transition_params = model.forward(sess,valid_words,lengths,valid_labels,'valid')
        # viterbi decode
            pred_result = []
            for scores in logits:
                viterbi_sequence,_ = tf.contrib.crf.viterbi_decode(scores,transition_params)
                pred_result.append(viterbi_sequence)
        # convert index to words/tags
            pred_tags = [list(corpus.id2tag[result].values) for result in pred_result]
            words = [list(corpus.id2word[words].values) for words in valid_words]
            labels = [list(corpus.id2tag[labels].values) for labels in valid_labels]
        # get true entity and pred entity
            for i in range(len(words)):
                true_result = extract_entity(words[i],labels[i])
                pred_result = extract_entity(words[i],pred_tags[i])
                true_entity = to_entity(true_result)
                pred_entity = to_entity(pred_result)
        # caculate precision and recall
                result_correct = [temp for temp in pred_entity if temp in true_entity]
                result_error = [temp for temp in pred_entity if temp not in true_entity]
                if len(result_correct) != 0:
                    len_correct += len(result_correct)
                    len_error += len(result_error)
                    len_label += len(true_entity)
            batch = valid_feed.next_batch()
            print("len_correct: {}, len_error: {}, len_label: {}".format(len_correct,len_error,len_label))
        prec_avg = len_correct / (len_correct + len_error)
        recall_avg = len_correct / len_label
        f_avg = prec_avg * recall_avg * 2 / (prec_avg + recall_avg)
        print('\t# valid data # precision: {:.5f}, recall: {:.5f}, F1: {:.5f}'.format(prec_avg,recall_avg,f_avg))
