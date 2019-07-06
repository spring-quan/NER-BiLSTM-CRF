import numpy as np
import os
from datetime import datetime
import tensorflow as tf

from config import cfg
from data import DataLoader
from batch import BatchGenerator
from model import Model


corpus = DataLoader()
x_train,y_train = corpus.train
x_valid,y_valid = corpus.valid
train_feed = BatchGenerator(x_train,y_train,'train')
valid_feed = BatchGenerator(x_valid,y_valid,'valid')

model = Model(corpus)

now_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
saved_path = os.path.join(cfg.saved_path,now_time + '-model')
if not os.path.exists(saved_path):
    os.mkdir(saved_path)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # initialize all parameters
    epoch = 0
    global_step = 0
    best_valid_loss = np.inf
    train_losses,valid_losses = [],[]
    all_avg_valid_loss,all_avg_train_loss,all_train_loss = [],[],[]
    saver = tf.train.Saver()
    while epoch < cfg.max_epoch:
        print('#'*10,"EPOCH: {}".format(epoch),'#'*10)
        train_feed.epoch_init()
        batch = train_feed.next_batch()
    # train
        while batch is not None:
            global_step += 1
            [train_words,word_lengths,train_labels] = batch
            [_,loss] = model.forward(sess,train_words,word_lengths,train_labels,'train')
            train_losses.append(loss)
            batch = train_feed.next_batch()
    # print train loss
            if global_step % cfg.num_print_steps == 0:
                print('global_step: {},  trian loss: {:.5f}'.format(global_step,loss))
    # valid
            if global_step % cfg.ckpt_steps == 0:
                print('\t','-'*10,'begin valid','-'*10)
                valid_feed.epoch_init()
                valid_batch = valid_feed.next_batch()
                while valid_batch is not None:
                    [valid_words,word_lengths,valid_labels] = valid_batch
                    [loss,_,_] = model.forward(sess,valid_words,word_lengths,valid_labels,'valid')
                    valid_losses.append(loss)
                    valid_batch = valid_feed.next_batch()
                    # break
                avg_valid_loss = np.mean(valid_losses)
                avg_train_loss = np.mean(train_losses)
            # record all train loss  and valid loss 
                all_train_loss.extend(train_losses)
                all_avg_train_loss.append(str(avg_train_loss))
                all_avg_valid_loss.append(str(avg_valid_loss))
                print('\tavg train loss: {:.5f} ,avg valid loss: {:.5f}'.format(avg_train_loss,avg_valid_loss))
                if avg_valid_loss < best_valid_loss:
                    best_valid_loss = avg_valid_loss
                    print('\tbest_valid_loss update to "{:.5f}",prepare to save model ...'.format(avg_valid_loss))
                    s_path = os.path.join(saved_path,'valid_loss_{:.2f}'.format(avg_valid_loss))
                    if not os.path.exists(s_path):
                        os.mkdir(s_path)
                    saver.save(sess,os.path.join(s_path,'model.ckpt'))
                valid_losses = []
                train_losses = []
            # break
        epoch += 1
    print("train over, loss data saved to {}".format(saved_path))