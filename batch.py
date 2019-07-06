import numpy as np 

from config import cfg

PAD_ID = 0

class BatchGenerator(object):
    def __init__(self,x,y,name):
        if type(x) != np.ndarray:
            x = np.asarray(x)
        if type(y) != np.ndarray:
            y = np.asarray(y)
        self.words = x
        self.labels = y
        self.name = name
        self.data_size = len(x)
        self.indexs = list(range(self.data_size))

    def epoch_init(self,shuffle = True):
        self.ptr = 0 # pointer to a batch
        batch_size = cfg.batch_size
        self.num_batch = self.data_size // batch_size
        print('\tshuffle {} dataset to init epoch'.format(self.name))
        print('\t{} dataset split to {} batches with {} samples left over'.format(self.name,
                                                self.num_batch,self.data_size % cfg.batch_size))
        if shuffle:
            np.random.shuffle(self.indexs)
        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexs[i * batch_size: (i+1) * batch_size])

    def _pad_sent(self,sent,maxlen):
        return sent + [PAD_ID] * (maxlen - len(sent))

    def _prepare_batch(self,selected_ids):
        selected_ids.sort(key = lambda x: len(self.words[x]),reverse = True)
        words = [self.words[idx] for idx in selected_ids]
        if self.name != 'test':
            labels = [self.labels[idx] for idx in selected_ids]
        else:
            labels = []
        max_length = max(map(len,words))
        lengths = np.array(list(map(len,words)))
        pad_words = np.array([self._pad_sent(sent,max_length) for sent in words])
        pad_labels = np.array([self._pad_sent(sent,max_length) for sent in labels])
        # assert pad_words.shape == pad_labels.shape
        return [pad_words,lengths,pad_labels]

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_ids)
        else:
            return None