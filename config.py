import os

class Config(object):
    is_training = True # set True to training, False to infer/test
    is_testing = True
    data_path = './data' # the path of data and vocab
    entity_path = './data/entity_vocab.txt' # the path of entity vocab
    test_path = './data/test.txt'
    vocab_path = './data/vocab.txt'
    saved_path = './log_model'
    model_path = './log_model/2019-07-06 15-58-17-model/valid_loss_0.37'
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)
    batch_size = 8
    max_epoch = 50
    num_print_steps = 1
    ckpt_steps = 32
    embed_size = 128
    hidden_size = 256
    num_layers = 2
    lr = 0.0001

cfg = Config()