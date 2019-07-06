# NER-BiLSTM-CRF
implement NER using Bi-LSTM+CRF model, built on tensorflow

## code info:
- train
    - to_ent_loc.py: process raw data before train. input "./data/raw_data.txt", get "./data/sent_ent_loc.json"
    - to_traindata.py: process data to get train data. input "./data/sent_ent_loc.json", get "./data/train.json"
- test/infer:
    - to_testdat.py: process raw data before test.

## train
set ``is_training`` to True in ``config.py``
```
python3 train.py
```

## test/infer
set ``is_testing`` to True in 'config.py'
```
python3 infer.py
```