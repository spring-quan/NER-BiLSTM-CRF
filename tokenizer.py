import jieba
from jieba import posseg

def segment(sentence,cut_type = 'word',pos = False):
    # segment sentence to words or chars
    if not pos:
        if cut_type == 'word':
            return jieba.lcut(sentence)
        elif cut_type == 'char':
            return list(sentence)
    else:
        if cut_type == "word":
            word_pos_seq = posseg.lcut(sentence)
            word_seq,pos_seq = [],[]
            for w,pos in word_pos_seq:
                word_seq.append(w)
                pos_seq.append(pos)
            return word_seq,pos_seq
        elif cut_type == 'char':
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p)
            return word_seq,pos_seq