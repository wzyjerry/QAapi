# -*- coding:utf-8 -*-
import os
import time
import jieba
import torch
import pickle
import codecs
import tempfile
from util.classify_dataloader import TextClassDataLoader
from util.classify_vocab import GloveVocabBuilder


class ClassifyLoader(object):
    def __init__(self,
                 classify_stat='data/stat/classify_stat.pkl',
                 pre_emb='data/cleaned_zh_vec'):
        self.stat = classify_stat
        self.pre_emb = pre_emb
        if not os.path.isfile(classify_stat):
            self.save_stat()

    def save_stat(self):
        print('First start...\nSaving stat...')
        st = time.time()
        v_builder = GloveVocabBuilder(path_glove=self.pre_emb)
        d_word_index, embed = v_builder.get_word_index()
        with codecs.open(self.stat, 'wb') as fout:
            pickle.dump(d_word_index, fout)
        ed = time.time()
        print('Stat saved. Save time:', ed - st)

    def load_stat(self, model_path):
        print('Loading model...')
        st = time.time()
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.use_gpu = 0
        with codecs.open(self.stat, 'rb') as fin:
            d_word_index = pickle.load(fin)
        ed = time.time()
        print('Model loaded. Load time:', ed - st)
        return model, d_word_index


def classify(txt, model, d_word_index):
    txt = txt.strip()
    word = []
    for item in txt.split():
        word.extend(jieba.cut(item))
    tmp_file = tempfile.mkstemp(text=True)
    with codecs.open(tmp_file[1], 'w', 'utf-8') as fout:
        fout.write('label\tbody\n')
        fout.write('0\t%s\n' % (' '.join(word)))

    train_loader = TextClassDataLoader(tmp_file[1], d_word_index, batch_size=1)
    for i, (seq, target, seq_lengths) in enumerate(train_loader):
        output = model(seq, seq_lengths)
        arr = output[0].data.numpy().tolist()
    return arr.index(max(arr))
