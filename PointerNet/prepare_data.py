import re
import pymysql
from sqlalchemy import create_engine
from collections import Counter
import pandas as pd
import numpy as np
import pickle

from py2neo import Graph, Node, Relationship
test_graph = Graph('ip', user='xx', password='xx')

def get_nodes(graph, entity="Subject"):
    node_names = graph.run("MATCH (n: %s) RETURN n.name AS nn" % entity).data()
    names = [name['nn'] for name in node_names]
    return names

node_labels = ['Subject', 'Action', 'Party', 'Descriptor', 'ActionDescriptor']
nodes = {label: get_nodes(test_graph, entity=label) for label in node_labels}

def get_connection(host,user,password,database):
    connection = pymysql.connect(
        host = host,
        user = user,
        password = password,
        db = database,
        charset='UTF8MB4')
    return connection


def is_sent_2node(sent):
    s = False
    a = False
    for word in sent.split():
        if word in nodes['Subject']:
            s = True
        if word in nodes['Action'] and word != 'recommended':
            a = True
        if s and a:
            return True
    return False


def get_pairs():
    # read data
    cat_cn = 'mysql+pymysql://'
    cat_engine = create_engine(cat_cn)

    sql_3s = "SELECT id, loc, rewrite_oc, content, test FROM temp_oc_net"
    df_set = pd.read_sql_query(sql_3s, cat_engine)

    # rewrite outcome
    # num = 0
    pairs = []
    train_pairs = []
    test_pairs = []
    for item in df_set.itertuples():
        _, idx, loc, roc, content, test = item
        if (not loc and not roc) or not content:
            continue
        content = re.sub(r"[^a-z.,;']+", r" ", content)
        sent_set = set(content.split())
        if roc:
            oc = " ".join([word for word in roc.split() if word in sent_set])
            if is_sent_2node(oc):
                pairs.append( [content, oc] )
                if test:
                    test_pairs.append([content, oc])
                else:
                    train_pairs.append([content, oc])
        elif loc:
            oc = " ".join([word for word in loc.split() if word in sent_set])
            if is_sent_2node(oc):
                pairs.append( [content, oc] )
                if test:
                    test_pairs.append([content, oc])
                else:
                    train_pairs.append([content, oc])

    assert len(pairs) == len(train_pairs) + len(test_pairs)
    return pairs, train_pairs, test_pairs


def get_train_test(train_pairs, lang, dot_id=10, end_id=930):
    xtrain = []
    ytrain = []
    yids = [] # location of element of ytrain in xtrain
    for srcstr, tgtstr in train_pairs:
        src = [lang.word2idx[word] for word in srcstr.split() if word in lang.word2idx]
        if dot_id not in src:
            src.append(dot_id)
        # if src[-1] != end_id:
        #     src.append(end_id)
        # tgt = [lang.word2idx[word] for word in tgtstr.split() if word in lang.word2idx] + [dot_id, end_id] # 10 = '.'
        tgt = [lang.word2idx[word] for word in tgtstr.split() if word in lang.word2idx] + [dot_id] # 10 = '.'
        tgt_ids = [src.index(elem) for elem in tgt]
        xtrain.append(src)
        ytrain.append(tgt)
        yids.append(tgt_ids)
    return xtrain, ytrain, yids


class Lang():
    def __init__(self, sents, min_count=10):
        self.sents = sents
        self.min_count = min_count
        self.word2idx = {"<SOS>": 0}
        self.idx2word = {0: "<SOS>"}
        self.n_words = self.process_sents()

    def process_sents(self):
        words = []
        for src, _ in self.sents:
            words += src.split()

        cc = 1
        counter = Counter(words)
        for word, num in counter.items():
            if num > self.min_count:
                self.word2idx[word] = cc
                self.idx2word[cc] = word
                cc += 1
        self.word2idx["<PAD>"] = cc # as CrossEntropyLoss ignore_index so last
        self.idx2word[cc] = "<PAD>"
        cc += 1

        # self.word2idx["<EOS>"] = cc # as CrossEntropyLoss ignore_index so last
        # self.idx2word[cc] = "<EOS>"
        # cc += 1
        return cc


def main(paths):
    # read data
    pairs, train_pairs, test_pairs = get_pairs()

    lang = Lang(pairs)

    # load word2vec
    with open("/data/cc/crawl-300d-2M.pkl", 'rb') as pf:
        w2v = pickle.load(pf)

    pretrain_num = 0
    embedding_matrix = np.random.rand(lang.n_words, 300)
    for word, idx in lang.word2idx.items():
        if word in w2v:
            embedding_matrix[idx] = w2v[word]
            pretrain_num += 1
    print("There are {} words with pre-trained in {} vocabs.\n".format(pretrain_num, lang.n_words))
    np.save(paths['embed'], embedding_matrix) # saved as EmbeddingMatrix.npy

    # split data
    xtrain, ytrain, yids_train = get_train_test(train_pairs, lang)
    xtest, ytest, yids_test = get_train_test(test_pairs, lang)
    assert len(xtrain) == len(ytrain) and len(xtrain) == len(yids_train)
    assert len(xtest) == len(ytest) and len(xtest) == len(yids_test)

    # save data
    data_1 = {'train' : {'x': xtrain, 'y': ytrain, 'yloc': yids_train},
              'test' : {'x': xtest, 'y': ytest, 'yloc': yids_test},
              'word2idx' : lang.word2idx,
              'idx2word' : lang.idx2word}

    with open(paths['data'], 'wb') as pl:
        pickle.dump(data_1, pl)


if __name__ == '__main__':
    paths = {'data': './data3.pkl', 'embed': 'EMatrix_3'}
    main(paths)
