from typing import Dict, List
from time import time
from functools import lru_cache
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn


def get_sk_type(sensekey):
    return int(sensekey.split('%')[1].split(':')[0])


def get_sk_pos(sk, tagtype='long'):
    # merges ADJ with ADJ_SAT

    if tagtype == 'long':
        type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
        return type2pos[get_sk_type(sk)]

    elif tagtype == 'short':
        type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
        return type2pos[get_sk_type(sk)]

def get_sk_lemma(sensekey):
    return sensekey.split('%')[0]

class SensesVSM(object):

    def __init__(self, vecs_path, normalize=True, TEST_MODE: bool = False):
        self.vecs_path = vecs_path
        self.lemma_keys = []
        self.lemma_key_embeddings = np.array([], dtype=np.float32)
        self.lemma_key_index = {}
        self.ndims = 0
        self._TEST_MODE = TEST_MODE

        if type(self.vecs_path) is dict:
            self.load_dict(self.vecs_path)
        else:
            if self.vecs_path.endswith('.txt'):
                self.load_txt(self.vecs_path)
            elif self.vecs_path.endswith('.npz'):
                self.load_npz(self.vecs_path)

        self.load_aux_senses()

        if normalize:
            self.normalize()

    def load_txt(self, txt_vecs_path):
        self.lemma_key_embeddings = []
        with open(txt_vecs_path, encoding='utf-8') as vecs_f:
            for line_idx, line in enumerate(vecs_f):
                elems = line.split()
                self.lemma_keys.append(elems[0])
                self.lemma_key_embeddings.append(np.array(list(map(float, elems[1:])), dtype=np.float32))
        self.lemma_key_embeddings = np.vstack(self.lemma_key_embeddings)

        self.labels_set = set(self.lemma_keys)
        self.lemma_key_index = {l: i for i, l in enumerate(self.lemma_keys)}
        self.ndims = self.lemma_key_embeddings.shape[1]

    def load_dict(self, dict_lemma_key_embeddings: Dict[str, List[float]]):
        self.lemma_key_embeddings = []

        for lemma_key, embedding in dict_lemma_key_embeddings.items():
            self.lemma_keys.append(lemma_key)
            self.lemma_key_embeddings.append(np.array(list(map(float, embedding)), dtype=np.float32))
        self.lemma_key_embeddings = np.vstack(self.lemma_key_embeddings)

        self.labels_set = set(self.lemma_keys)
        self.lemma_key_index = {lemma_key: index for index, lemma_key in enumerate(self.lemma_keys)}
        self.ndims = self.lemma_key_embeddings.shape[1]

    def load_npz(self, npz_vecs_path):
        loader = np.load(npz_vecs_path)
        self.lemma_keys = loader['labels'].tolist()
        self.lemma_key_embeddings = loader['vectors']

        self.labels_set = set(self.lemma_keys)
        self.lemma_key_index = {l: i for i, l in enumerate(self.lemma_keys)}
        self.ndims = self.lemma_key_embeddings.shape[1]

    def load_aux_senses(self):

        self.lemma_key_to_lemma = {lemma_key: get_sk_lemma(lemma_key) for lemma_key in self.lemma_keys}
        self.lemma_key_to_postag = {lemma_key: get_sk_pos(lemma_key) for lemma_key in self.lemma_keys}

        self.lemma_to_lemma_keys = defaultdict(list)
        for lemma_key, lemma in self.lemma_key_to_lemma.items():
            self.lemma_to_lemma_keys[lemma].append(lemma_key)
        self.known_lemmas = set(self.lemma_to_lemma_keys.keys())

        self.pos_to_lemma_keys = defaultdict(list)
        for lemma_key in self.lemma_keys:
            self.pos_to_lemma_keys[self.lemma_key_to_postag[lemma_key]].append(lemma_key)
        self.known_postags = set(self.pos_to_lemma_keys.keys())

    def save_npz(self):
        npz_path = self.vecs_path.replace('.txt', '.npz')
        np.savez_compressed(npz_path,
                            labels=self.lemma_keys,
                            vectors=self.lemma_key_embeddings)

    def normalize(self, norm='l2'):
        norms = np.linalg.norm(self.lemma_key_embeddings, axis=1)
        self.lemma_key_embeddings = (self.lemma_key_embeddings.T / norms).T

    def get_vec(self, label):
        return self.lemma_key_embeddings[self.lemma_key_index[label]]

    def similarity(self, label1, label2):
        v1 = self.get_vec(label1)
        v2 = self.get_vec(label2)
        return np.dot(v1, v2).tolist()

    def get_candidate_lemma_keys(self, lemma: str, postag: str):
        lst_lemma_keys = []
        for lemma_key in self.lemma_keys:
            if (lemma is None) or (self.lemma_key_to_lemma[lemma_key] == lemma):
                if (postag is None) or (self.lemma_key_to_postag[lemma_key] == postag):
                    lst_lemma_keys.append(lemma_key)

        if self._TEST_MODE:
            pos_short = getattr(wn, postag) # ex. NOUN -> n
            expected = set([lemma.key() for lemma in wn.lemmas(lemma, pos_short)])
            actual = set(lst_lemma_keys)
            assert expected == actual, f"wrong candidate lemma keys: {lemma}|{postag}"

        return lst_lemma_keys

    def match_senses(self, vec, lemma=None, postag=None, topn=100):
        lst_candidate_lemma_keys = self.get_candidate_lemma_keys(lemma, postag)
        lst_candidate_lemma_embedding_indices = list(map(self.lemma_key_index.get, lst_candidate_lemma_keys))

        sims = np.dot(self.lemma_key_embeddings[lst_candidate_lemma_embedding_indices], np.array(vec))
        matches = list(zip(lst_candidate_lemma_keys, sims))
        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        return matches[:topn]

    def most_similar_vec(self, vec, topn=10):
        sims = np.dot(self.lemma_key_embeddings, vec).astype(np.float32)
        sims_ = sims.tolist()
        r = []
        for top_i in sims.argsort().tolist()[::-1][:topn]:
            r.append((self.lemma_keys[top_i], sims_[top_i]))
        return r

    def sims(self, vec):
        return np.dot(self.lemma_key_embeddings, np.array(vec)).tolist()


class VSM(object):

    def __init__(self, vecs_path, normalize=True):
        self.labels = []
        self.vectors = np.array([], dtype=np.float32)
        self.indices = {}
        self.ndims = 0

        self.load_txt(vecs_path)

        if normalize:
            self.normalize()

    def load_txt(self, vecs_path):
        self.vectors = []
        with open(vecs_path, encoding='utf-8') as vecs_f:
            for line_idx, line in enumerate(vecs_f):
                elems = line.split()
                self.labels.append(elems[0])
                self.vectors.append(np.array(list(map(float, elems[1:])), dtype=np.float32))

                # if line_idx % 100000 == 0:
                #     print(line_idx)

        self.vectors = np.vstack(self.vectors)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def normalize(self, norm='l2'):
        self.vectors = (self.vectors.T / np.linalg.norm(self.vectors, axis=1)).T

    def get_vec(self, label):
        return self.vectors[self.indices[label]]

    def similarity(self, label1, label2):
        v1 = self.get_vec(label1)
        v2 = self.get_vec(label2)
        return np.dot(v1, v2).tolist()

    def most_similar_vec(self, vec, topn=10):
        sims = np.dot(self.vectors, vec).astype(np.float32)
        sims_ = sims.tolist()
        r = []
        for top_i in sims.argsort().tolist()[::-1][:topn]:
            r.append((self.labels[top_i], sims_[top_i]))
        return r

    def sims(self, vec):
        return np.dot(self.vectors, np.array(vec)).tolist()
