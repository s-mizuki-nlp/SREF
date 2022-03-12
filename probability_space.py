#!/usr/bin/env python
# -*- coding:utf-8 -*-
import copy
from typing import Optional, List, Tuple, Union
import sys, io, os, pickle
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
from nltk.corpus import wordnet as wn

from utils.wordnet import extract_synset_taxonomy, lemma_key_to_synset_id, lemma_key_to_pos, lemma_key_to_lemma_name
from distribution.continuous import MultiVariateNormal, vonMisesFisher
from distribution import distance_mvn

prob_dist_types = Union[MultiVariateNormal, vonMisesFisher]

class SenseRepresentationModel(object):

    _AVAILABLE_REPR_TYPES = ("MultiNormal", "vonMisesFisher")

    def __init__(self, path: str, repr_type: str, default_similarity_metric: str, TEST_MODE: bool = False):
        assert repr_type in self._AVAILABLE_REPR_TYPES, f"invalid repr_type value: {repr_type}"

        self._path = path
        self._repr_type = repr_type
        self._TEST_MODE = TEST_MODE

        with io.open(path, mode="rb") as ifs:
            object = pickle.load(ifs)
            self._dict_lemma_sense_representations = object["lemma"]
            self._dict_synset_sense_representations = object["synset"]
            self._metadata = object.get("meta", {})

        self._similarity_metric = default_similarity_metric

        if default_similarity_metric == "hierarchical":
            test_repr = next(iter(self._dict_synset_sense_representations.values()))
            assert test_repr is not None, f"synset sense repr. is not available. this file cannot be used for metric=hierarchical: {path}"

            # dict_synset_taoxnomy[synset_id] = AnyNode[Synset]
            # lemma keys: node.lemma_keys, lemma_names: node.lemma_names
            # parent: node.parent, children: node.children
            self._dict_synset_taxonomy = extract_synset_taxonomy(target_part_of_speech=["n","v"], include_instance_of_lemmas=True)

        self.load_aux_senses()

    @property
    def metadata(self):
        dict_ret = self._metadata
        dict_ret["repr_type"] = self._repr_type
        dict_ret["default_similarity_metric"] = self._similarity_metric
        return dict_ret

    @property
    def lemma_keys(self):
        return self._dict_lemma_sense_representations.keys()

    def load_aux_senses(self):
        self.lemma_key_to_lemma = {lemma_key: lemma_key_to_lemma_name(lemma_key) for lemma_key in self.lemma_keys}
        self.lemma_key_to_postag = {lemma_key: lemma_key_to_pos(lemma_key) for lemma_key in self.lemma_keys}
        self.lemma_key_to_synset = {lemma_key: lemma_key_to_synset_id(lemma_key) for lemma_key in self.lemma_keys}

        self.lemma_to_lemma_keys = defaultdict(list)
        for lemma_key, lemma in self.lemma_key_to_lemma.items():
            self.lemma_to_lemma_keys[lemma].append(lemma_key)
        self.known_lemmas = set(self.lemma_to_lemma_keys.keys())

        self.pos_to_lemma_keys = defaultdict(list)
        for lemma_key in self.lemma_keys:
            self.pos_to_lemma_keys[self.lemma_key_to_postag[lemma_key]].append(lemma_key)
        self.known_postags = set(self.pos_to_lemma_keys.keys())

    def get_candidate_lemma_keys(self, lemma: str, postag: str):
        lst_lemma_keys = []
        for lemma_key in self.lemma_keys:
            if self.lemma_key_to_lemma[lemma_key] == lemma:
                if self.lemma_key_to_postag[lemma_key] == postag:
                    lst_lemma_keys.append(lemma_key)

        if self._TEST_MODE:
            pos_short = getattr(wn, postag) # ex. NOUN -> n
            expected = set([lemma.key() for lemma in wn.lemmas(lemma, pos_short)])
            actual = set(lst_lemma_keys)
            assert expected == actual, f"wrong candidate lemma keys: {lemma}|{postag}"

        return lst_lemma_keys

    def match_senses(self, entity_embedding: np.ndarray, lemma: str, postag: str,
                     metric: Optional[str] = None, topn: Optional[int] = None) -> List[Tuple[str, float]]:
        lst_candidate_lemma_keys = self.get_candidate_lemma_keys(lemma, postag)

        if entity_embedding.ndim == 1:
            entity_embedding = entity_embedding.reshape(1, -1)

        lst_similarities = [self.calc_similarity(query_embeddings=entity_embedding, lemma_key=lemma_key, metric=metric) for lemma_key in lst_candidate_lemma_keys]
        matches = list(zip(lst_candidate_lemma_keys, lst_similarities))
        matches = sorted(matches, key=lambda x: x[1], reverse=True)

        if self._TEST_MODE:
            for idx, (lemma_key, similarity) in enumerate(matches):
                print(f"{idx}: {lemma_key} -> {similarity:1.3f}")

        if topn is None:
            return matches
        else:
            return matches[:topn]

    def load_probability_distribution(self, lemma_key_or_synset_id: str) -> prob_dist_types:
        if lemma_key_or_synset_id in self._dict_lemma_sense_representations:
            params = self._dict_lemma_sense_representations[lemma_key_or_synset_id]
        elif lemma_key_or_synset_id in self._dict_synset_sense_representations:
            params = self._dict_synset_sense_representations[lemma_key_or_synset_id]
        else:
            raise ValueError(f"invalid lemma key or synset id: {lemma_key_or_synset_id}")

        if self._repr_type == "MultiNormal":
            prob_dist = MultiVariateNormal.deserialize(params)
        elif self._repr_type == "vonMisesFisher":
            prob_dist = vonMisesFisher.deserialize(params)

        return prob_dist

    def calc_similarity(self, query_embeddings: np.ndarray, lemma_key: str, metric: Optional[str] = None) -> float:
        """
        calc similarity between query embeddings and lemma representations (specified by lemma key). Greater is more similar.

        :param query_embeddings: query embeddings (>=1). for vector-type metric, it returns average similarity.
                    for prob. dist. type metric, these are first used to estimate prob. dist. then calculate similarity between prob. dists.
        :param lemma_key: lemma key. e.g., hood%1:15:00::
        :param metric: specify when you overwrite default similarity metric.
        :return: similarity value.
        """
        # query_embeddings: (n_sample, n_dim)
        metric = self._similarity_metric if metric is None else metric
        prob_dist = self.load_probability_distribution(lemma_key)

        if metric == "cosine":
            lst_sims = [self.calc_cosine_similarity(query_emb, prob_dist) for query_emb in query_embeddings]
        elif metric == "l2":
            lst_sims = [self.calc_l2_similarity(query_emb, prob_dist) for query_emb in query_embeddings]
        elif metric == "likelihood":
            lst_sims = [self.calc_log_likelihood(query_emb, prob_dist) for query_emb in query_embeddings]
        elif metric == "likelihood_wo_norm":
            lst_sims = [self.calc_unnormalized_log_likelihood(query_emb, prob_dist) for query_emb in query_embeddings]
        elif metric == "hierarchical":
            lst_sims = [self.calc_hierarchical_classification_log_probability(query_emb, lemma_key, normalize=True) for query_emb in query_embeddings]
        elif metric == "hierarchical_wo_norm":
            lst_sims = [self.calc_hierarchical_classification_log_probability(query_emb, lemma_key, normalize=False) for query_emb in query_embeddings]

        similarity = np.mean(lst_sims)

        return similarity

    def calc_cosine_similarity(self, vector: np.ndarray, prob_dist: prob_dist_types) -> float:
        if isinstance(prob_dist, MultiVariateNormal):
            vec_mu = prob_dist.mean
        elif isinstance(prob_dist, vonMisesFisher):
            vec_mu = prob_dist.mu
        sim = np.dot(vector, vec_mu) / ( np.linalg.norm(vector) * np.linalg.norm(vec_mu) + 1E-15 )
        return sim

    def calc_l2_similarity(self, vector: np.ndarray, prob_dist: prob_dist_types) -> float:
        if isinstance(prob_dist, MultiVariateNormal):
            vec_mu = prob_dist.mean
        elif isinstance(prob_dist, vonMisesFisher):
            vec_mu = prob_dist.mu
        sim = - np.linalg.norm(vec_mu - vector)
        return sim

    def calc_unnormalized_log_likelihood(self, vector: np.ndarray, prob_dist: prob_dist_types) -> float:
        return prob_dist.unnormalized_logpdf(vector)

    def calc_log_likelihood(self, vector: np.ndarray, prob_dist: prob_dist_types) -> float:
        return prob_dist.logpdf(vector)

    def _calc_classificaiton_log_probability(self, parent_synset_id: str, target_child_synset_id: str, vector: np.ndarray, normalize: bool) -> float:
        target_log_prob = self.load_probability_distribution(target_child_synset_id).logpdf(vector)

        if not normalize:
            return target_log_prob

        # calculate denominator = logsumexp( logPr{x|c} )
        lst_child_synset_ids = [node.id for node in self._dict_synset_taxonomy[parent_synset_id].children]
        if len(lst_child_synset_ids) == 1:
            return 0.0

        lst_log_probs = [self.load_probability_distribution(synset_id).logpdf(vector) for synset_id in lst_child_synset_ids]
        denominator = logsumexp(lst_log_probs)

        return target_log_prob - denominator

    def calc_hierarchical_classification_log_probability(self, vector: np.ndarray, lemma_key: str, normalize: bool = True) -> float:
        synset_id = lemma_key_to_synset_id(lemma_key)
        target_synset_node = self._dict_synset_taxonomy[synset_id]
        lst_ancestor_nodes = list(target_synset_node.ancestors) + [target_synset_node]
        lst_ancestor_synset_ids = [node.id for node in lst_ancestor_nodes]
        assert len(lst_ancestor_synset_ids) > 1, f"could not traverse to root: {lemma_key}|{synset_id}"

        lst_log_probs = []
        for parent_synset_id, target_child_synset_id in zip(lst_ancestor_synset_ids[:-1], lst_ancestor_synset_ids[1:]):
            log_classification_prob = self._calc_classificaiton_log_probability(parent_synset_id=parent_synset_id,
                                                                                target_child_synset_id=target_child_synset_id,
                                                                                vector=vector,
                                                                                normalize=normalize)
            lst_log_probs.append(log_classification_prob)

        mean_log_prob = np.mean(lst_log_probs)

        return mean_log_prob