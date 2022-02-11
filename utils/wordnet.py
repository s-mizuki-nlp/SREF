#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import List, Set
from anytree import AnyNode

import nltk
try:
    nltk.data.find("wordnet")
except Exception as e:
    print("downloading wordnet...")
    nltk.download("wordnet")

from nltk.corpus import wordnet as wn

def get_sk_type(sensekey):
    return int(sensekey.split('%')[1].split(':')[0])

def lemma_key_to_pos(sk, tagtype='long'):
    # merges ADJ with ADJ_SAT

    if tagtype == 'long':
        type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
        return type2pos[get_sk_type(sk)]

    elif tagtype == 'short':
        type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
        return type2pos[get_sk_type(sk)]

def lemma_key_to_lemma_name(sensekey):
    return sensekey.split('%')[0]

def is_isolated_synset(synset: wn.synset, include_instance_of_relation: bool):
    if include_instance_of_relation:
        return len(synset.hypernyms()) + len(synset.instance_hypernyms()) + len(synset.hyponyms()) + len(synset.instance_hyponyms()) == 0
    else:
        return len(synset.hypernyms()) + len(synset.hyponyms()) == 0

def synset_depth(synset: wn.synset):
    return synset.max_depth()

def synset_to_lemma_keys(synset: wn.synset):
    return [lemma.key() for lemma in synset.lemmas()]

def synset_to_lemma_key_and_names(synset: wn.synset, include_instance_of_lemmas: bool = True):
    dict_ret = {}
    for lemma in synset.lemmas():
        if synset.pos() == "n" and (not include_instance_of_lemmas):
            if is_instance_of_lemma(lemma.name(), synset.pos()):
                continue
        dict_ret[lemma.key()] = lemma.name()

    return dict_ret

def lemma_key_to_synset_id(lemma_key: str):
    try:
        return wn.lemma_from_key(lemma_key).synset().name()
    except:
        lemma_name = lemma_key_to_lemma_name(lemma_key)
        pos = lemma_key_to_pos(lemma_key, tagtype="short")
        for synset in wn.synsets(lemma_name, pos):
            if lemma_key in synset_to_lemma_keys(synset):
                return synset.name()
        raise ValueError(f"No synset found for key {lemma_key}")

# instance-of lemmaを判定する関数
def is_instance_of_lemma(str_lemma: str, pos: str):
    lst_lemmas = wn.lemmas(str_lemma, pos)
    assert len(lst_lemmas) > 0, ValueError(f"invalid lemma: {str_lemma}|{pos}")
    # どのsynsetもinstance-of relation以外を持たない場合は instance-of lemmaだと判定する
    n_hypernyms = sum([len(lemma.synset().hypernyms()) for lemma in lst_lemmas])
    if n_hypernyms == 0:
        return True
    else:
        return False


def extract_synset_taxonomy(target_part_of_speech: List[str] = ["n", "v"], valid_synset_ids: Set[str] = None,
                            include_instance_of_lemmas: bool = True):
    ROOT_SYNSETS = {
        "n": "entity.n.01",
        "v": "verb_dummy_root.v.01"
    }
    ROOT_NODES = {
        "entity.n.01": AnyNode(id="entity.n.01", parent=None,
                               parent_synset_id=None, pos="n",
                               lemma_names=wn.synset("entity.n.01").lemma_names(),
                               lexname=wn.synset("entity.n.01").lexname(),
                               lemma_keys=synset_to_lemma_keys(wn.synset("entity.n.01"))),
        "verb_dummy_root.v.01": AnyNode(id="verb_dummy_root.v.01", parent=None,
                                        parent_synset_id=None, pos="v",
                                        lemma_names=[], lexname="verb_dummy_root", lemma_keys=[])
    }

    # 初期化
    dict_synset_taxonomy = {}
    dict_synset_taxonomy.update(ROOT_NODES)

    # Nodeを登録．parentはNoneにしておく
    for pos in target_part_of_speech:
        root_synset_id = ROOT_SYNSETS[pos]
        print(f"root: {root_synset_id}")

        for synset in wn.all_synsets(pos=pos):
            if synset.name() in dict_synset_taxonomy:
                continue
            if valid_synset_ids is not None:
                if synset.name() not in valid_synset_ids:
                    continue

            lst_hypernyms = synset.hypernyms() + synset.instance_hypernyms()

            n_hypernym = len(lst_hypernyms)
            if n_hypernym == 0:
                if pos == "n":
                    # noun root must be `entity.n.01`.
                    assert synset.name() == root_synset_id, f"unexpected root node: {synset.name()}"
                    parent_synset_id = root_synset_id
                elif pos == "v":
                    parent_synset_id = root_synset_id
            elif n_hypernym == 1:
                hypernym = lst_hypernyms[0]
                parent_synset_id = hypernym.name()
            elif n_hypernym > 1:
                # take the deepest hypernym among possible hypernyms.
                hypernym = sorted(lst_hypernyms, key=synset_depth, reverse=True)[0]
                parent_synset_id = hypernym.name()

            # get lemma info
            dict_lemma_key_and_names = synset_to_lemma_key_and_names(synset, include_instance_of_lemmas=include_instance_of_lemmas)

            # create new node
            node = AnyNode(id=synset.name(), parent=None,
                           parent_synset_id=parent_synset_id, lexname=synset.lexname(),
                           pos=pos, lemma_names=list(dict_lemma_key_and_names.values()),
                           lemma_keys=list(dict_lemma_key_and_names.keys()))
            dict_synset_taxonomy[synset.name()] = node

    # parentを登録．もしもparent nodeが見つからない場合は異常．
    for synset_id in dict_synset_taxonomy.keys():
        node = dict_synset_taxonomy[synset_id]
        root_synset_id = ROOT_SYNSETS[node.pos]
        if synset_id == root_synset_id:
            print(f"root: {node.id}")
            continue

        parent_node = dict_synset_taxonomy.get(node.parent_synset_id, None)
        node.parent = parent_node

    return dict_synset_taxonomy
