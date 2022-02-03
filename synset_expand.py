from typing import List, Dict
import io, sys, os
from nltk.corpus import wordnet as wn
import logging
import numpy as np
from tqdm import tqdm
import pickle
import argparse

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

def normalize(vec_as_list):
    vector = np.array(vec_as_list)
    vector = vector / np.linalg.norm(vector)
    return vector

def retrieve_sense(word, pos=None):
    """
        retrieve sense glosses, sense inventory and sense frequency of word as a dict, list and list respectively
    """
    sense_inventory = [i for i in wn.synsets(word) if i.name().split('.')[-2] in pos]

    name_list, sense_inventory_final = list(), list()
    for sense in sense_inventory:
        lemma_names = [i.name().lower() for i in sense.lemmas()]
        if word.lower() in lemma_names:
            name = sense.name()
            name_list.append(name)
    return name_list

def load_basic_lemma_embeddings(path: str, l2_norm: bool, return_first_embeddings_only: bool, force_ndim_to_2: bool = False) -> Dict[str, np.ndarray]:
    dict_lemma_key_embeddings = {}

    with io.open(path, mode="rb") as ifs:
        dict_lst_lemma_embeddings = pickle.load(ifs)

    for lemma_key, lst_lemma_key_embeddings in tqdm(dict_lst_lemma_embeddings.items()):
        if return_first_embeddings_only:
            # DUBIOUS: it just accounts for first embedding of each lemma keys.
            vectors = np.array(lst_lemma_key_embeddings[0])
        else:
            vectors = np.array(lst_lemma_key_embeddings)

        # normalize to unit length.
        if l2_norm:
            vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)

        # adjust dimension
        if force_ndim_to_2:
            if vectors.ndim == 1:
                # vectors: (1, n_dim)
                vectors = vectors.reshape(1,-1)

        dict_lemma_key_embeddings[lemma_key] = vectors

    return dict_lemma_key_embeddings

def get_related(names, relation='hypernyms'):
    """
    :param names: the synset list
    :param relation: all the relations
    :return: the extended gloss list with its according synset name
    """
    related_list = []
    for name in names:
        if relation == 'hypernyms':
            wn_relation = wn.synset(name).hypernyms()
        elif relation == 'hyponyms':
            wn_relation = wn.synset(name).hyponyms()
        elif relation == 'part_holonyms':
            wn_relation = wn.synset(name).part_holonyms()
        elif relation == 'part_meronyms':
            wn_relation = wn.synset(name).part_meronyms()
        elif relation == 'member_holonyms':
            wn_relation = wn.synset(name).member_holonyms()
        elif relation == 'member_meronyms':
            wn_relation = wn.synset(name).member_meronyms()
        elif relation == 'entailments':
            wn_relation = wn.synset(name).entailments()
        elif relation == 'attributes':
            wn_relation = wn.synset(name).attributes()
        elif relation == 'also_sees':
            wn_relation = wn.synset(name).also_sees()
        elif relation == 'similar_tos':
            wn_relation = wn.synset(name).similar_tos()
        elif relation == 'causes':
            wn_relation = wn.synset(name).causes()
        elif relation == 'verb_groups':
            wn_relation = wn.synset(name).verb_groups()
        elif relation == 'substance_holonyms':
            wn_relation = wn.synset(name).substance_holonyms()
        elif relation == 'substance_meronyms':
            wn_relation = wn.synset(name).substance_meronyms()
        elif relation == 'usage_domains':
            wn_relation = wn.synset(name).usage_domains()
        elif relation == 'pertainyms':
            wn_relation = [j.synset() for j in sum([i.pertainyms() for i in wn.synset(name).lemmas()], [])]
        elif relation == 'antonyms':
            wn_relation = [j.synset() for j in sum([i.antonyms() for i in wn.synset(name).lemmas()], [])]
        else:
            wn_relation = []
            print('no such relation, process terminated.')
        related_list += wn_relation
    return related_list


def morpho_extend(extended_list):
    morpho_list = list()
    for synset in extended_list:
        morpho_list += [j.synset() for j in
                        list(sum([i.derivationally_related_forms() for i in synset.lemmas()], []))]
    return morpho_list


def gloss_extend(o_sense, emb_strategy) -> List[wn.synset]:
    """
    note: this is the main algorithm for relation exploitation,
    use different relations to retrieve bag-of-synset
    :param o_sense: the potential sense that is under expansion
    :param relation_list: all the available relations that a synset might have, except 'verb_group'
    :return: extended_list_gloss: the bag-of-synset
    """
    extended_list, combine_list = list(), [wn.synset(o_sense)]
    if emb_strategy == "all-relations":
        relation_list = ['hyponyms', 'part_holonyms', 'part_meronyms', 'member_holonyms', 'antonyms',
                     'member_meronyms', 'entailments', 'attributes', 'similar_tos', 'causes', 'pertainyms',
                     'substance_holonyms', 'substance_meronyms', 'usage_domains', 'also_sees']
        extended_list += morpho_extend([wn.synset(o_sense)])
    elif emb_strategy == "hyponymy":
        relation_list = ['hyponyms']
    else:
        raise ValueError(f"undefined `emb_strategy` value: {emb_strategy}")

    # expand the original sense with nearby senses using all relations but hypernyms
    for index, relation in enumerate(relation_list):
        combine_list += get_related([o_sense], relation)

    # expand the original sense with in-depth hypernyms (only one branch)
    for synset in [wn.synset(o_sense)]:
        # 親語義は原則として1個だが，2個以上の場合もある
        extended_list += synset.hypernyms()

    extended_list += combine_list

    return extended_list


def vector_merge(synset_id: str, lst_lemma_keys: List[str],
                 lemma_key_embeddings, emb_strategy) -> Dict[str, List[float]]:
    new_dict = dict()
    extend_synset = gloss_extend(synset_id, emb_strategy)
    for lemma_key in lst_lemma_keys:
        sense_vec = np.array(lemma_key_embeddings[lemma_key])
        for exp_synset in extend_synset:
            distance = wn.synset(synset_id).shortest_path_distance(exp_synset)
            # if distance is unknown, five will be used.
            distance = distance if distance else 5
            for lemma_exp in exp_synset.lemmas():
                sense_vec += 1 / (1 + distance) * np.array(lemma_key_embeddings[lemma_exp.key()])
        sense_vec = sense_vec / np.linalg.norm(sense_vec)
        new_dict[lemma_key] = sense_vec.tolist()
    return new_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply sense embedding enhancement using semantic relation.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-input_path", type=str, help="input path of basic lemma embeddings.", required=True)
    parser.add_argument('-emb_strategy', type=str, default="all-relations",
                        choices=["all-relations", "hyponymy"],
                        help='semantic relations that will be used to lookup related synsets. hypernyms are always used.', required=True)
    parser.add_argument('-out_path', type=str, help='output path of enhanced lemma embeddings.', required=False,
                        default='data/vectors/emb_wn_%s_%s.pkl')
    args = parser.parse_args()

    path_output = args.out_path
    # e.g., aug_gloss+examples
    corpus_name = os.path.splitext(os.path.basename(args.input_path))[0].replace("emb_glosses_", "")
    if path_output.find("%s") != -1:
        path_output = path_output % (args.emb_strategy, corpus_name)
    assert not os.path.exists(path_output), f"file already exists: {path_output}"

    logging.info(f"enhancement strategy: {args.emb_strategy}")
    logging.info(f"result will be saved as: {path_output}")

    emb_strategy = args.emb_strategy

    logging.info(f"Loading basic lemma embeddings: {args.input_path}")
    dict_lemma_key_embeddings = load_basic_lemma_embeddings(path=args.input_path, l2_norm=True, return_first_embeddings_only=True)

    logging.info('Combining lemma-key embeddings using semantic relations...')

    synset_dict = dict()
    for synset in wn.all_synsets():
        lst_lemma_keys = [i.key() for i in synset.lemmas()]
        synset_dict[synset.name()] = lst_lemma_keys

    vector_all = dict()
    for synset_id, lst_lemma_keys in tqdm(synset_dict.items()):
        dict_new_lemma_embs = vector_merge(synset_id=synset_id, lst_lemma_keys=lst_lemma_keys,
                                           lemma_key_embeddings=dict_lemma_key_embeddings, emb_strategy=emb_strategy)
        vector_all.update(dict_new_lemma_embs)

    logging.info(f'num. of all synsets: {len(synset_dict)}')
    logging.info(f'num. of enhanced lemma-key embeddings: {len(vector_all)}')

    with io.open(path_output, mode="wb") as ofs:
        pickle.dump(vector_all, ofs, -1)