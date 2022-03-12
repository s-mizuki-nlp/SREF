import sys, io, os
import logging
import argparse
from pprint import pprint
from functools import lru_cache
from collections import defaultdict
from collections import Counter
import xml.etree.ElementTree as ET
import numpy as np
from nltk.corpus import wordnet as wn

from bert_as_service import BertEncoder
from probability_space import SenseRepresentationModel
from utils.wordnet import lemma_key_to_pos

from synset_expand import retrieve_sense, gloss_extend
import pickle

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def load_wsd_fw_set(wsd_fw_set_path):
    """Parse XML of split set and return list of instances (dict)."""
    eval_instances = []
    tree = ET.parse(wsd_fw_set_path)
    for text in tree.getroot():
        for sent_idx, sentence in enumerate(text):
            inst = {'tokens': [], 'tokens_mw': [], 'lemmas': [], 'senses': [], 'pos': []}
            for e in sentence:
                inst['tokens_mw'].append(e.text)
                inst['lemmas'].append(e.get('lemma'))
                inst['senses'].append(e.get('id'))
                inst['pos'].append(e.get('pos'))

            inst['tokens'] = sum([t.split() for t in inst['tokens_mw']], [])

            # handling multi-word expressions, mapping allows matching tokens with mw features
            idx_map_abs = []
            idx_map_rel = [(i, list(range(len(t.split()))))
                           for i, t in enumerate(inst['tokens_mw'])]
            token_counter = 0
            for idx_group, idx_tokens in idx_map_rel:  # converting relative token positions to absolute
                idx_tokens = [i+token_counter for i in idx_tokens]
                token_counter += len(idx_tokens)
                idx_map_abs.append([idx_group, idx_tokens])

            inst['tokenized_sentence'] = ' '.join(inst['tokens'])
            inst['idx_map_abs'] = idx_map_abs
            inst['idx'] = sent_idx

            eval_instances.append(inst)

    return eval_instances


@lru_cache()
def wn_sensekey2synset(sensekey):
    """Convert sensekey to synset."""
    lemma = sensekey.split('%')[0]
    for synset in wn.synsets(lemma):
        for lemma in synset.lemmas():
            if lemma.key() == sensekey:
                return synset
    return None


def get_id2sks(wsd_eval_keys):
    """Maps ids of split set to sensekeys, just for in-code evaluation."""
    id2sks = {}
    with open(wsd_eval_keys) as keys_f:
        for line in keys_f:
            id_ = line.split()[0]
            keys = line.split()[1:]
            id2sks[id_] = keys
    return id2sks


def run_scorer(wsd_fw_path, test_set, results_path):
    """Runs the official java-based scorer of the WSD Evaluation Framework."""
    cmd = 'cd %s && java Scorer %s %s' % (wsd_fw_path + 'Evaluation_Datasets/',
                                          '%s/%s.gold.key.txt' % (test_set, test_set),
                                          '../../../../' + results_path)
    print(cmd)
    os.system(cmd)


def chunks(l, n):
    """Yield successive n-sized chunks from given list."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def str_scores(scores, n=3, r=5):
    """Convert scores list to a more readable string."""
    return str([(l, round(s, r)) for l, s in scores[:n]])

@lru_cache()
def wn_all_lexnames_groups():
    groups = defaultdict(list)
    for synset in wn.all_synsets():
        groups[synset.lexname()].append(synset)
    return dict(groups)


def sec_wsd(matches):
    lexname_groups = wn_all_lexnames_groups()
    preds = [sk for sk, sim in matches if sim > args.thresh][:]
    preds_sim = [sim for sk, sim in matches if sim > args.thresh][:]
    norm_predsim = np.exp(preds_sim) / np.sum(np.exp(preds_sim))
    name = locals()
    if len(preds) != 1:
        pos2type = {'ADJ': 'as', 'ADV': 'r', 'NOUN': 'n', 'VERB': 'v'}
        synset_list = retrieve_sense(curr_lemma, pos2type[curr_postag])
        keys = [k[0] for k in matches][:2]
        try:
            synsets = {wn.lemma_from_key(j).synset(): i for i, j in enumerate(keys)}
        except:
            synsets = {
                [wn.synset(k) for k in synset_list if j in [l.key() for l in wn.synset(k).lemmas()]][0]: i for
                i, j in enumerate(keys)}
        strategy = 'all-relations'
        # print([i.lexname() for i in synsets])
        all_related = Counter()
        for potential_synset in synsets.keys():
            name[potential_synset.name()] = set(gloss_extend(potential_synset.name(), strategy))
            all_related.update(list(name[potential_synset.name()]))

        for synset, count in all_related.items():
            if count == 1:
                continue
            for potential_synset in synsets.keys():
                while synset in name[potential_synset.name()]:
                    name[potential_synset.name()].remove(synset)

        for synset_index, potential_synset in enumerate(synsets.keys()):
            lexname = potential_synset.lexname()
            name['sim_%s' % potential_synset.name()] = dict()

            if len(set([i.lexname() for i in synsets])) > 1:
                combine_list = list(name[potential_synset.name()]) + lexname_groups[lexname]
            else:
                combine_list = list(name[potential_synset.name()])
            for synset in combine_list:
                if synset in synsets.keys() and curr_postag not in ['ADJ', 'ADV']:
                    continue
                sim = np.dot(curr_vector, model.get_vec(synset.lemmas()[0].key()))
                name['sim_%s' % potential_synset.name()][synset] = (
                    sim, 'relation' if synset in name[potential_synset.name()] else 'lexname')

        key_score = {keys[j]: preds_sim[j] + np.sum(
            sorted([syn[0] for syn in name['sim_%s' % i.name()].values()], reverse=True)[:1]) for i, j in
                     synsets.items()}

        final_key = [sorted(key_score.items(), key=lambda x: x[1], reverse=True)[0][0]]

    else:
        final_key = preds
    
    return final_key


def _parse_args():

    parser = argparse.ArgumentParser(description='Nearest Neighbors WSD Evaluation.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bert_host", required=True, help="bert-as-service hostname and ip address. e.g. localhost:5555")
    parser.add_argument('--sense_representation_path', type=str, required=True, help="Path to enhanced lemma-key embeddings.")
    parser.add_argument('--sense_representation_type', type=str, required=True, choices=["MultiNormal", "vonMisesFisher"],
                        help="probability representation type of the sense representation.")
    parser.add_argument('--similarity_metric', type=str, choices=["cosine", "l2", "likelihood", "likelihood_wo_norm", "hierarchical"],
                        help="Similarity metric used for nearest neighbor lookup.")
    parser.add_argument('--wsd_fw_path', help='Path to WSD Evaluation Framework', required=False,
                        default='data/wsd_eval/WSD_Evaluation_Framework/')
    parser.add_argument("--work_dir", type=str, required=False, default="./results/ours/", help="directory used for temporary output. DEFAULT: ./results/ours/")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (BERT)', required=False)
    parser.add_argument('--merge_strategy', type=str, default='mean', choices=["first", "sum", "mean"],
                        help='How to merge subwords of target word. DEFAULT: mean', required=False)
    parser.add_argument('--thresh', type=float, default=-float("inf"), help='Similarity threshold. DEFAULT: -inf (=disable threshold)', required=False)
    parser.add_argument('--k', type=int, default=1, help='Number of Neighbors to accept', required=False)
    parser.add_argument('--try_again', action="store_true", help='enable Try-Again mechanism. DEFAULT: False')
    parser.add_argument("--sanity_check", action="store_true", help="enable sanity check (e.g., assert candidate lemma keys)")
    parser.add_argument("--disable_bert_encoder", action="store_true", help="disable bert-as-service.")
    parser.add_argument("--disable_bert_layer_pooling", action="store_true", help="disable sum pooling over BERT layers. DEFAULT: False")
    parser.add_argument("--force_cosine_similarity_for_adj_and_adv", action="store_true", help="force cosine similarity for adjective and adverb. DEFAULT: False")
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = _parse_args()

    """
    Load sense embeddings for evaluation.
    Check the dimensions of the sense embeddings to guess that they are composed with static embeddings.
    """

    if not args.disable_bert_encoder:
        logging.info(f"connecting to bert-as-service: {args.bert_host}")
        host_info = args.bert_host.split(":")
        if len(host_info) == 1:
            bert_encoder = BertEncoder(host=host_info[0])
        elif len(host_info) == 2:
            bert_encoder = BertEncoder(host=host_info[0], port=int(host_info[1]))
        else:
            raise ValueError(f"unexpected `bert_host` value: {args.bert_host}")

    logging.info(f"similarity_metric:{args.similarity_metric}, try-again mechanism:{args.try_again}")
    logging.info(f"Loading sense representation from: {args.sense_representation_path}")

    logging.info("initialize sense representation model.")
    model = SenseRepresentationModel(path=args.sense_representation_path,
                                     repr_type=args.sense_representation_type,
                                     default_similarity_metric=args.similarity_metric,
                                     TEST_MODE=args.sanity_check)
    logging.info("sense representation model summary is as follows.")
    pprint(model.metadata)

    """
    Initialize various counters for calculating supplementary metrics for ALL dataset.
    """
    num_all, num_correct = 0, 0
    pos_correct, pos_all = np.array([0]*4), np.array([0]*4)
    mfs_correct, mfs_all = 0, 0
    lfs_correct, lfs_all = 0, 0
    pos_position = ['NOUN', 'VERB', 'ADJ', 'ADV']
    for data_index, test_set in enumerate(['senseval2', 'senseval3', 'semeval2007', 'semeval2013', 'semeval2015']):
        """
        Initialize various counters for calculating supplementary metrics for each test set.
        """
        logging.info(f"dataset: {test_set}")

        n_instances, n_correct, n_unk_lemmas = 0, 0, 0
        correct_idxs = []
        num_options = []
        failed_by_pos = defaultdict(list)
        pos_confusion = defaultdict(lambda : defaultdict(int))

        """
        Load evaluation instances and gold labels.
        Gold labels (sensekeys) only used for reporting accuracy during evaluation.
        """
        wsd_fw_set_path = args.wsd_fw_path + f"Evaluation_Datasets/{test_set}/{test_set}.data.xml"
        wsd_fw_gold_path = args.wsd_fw_path + f"Evaluation_Datasets/{test_set}/{test_set}.gold.key.txt"
        id2senses = get_id2sks(wsd_fw_gold_path)
        logging.info(f"loading evalset from original file: {wsd_fw_set_path}")
        eval_instances = load_wsd_fw_set(wsd_fw_set_path)

        """
        Iterate over evaluation instances and write predictions in WSD_Evaluation_Framework's format.
        File with predictions is processed by the official scorer after iterating over all instances.
        """

        # temporary file used for saving prediction results.
        sense_repr_name, _ = os.path.splitext(os.path.basename(args.sense_representation_path))
        result_filename = f"{sense_repr_name}_{args.similarity_metric}_TA-{args.try_again}_{test_set}.key"
        results_path = os.path.join(args.work_dir, result_filename)

        with open(results_path, 'w') as results_f:
            for batch_idx, batch in enumerate(chunks(eval_instances, args.batch_size)):
                batch_sents = [sent_info['tokenized_sentence'] for sent_info in batch]

                # process contextual embeddings in sentences batches of size args.batch_size
                batch_bert = bert_encoder.bert_embed(batch_sents, merge_strategy=args.merge_strategy, apply_sum_pooling=True)

                for sent_info, sent_bert in zip(batch, batch_bert):
                    idx_map_abs = sent_info['idx_map_abs']

                    for mw_idx, tok_idxs in idx_map_abs:
                        curr_sense = sent_info['senses'][mw_idx]

                        if curr_sense is None:
                            continue

                        curr_lemma = sent_info['lemmas'][mw_idx]
                        curr_postag = sent_info['pos'][mw_idx]

                        if curr_lemma not in model.known_lemmas:
                            logging.warning(f"unknown lemma. ignore: {curr_lemma}|{curr_postag}")
                            n_unk_lemmas += 1
                            continue  # skips hurt performance in official scorer

                        curr_tokens = [sent_info['tokens'][i] for i in tok_idxs]
                        curr_vector = np.array([sent_bert[i][1] for i in tok_idxs]).mean(axis=0)

                        """
                        Matches test-time embedding against sense embeddings in SensesVSM.
                        use_lemma and use_pos flags condition filtering of candidate senses.
                        Matching is actually cosine similarity (most similar), or 1-NN.
                        """
                        if curr_postag in ('NOUN', 'VERB'):
                            metric = args.similarity_metric
                        elif curr_postag in ('ADJ', 'ADV'):
                            if args.force_cosine_similarity_for_adj_and_adv:
                                metric = "cosine"
                            else:
                                metric = args.similarity_metric
                        lst_predicted_senses = model.match_senses(metric=metric,
                                                                  entity_embedding=curr_vector,
                                                                  lemma=curr_lemma, postag=curr_postag,
                                                                  topn=None)
                        num_options.append(len(lst_predicted_senses))

                        # predictions can be further filtered by similarity threshold or number of accepted neighbors
                        # if specified in CLI parameters
                        preds = [lemma_key for lemma_key, similarity in lst_predicted_senses if similarity > args.thresh][:args.k]
                        if args.try_again:
                            preds = sec_wsd(lst_predicted_senses)[:1]
                        if len(preds) > 0:
                            results_f.write('%s %s\n' % (curr_sense, preds[0]))

                        """
                        Processing additional performance metrics.
                        """

                        # check if our prediction(s) was correct, register POS of mistakes
                        n_instances += 1
                        wsd_correct = False
                        gold_sensekeys = id2senses[curr_sense]
                        pos_dict = {'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r', 'VERB': 'v'}
                        wn1st = [i.key() for i in wn.synsets(curr_lemma, pos_dict[curr_postag])[0].lemmas()]
                        if len(set(preds).intersection(set(gold_sensekeys))) > 0:
                            n_correct += 1
                            wsd_correct = True
                            if len(preds) > 0:
                                failed_by_pos[curr_postag].append((preds[0], gold_sensekeys))
                            else:
                                failed_by_pos[curr_postag].append((None, gold_sensekeys))

                        # register if our prediction belonged to a different POS than gold
                        if len(preds) > 0:
                            pred_sk_pos = lemma_key_to_pos(preds[0])
                            gold_sk_pos = lemma_key_to_pos(gold_sensekeys[0])
                            pos_confusion[gold_sk_pos][pred_sk_pos] += 1

                        # register how far the correct prediction was from the top of our matches
                        correct_idx = None
                        for idx, (matched_sensekey, matched_score) in enumerate(lst_predicted_senses):
                            if matched_sensekey in gold_sensekeys:
                                correct_idx = idx
                                correct_idxs.append(idx)
                                break

        logging.info('Running official scorer ...')
        run_scorer(args.wsd_fw_path, test_set, results_path)
        num_all += n_instances
        num_correct += n_correct
        pos_all += np.array([sum(pos_confusion[i].values()) for i in pos_position])
        pos_correct += np.array([len(failed_by_pos[i]) for i in pos_position])
    print('F-all %f' % (num_correct/num_all))
    print(pos_position, pos_all.tolist(), pos_correct.tolist(), (pos_correct/pos_all).tolist())
