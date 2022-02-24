#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Dict, List, Tuple, Union
import os, sys, io, copy, pickle
import argparse
from nltk.corpus import wordnet as wn
import numpy as np
import logging
from pprint import pprint

from synset_expand import load_basic_lemma_embeddings, vector_merge, gloss_extend
from utils.wordnet import extract_synset_taxonomy, synset_to_lemma_keys
from distribution.continuous import MultiVariateNormal
from distribution.prior import NormalInverseWishart


wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


def extract_lemma_keys_and_weights_from_semantically_related_synsets(synset_id: str, distinct: bool = False) -> Tuple[List[str], List[int]]:
    lst_lemma_keys = []; lst_weights = []
    lst_related_synsets = gloss_extend(o_sense=synset_id, emb_strategy="all-relations")
    synset_src = wn.synset(synset_id)
    for synset_rel in lst_related_synsets:
        distance = synset_src.shortest_path_distance(synset_rel)
        # if distance is unknown, five will be used.
        distance = distance if distance else 5
        weight = 1 / (1 + distance)
        for lemma in synset_rel.lemmas():
            if distinct and (lemma.key() in lst_lemma_keys):
                continue
            lst_lemma_keys.append(lemma.key())
            lst_weights.append(weight)

    return lst_lemma_keys, lst_weights

def update_children_priors(parent_node, semantic_relation: str, basic_lemma_embeddings: Dict[str, np.ndarray]):
    # synset prior is the prob. distribution of parent synset.
    prior = parent_node.prior

    # then we collect the parent synset's basic lemma embeddings along with semanticall-related synsets.
    if semantic_relation == "synonym":
        lst_related_lemma_keys = parent_node.lemma_keys
        lst_weights = None
    elif semantic_relation == "all-relations":
        lst_related_lemma_keys, lst_weights = extract_lemma_keys_and_weights_from_semantically_related_synsets(synset_id=parent_node.id)
    elif semantic_relation == "all-relations-wo-weight":
        lst_related_lemma_keys, _ = extract_lemma_keys_and_weights_from_semantically_related_synsets(synset_id=parent_node.id)
        lst_weights = None
    else:
        raise NotImplementedError(f"invalid `semantic_relation` value: {semantic_relation}")

    # we compute posterior prob. distribution of parent synset.
    lst_embs = [basic_lemma_embeddings[lemma_key] for lemma_key in lst_related_lemma_keys]
    if len(lst_embs) > 0:
        mat_s = np.stack(lst_embs).squeeze()
        posterior = prior.posterior(mat_obs=mat_s, sample_weights=lst_weights)
    else:
        posterior = copy.deepcopy(prior)

    # posterior distribution is assigned as the prior distribution of children nodes.
    # i.e., let \pi as prior, \pi_{child}(\theta) = p(\theta|embs_{parent}) \propto p(embs_{parent}|\theta) \pi_{parent}(\theta)
    # synset prior will be used as the prior distribution of of synset or lemma representation. ref: compute_sense_representations_noun_verb()
    for child_node in parent_node.children:
        child_node.prior = posterior
        update_children_priors(parent_node=child_node, semantic_relation=semantic_relation, basic_lemma_embeddings=basic_lemma_embeddings)


def compute_sense_representations_adverb_adjective(synset: wn.synset,
                                                   basic_lemma_embeddings: Dict[str, np.array],
                                                   variance: float = 1.0) -> Dict[str, Dict[str, np.ndarray]]:
    """
    It returns predictive posterior (?) Multivariate Normal distribution for each lemma keys which belong to specified synset.
    mean is equivalent to SREF embedding (=synset relation expansion), variance is 1.0 by default.

    :param synset:
    :param basic_lemma_embeddings:
    :param variance:
    """
    assert synset.pos() in ("a","r","s"), f"invalid synset: {synset.name()}"
    synset_id = synset.name()

    lst_lemma_keys = synset_to_lemma_keys(synset)
    lemma_vectors = vector_merge(synset_id=synset_id, lst_lemma_keys=lst_lemma_keys,
                                 lemma_key_embeddings=basic_lemma_embeddings,
                                 emb_strategy="all-relations")
    dict_ret = {}
    for lemma_key, vector in lemma_vectors.items():
        # vec_cov = np.full(shape=(len(vector),), fill_value=variance)
        # p_dist = MultiVariateNormal(vec_mu=np.array(vector), vec_cov=vec_cov)
        p_dist = MultiVariateNormal(vec_mu=np.array(vector), scalar_cov=variance)
        dict_ret[lemma_key] = p_dist.serialize()
        del p_dist

    return dict_ret

def _compute_posterior_multivariate_normal_params(posterior_distribution: "NormalInverseWishart", method: str) -> Dict[str, np.ndarray]:
    if method == "posterior_predictive":
        # \mu = \mu_{0}; \Sigma = \frac{\Phi}{\nu_0 -p + 1}\frac{\kappa_0+1}{\kappa_0}
        params = posterior_distribution.approx_posterior_predictive().serialize()
    elif method == "mean":
        # \mu = \mu_{0}; \Sigma = \frac{\Phi}{\nu_0 - p - 1}
        vec_mean, vec_cov_diag = posterior_distribution.mean
        # return as dict
        params = {
            "vec_mu": vec_mean,
            "vec_cov": vec_cov_diag
        }
    return params

def compute_sense_representations_noun_verb(synset: wn.synset,
                                            prior_distribution: "NormalInverseWishart",
                                            posterior_parameter_estimatnion: str,
                                            semantic_relation: str,
                                            basic_lemma_embeddings: Dict[str, np.ndarray],
                                            inference_strategy: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], Union[None, Dict[str, np.ndarray]]]:
    dict_ret = {}

    lst_lemma_keys_in_target_synset = synset_to_lemma_keys(synset)

    # collect basic lemma embeddings that are used for updating synset-level prob. dist. (=prior of lemma-level prob. dist.).
    if semantic_relation == "synonym":
        lst_related_lemma_keys = lst_lemma_keys_in_target_synset
        lst_related_lemma_weights = None
    elif semantic_relation == "all-relations":
        lst_related_lemma_keys, lst_related_lemma_weights = extract_lemma_keys_and_weights_from_semantically_related_synsets(synset_id=synset.name())
    elif semantic_relation == "all-relations-wo-weight":
        lst_related_lemma_keys, _ = extract_lemma_keys_and_weights_from_semantically_related_synsets(synset_id=synset.name())
        lst_related_lemma_weights = None
    else:
        raise NotImplementedError(f"invalid `semantic_relation` value: {semantic_relation}")

    dict_lemma_vectors = {lemma_key:basic_lemma_embeddings[lemma_key] for lemma_key in lst_related_lemma_keys}

    if inference_strategy in ("synset-then-lemma", "synset", "synset-and-lemma"):
        # mat_x: all lemma embeddings which relates to the target synset.
        lst_related_lemma_vectors = [basic_lemma_embeddings[lemma_key] for lemma_key in lst_related_lemma_keys]
        mat_s = np.stack(lst_related_lemma_vectors).squeeze()

        # synset-level posterior
        p_posterior_s = prior_distribution.posterior(mat_obs=mat_s, sample_weights=lst_related_lemma_weights)

        for lemma_key in lst_lemma_keys_in_target_synset:
            if inference_strategy == "synset-then-lemma":
                # compute lemma-level posterior if possible.
                if lemma_key in dict_lemma_vectors:
                    # mat_l: (n_obs, n_dim)
                    mat_l = dict_lemma_vectors[lemma_key]
                    p_posterior_s_l = p_posterior_s.posterior(mat_obs=mat_l)
                    dict_ret[lemma_key] = _compute_posterior_multivariate_normal_params(p_posterior_s_l, method=posterior_parameter_estimatnion)
                else:
                    dict_ret[lemma_key] = _compute_posterior_multivariate_normal_params(p_posterior_s, method=posterior_parameter_estimatnion)

            elif inference_strategy == "synset":
                dict_ret[lemma_key] = _compute_posterior_multivariate_normal_params(p_posterior_s, method=posterior_parameter_estimatnion)

            elif inference_strategy == "synset-and-lemma":
                # except the effect of prior, this setup is identical to SREF algorithm.
                # append target lemma embedding with sample weights = 1.0
                lst_embs = [basic_lemma_embeddings[lemma_key]] + lst_related_lemma_vectors
                mat_embs = np.stack(lst_embs).squeeze()
                lst_weights = None if lst_related_lemma_weights is None else [1.0] + lst_related_lemma_weights
                p_posterior_s_plus_l = prior_distribution.posterior(mat_obs=mat_embs, sample_weights=lst_weights)
                dict_ret[lemma_key] = _compute_posterior_multivariate_normal_params(p_posterior_s_plus_l, method=posterior_parameter_estimatnion)

    elif inference_strategy == "lemma":
        p_posterior_s = None

        # lemma-level posterior
        for lemma_key in lst_lemma_keys_in_target_synset:
            if lemma_key in dict_lemma_vectors:
                mat_l = dict_lemma_vectors[lemma_key]
                p_posterior_l = prior_distribution.posterior(mat_obs=mat_l)
                dict_ret[lemma_key] = _compute_posterior_multivariate_normal_params(p_posterior_l, method=posterior_parameter_estimatnion)
            else:
                dict_ret[lemma_key] = _compute_posterior_multivariate_normal_params(prior_distribution, method=posterior_parameter_estimatnion)

    # return lemma-key level repr. and synset level repr (if available).
    if p_posterior_s is None:
        return dict_ret, None
    else:
        return dict_ret, _compute_posterior_multivariate_normal_params(p_posterior_s, method=posterior_parameter_estimatnion)


def _parse_args():

    parser = argparse.ArgumentParser(description='Compute sense representation as prob. distribution based on WordNet synset taxonomy.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_path", type=str, help="input path of basic lemma embeddings.", required=True)
    parser.add_argument("--normalize_lemma_embeddings", action="store_true", help="normalize basic lemma embeddings before inference.")
    parser.add_argument('--inference_strategy', type=str, required=True,
                        choices=["synset-then-lemma", "synset-and-lemma", "synset", "lemma"], help='methodologies that will be applied to sense embeddings.')
    parser.add_argument('--semantic_relation', type=str, required=True,
                        choices=["synonym", "all-relations", "all-relations-wo-weight"],
                        help="semantic relation which are used to update synset-level probability distribution. `all-relations` is identical to SREF [Wang and Wang, EMNLP2020]")
    parser.add_argument("--posterior_inference_method", type=str, required=True, choices=NormalInverseWishart.AVAILABLE_POSTERIOR_INFERENCE_METHOD(),
                        help=f"method used for posterior inference.")
    parser.add_argument("--posterior_inference_parameter_estimation", type=str, required=False, default="mean", choices=["posterior_predictive", "mean"],
                        help=f"parameter estimation method of posterior inference. DEFAULT: mean")
    parser.add_argument('--out_path', type=str, help='output path of sense embeddings.', required=False,
                        default='data/representations/norm-{normalize}_str-{strategy}_semrel-{relation}_posterior-{posterior_inference_method}_k-{kappa:1.4f}_nu-{nu_minus_dof:1.4f}_{lemma_embeddings_name}.pkl')
    parser.add_argument('--kappa', type=float, required=True, help="\kappa for NIW distribution. 0 < \kappa << 1. Smaller is less confident for mean.")
    parser.add_argument('--nu_minus_dof', type=float, required=True, help="\nu - n_dim - 1 for NIW distribution. 0 < \nu_{-DoF}. Smaller is less confident for variance.")
    parser.add_argument('--cov', type=float, required=False, default=-1, help="\Phi = \cov * (\nu_{-DoF})")
    args = parser.parse_args()

    # assertion
    path = args.input_path
    assert os.path.exists(path), f"invalid path specified: {path}"

    assert args.nu_minus_dof > 0, f"`nu_minus_dof` must be positive: {args.nu_minus_dof}"

    # output
    path_output = args.out_path
    if path_output.find("{") != -1:
        lemma_embeddings_name = os.path.splitext(os.path.basename(args.input_path))[0].replace("emb_glosses_", "")
        path_output = path_output.format(normalize=args.normalize_lemma_embeddings,
                                         strategy=args.inference_strategy,
                                         relation=args.semantic_relation,
                                         posterior_inference_method=args.posterior_inference_method,
                                         kappa=args.kappa,
                                         nu_minus_dof=args.nu_minus_dof,
                                         lemma_embeddings_name=lemma_embeddings_name)
    assert not os.path.exists(path_output), f"file already exists: {path_output}"
    logging.info(f"result will be saved as: {path_output}")
    args.out_path = path_output

    return args


if __name__ == "__main__":

    args = _parse_args()

    pprint(vars(args), compact=True)

    # load basic lemma embeddings
    # dict_lemma_key_embeddings: used for our original sense representation
    # dict_lemma_key_embeddings_for_sref: used for SREF sense representation [Wang+, EMNLP2020]
    logging.info(f"Loading basic lemma embeddings: {args.input_path}")
    dict_lemma_key_embeddings = load_basic_lemma_embeddings(path=args.input_path, l2_norm=args.normalize_lemma_embeddings,
                                                            return_first_embeddings_only=True)
    dict_lemma_key_embeddings_for_sref = load_basic_lemma_embeddings(path=args.input_path, l2_norm=True,
                                                            return_first_embeddings_only=True)
    n_dim = len(next(iter(dict_lemma_key_embeddings.values())))
    logging.info(f"done. embeddings dimension size: {n_dim}")

    # synset taxonomy for noun and verb
    logging.info(f"extracting synset taxonomy from WordNet...")
    dict_synset_taxonomy = extract_synset_taxonomy(target_part_of_speech=["n","v"], include_instance_of_lemmas=True)
    logging.info(f"done. number of synsets: {len(dict_synset_taxonomy)}")

    # precompute prior distribution params = NIW(\mu, \kappa, \Phi, \nu)
    # \mu = 0
    pi_vec_mu = np.zeros(shape=(n_dim,), dtype=np.float)
    # \kappa = args.kappa
    pi_kappa = args.kappa
    # \nu = \nu_{-DoF} + n_dim + 1
    pi_nu = args.nu_minus_dof + n_dim + 1
    # \Phi = \V * \nu_{-DoF}
    if args.cov > 0:
        # \V = diag(args.cov)
        vec_cov_diag = np.ones(shape=(n_dim,), dtype=np.float) * args.cov
    else:
        # \V = diag(COV[X])
        mat_x = np.stack(dict_lemma_key_embeddings.values())
        vec_cov_diag = np.mean( (mat_x - mat_x.mean(axis=0))**2, axis=0)
        scalar_cov = np.exp(np.mean(np.log(vec_cov_diag + 1E-15)))
        logging.info(f"empirical variance(geo. mean): {scalar_cov:1.5f}")
    vec_phi_diag = vec_cov_diag * args.nu_minus_dof
    scalar_phi_diag = np.exp(np.mean(np.log(vec_phi_diag + 1E-15)))
    logging.info(f"kappa = {pi_kappa:1.4f}, delta_nu = {args.nu_minus_dof:1.4f}, Phi(geo. mean) = {scalar_phi_diag:1.5f}")

    # compute synset prior distributions
    # [warning] it assigns .prior attribute for each node.
    ROOT_SYNSETS = ["entity.n.01"] # noun
    for node in dict_synset_taxonomy["verb_dummy_root.v.01"].children: # verb
        ROOT_SYNSETS.append(node.id)
    root_prior = NormalInverseWishart(vec_mu=pi_vec_mu, kappa=pi_kappa, vec_phi=vec_phi_diag, nu=pi_nu,
                                      posterior_inference_method=args.posterior_inference_method)
    for root_synset_id in ROOT_SYNSETS:
        logging.info(f"precompute synset priors. root: {root_synset_id}")
        root_node = dict_synset_taxonomy[root_synset_id]
        root_node.prior = root_prior
        update_children_priors(parent_node=root_node,
                               semantic_relation=args.semantic_relation,
                               basic_lemma_embeddings=dict_lemma_key_embeddings)

    # compute predictive posterior distributions
    logging.info(f"compute sense representation for all lemmas...")
    dict_lemma_sense_representations = {}; dict_synset_sense_representations = {}
    for synset in wn.all_synsets():
        pos = synset.pos()
        synset_id = synset.name()
        if pos in ["a","r","s"]:
            dict_lemma_reprs = compute_sense_representations_adverb_adjective(synset=synset,
                                                                              basic_lemma_embeddings=dict_lemma_key_embeddings_for_sref,
                                                                              variance=1.0)
        elif pos in ["n","v"]:
            p_synset_prior = dict_synset_taxonomy[synset_id].prior
            dict_lemma_reprs, synset_repr = compute_sense_representations_noun_verb(synset=synset,
                                                                                    semantic_relation=args.semantic_relation,
                                                                                    posterior_parameter_estimatnion=args.posterior_inference_parameter_estimation,
                                                                                    prior_distribution=p_synset_prior,
                                                                                    basic_lemma_embeddings=dict_lemma_key_embeddings,
                                                                                    inference_strategy=args.inference_strategy
                                                                                    )
            dict_synset_sense_representations[synset_id] = synset_repr

        dict_lemma_sense_representations.update(dict_lemma_reprs)

    logging.info(f"done. number of sense repr. #lemmas: {len(dict_lemma_sense_representations)}, #synsets: {len(dict_synset_sense_representations)}")

    # save as binary file (serialize)
    object = {"lemma":dict_lemma_sense_representations, "synset":dict_synset_sense_representations, "meta":vars(args)}
    path = args.out_path
    logging.info(f"sense repr. will be saved as: {path}")
    with io.open(path, mode="wb") as ofs:
        pickle.dump(object, ofs)

    logging.info(f"finished. good-bye.")