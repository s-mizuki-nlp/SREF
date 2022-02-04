
from typing import Optional

import numpy as np
import bert_tokenization
from bert_serving.client import BertClient


BERT_BASE_DIR = 'data/bert/cased_L-24_H-1024_A-16/'
tokenizer = bert_tokenization.FullTokenizer(vocab_file=BERT_BASE_DIR+'vocab.txt',
                                            do_lower_case=False)


class BertEncoder():

    def __init__(self, host: str, port: Optional[int] = None):
        port = 5555 if port is None else port
        self.bert_client = BertClient(ip=host, port=port)

    def bert_embed(self, sents, merge_strategy: str, merge_subtokens=True, apply_sum_pooling: bool = False):
        sents_encodings_full = self.bert_client.encode(sents)
        sents_tokenized = [tokenizer.tokenize(s) for s in sents]

        sents_encodings = []
        for sent_tokens, sent_vecs in zip(sents_tokenized, sents_encodings_full):
            sent_encodings = []
            sent_vecs = sent_vecs[1:-1]  # ignoring [CLS] and [SEP]
            for token, vec in zip(sent_tokens, sent_vecs):
                layers_vecs = np.split(vec, 4)  # due to -pooling_layer -4 -3 -2 -1
                layers_repr = np.array(layers_vecs, dtype=np.float32)
                if apply_sum_pooling:
                    layers_repr = layers_repr.sum(axis=0)
                sent_encodings.append((token, layers_repr))
            sents_encodings.append(sent_encodings)

        if merge_subtokens:
            sents_encodings_merged = []
            for sent, sent_encodings in zip(sents, sents_encodings):

                sent_tokens_vecs = []
                for token in sent.split():  # these are preprocessed tokens

                    token_vecs = []
                    for subtoken in tokenizer.tokenize(token):
                        if len(sent_encodings) == 0:  # sent may be longer than max_seq_len
                            # print('ERROR: seq too long ?')
                            break

                        encoded_token, encoded_vec_or_mat = sent_encodings.pop(0)
                        assert subtoken == encoded_token
                        token_vecs.append(encoded_vec_or_mat)

                    if len(token_vecs) == 0:
                        if apply_sum_pooling:
                            token_vec = np.zeros(1024)
                        else:
                            token_vec = np.zeros((4, 1024))
                    elif merge_strategy == 'first':
                        token_vec = np.array(token_vecs[0])
                    elif merge_strategy == 'sum':
                        token_vec = np.array(token_vecs).sum(axis=0)
                    elif merge_strategy == 'mean':
                        token_vec = np.array(token_vecs).mean(axis=0)

                    sent_tokens_vecs.append((token, token_vec))

                sents_encodings_merged.append(sent_tokens_vecs)

            sent_encodings = sents_encodings_merged

        return sent_encodings


    def bert_embed_sents(self, sents):
        sents_encodings_full = self.bert_client.encode(sents)
        sents_encodings = []
        for sent, sent_vec in zip(sents, sents_encodings_full):
            #sent_vec = np.mean(sent_vec, 0)
            layers_vecs = np.split(sent_vec, 4)  # due to -pooling_layer -4 -3 -2 -1
            layers_sum = np.array(layers_vecs).sum(axis=0)
            sents_encodings.append((sent, layers_sum))
        return sents_encodings