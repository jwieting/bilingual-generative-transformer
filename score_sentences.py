import sys
import sentencepiece as spm
import numpy as np

from embed import Embedder
from fairseq import options
from sacremoses import MosesTokenizer

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class SentencePairScorer(object):

    def __init__(self):
        parser = options.get_generation_parser(interactive=True)
        self.args = options.parse_args_and_arch(parser)

        self.embedder = Embedder(self.args)
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

    def score_sentences(self, pairs):

        sp = spm.SentencePieceProcessor()
        sp.Load(self.args.sentencepiece)

        entok = MosesTokenizer(lang='en')

        def process_example(i):
            tok = entok.tokenize(i, escape=False)
            p = " ".join(tok).lower()
            p = sp.EncodeAsPieces(p)
            p = " ".join(p)
            return p

        embeddings_1 = []
        embeddings_2 = []
        sentences_1 = []
        sentences_2 = []
        for i in pairs:
            p1, p2 = process_example(i[0]), process_example(i[1])
            sentences_1.append(p1)
            sentences_2.append(p2)

            if len(sentences_1) == 32:
                vecs = self.embedder.embed(sentences_1, self.args.eval_encoder)
                embeddings_1.append(vecs)

                vecs = self.embedder.embed(sentences_2, self.args.eval_encoder)
                embeddings_2.append(vecs)

                sentences_1 = []
                sentences_2 = []

        if len(sentences_1) > 0:
            vecs = self.embedder.embed(sentences_1, self.args.eval_encoder)
            embeddings_1.append(vecs)

            vecs = self.embedder.embed(sentences_2, self.args.eval_encoder)
            embeddings_2.append(vecs)

        embeddings_1 = np.vstack(embeddings_1)
        embeddings_2 = np.vstack(embeddings_2)

        scores = []
        for i in range(embeddings_1.shape[0]):
            s = self.similarity(embeddings_1[i], embeddings_2[i])
            scores.append(s)

        return scores

s = SentencePairScorer()
scores = s.score_sentences([("blah", "blah")])
print(scores)