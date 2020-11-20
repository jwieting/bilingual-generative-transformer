import sentencepiece as spm
import os
import io
import numpy as np
import logging

from sacremoses import MosesTokenizer
from scipy.stats import spearmanr, pearsonr

#python -u evaluate.py training/fr/data-joint-bin/ -s en -t fr --path checkpoints/bgt5-fr-65536-25-0-sample-0.5-1.0/checkpoint20.pt  --sentencepiece training/fr/fr-en.1m.tok.all.sp.20k.model --cpu --model-overrides "{'cpu': 1}"

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self):
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

    def run(self, params, batcher):
        results = {}
        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
            lang_1 = "en"
            lang_2 = "en"
            if dataset == "STS.input.track1.ar-ar.txt":
                lang_1 = "ar"
                lang_2 = "ar"
            elif dataset == "STS.input.track2.ar-en.txt":
                lang_1 = "en"
                lang_2 = "ar"
            elif dataset == "STS.input.track3.es-es.txt":
                lang_1 = "es"
                lang_2 = "es"
            elif dataset == "STS.input.track4a.es-en.txt":
                lang_1 = "es"
                lang_2 = "en"
            elif dataset == "STS.input.track6.tr-en.txt":
                lang_1 = "en"
                lang_2 = "tr"
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1, lang_1)
                    enc2 = batcher(params, batch2, lang_2)

                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)

            results[self.name + "." + dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores)}
            logging.debug('%s : pearson = %.4f, spearman = %.4f' %
                          (dataset, results[self.name + "." + dataset]['pearson'][0],
                           results[self.name + "." + dataset]['spearman'][0]))

        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)

        results[self.name + "." + 'all'] = {'pearson': {'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'mean': avg_spearman,
                                       'wmean': wavg_spearman}}
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        return results


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)
        self.name = "STS12"


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN', 'SMT']
        self.loadFile(taskpath)
        self.name = "STS13"

class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)
        self.name = "STS14"

class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)
        self.name = "STS15"

class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)
        self.name = "STS16"

class STSBenchmarkEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.data = {}
        self.data['dev'] = dev
        self.data['test'] = test
        self.datasets = ["dev", "test"]
        self.name = "Benchmark"

    def loadFile(self, fpath):
        gs_scores = []
        sent1 = []
        sent2 = []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sent1.append(text[5].split())
                sent2.append(text[6].split())
                gs_scores.append(float(text[4]))

        sorted_data = sorted(zip(sent1, sent2, gs_scores),
                    key=lambda z: (len(z[0]), len(z[1]), z[2]))

        sent1, sent2, gs_scores = map(list, zip(*sorted_data))

        return sent1, sent2, gs_scores

class STSHard(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSHard*****\n\n')
        self.seed = seed
        hard_pos = self.loadFile(os.path.join(task_path, 'hard-pos.txt'))
        hard_neg = self.loadFile(os.path.join(task_path, 'hard-neg.txt'))
        self.data = {}
        self.data['hard-pos'] = hard_pos
        self.data['hard-neg'] = hard_neg
        self.datasets = ["hard-pos", "hard-neg"]
        self.name = "Hard"

    def loadFile(self, fpath):
        gs_scores = []
        sent1 = []
        sent2 = []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sent1.append(text[0].split())
                sent2.append(text[1].split())
                gs_scores.append(float(text[2]))

        sorted_data = sorted(zip(sent1, sent2, gs_scores),
                    key=lambda z: (len(z[0]), len(z[1]), z[2]))

        sent1, sent2, gs_scores = map(list, zip(*sorted_data))

        return sent1, sent2, gs_scores

class SemEval17(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SemEval17*****\n\n')
        self.seed = seed
        self.data = {}
        self.datasets = ["STS.input.track1.ar-ar.txt",
                         "STS.input.track2.ar-en.txt",
                         "STS.input.track3.es-es.txt",
                         "STS.input.track4a.es-en.txt",
                         "STS.input.track5.en-en.txt",
                         "STS.input.track6.tr-en.txt"]

        for i in self.datasets:
            self.data[i] = self.loadFile(os.path.join(task_path, i))

        self.name = "SemEval17"

    def loadFile(self, fpath):
        gs_scores = []
        sent1 = []
        sent2 = []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                if len(text) != 3:
                    continue
                sent1.append(text[0].split())
                sent2.append(text[1].split())
                gs_scores.append(float(text[2]))

        sorted_data = sorted(zip(sent1, sent2, gs_scores),
                    key=lambda z: (len(z[0]), len(z[1]), z[2]))

        sent1, sent2, gs_scores = map(list, zip(*sorted_data))

        return sent1, sent2, gs_scores

def batcher(params, batch, lang="en"):
    batch = [" ".join(s) for s in batch]
    new_batch = []
    for p in batch:
        if params.tokenize:
            tok = params.entok.tokenize(p, escape=False)
            p = " ".join(tok)
        p = p.lower()
        p = params.sp.EncodeAsPieces(p)
        p = " ".join(p)
        new_batch.append(p)
    vecs = params.embedder.embed(new_batch, params.encoder)
    return vecs

def evaluate(embedder, args):

    sp = spm.SentencePieceProcessor()
    sp.Load(args.sentencepiece)

    entok = MosesTokenizer(lang='en')

    from argparse import Namespace

    args = Namespace(batch_size=32, entok=entok, sp=sp, embedder=embedder,
                     encoder=args.eval_encoder, tokenize=args.tokenize)

    s = STS12Eval('STS/STS12-en-test')
    s.do_prepare()
    results = s.run(args, batcher)
    s = STS13Eval('STS/STS13-en-test')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = STS14Eval('STS/STS14-en-test')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = STS15Eval('STS/STS15-en-test')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = STS16Eval('STS/STS16-en-test')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = SemEval17('STS/STS17-test')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = STSBenchmarkEval('STS/STSBenchmark')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = STSHard('STS/STSHard')
    s.do_prepare()
    results.update(s.run(args, batcher))

    for i in results:
        print(i, results[i])

    total = []
    all = []
    cross = []
    foreign = []
    for i in results:
        if "STS" in i and "all" not in i and "SemEval17" not in i:
            total.append(results[i]["pearson"][0])
        if "STS" in i and "all" in i:
            all.append(results[i]["pearson"]["mean"])
        if i == "SemEval17.STS.input.track2.ar-en.txt" or i == "SemEval17.STS.input.track4a.es-en.txt" \
                or i == "SemEval17.STS.input.track6.tr-en.txt":
            cross.append(results[i]["pearson"][0])
        if i == "SemEval17.STS.input.track1.ar-ar.txt" or i == "SemEval17.STS.input.track3.es-es.txt":
            foreign.append(results[i]["pearson"][0])

    print("Average (cross): {0}".format(np.mean(cross)))
    print("Average (foreign): {0}".format(np.mean(foreign)))
    print("Average (datasets): {0}".format(np.mean(total)))
    print("Average (comps): {0}".format(np.mean(all)), flush=True)
    return np.mean(all)

if __name__ == '__main__':

    from embed import Embedder
    from fairseq import options

    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)

    embedder = Embedder(args)

    evaluate(embedder, args)
