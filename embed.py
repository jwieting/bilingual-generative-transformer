#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput

import torch
import numpy as np

from fairseq import checkpoint_utils, options, tasks, utils

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )

class Embedder():

    def __init__(self, args, model=None, task=None):

        if model is None:
            utils.import_user_module(args)

            if args.buffer_size < 1:
                args.buffer_size = 1
            if args.max_tokens is None and args.max_sentences is None:
                args.max_sentences = 1

            assert not args.sampling or args.nbest == args.beam, \
                '--sampling requires --nbest to be equal to --beam'
            assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
                '--max-sentences/--batch-size cannot be larger than --buffer-size'

            print(args)

            self.use_cuda = torch.cuda.is_available() and not args.cpu

            # Setup task, e.g., translation
            self.task = tasks.setup_task(args)

            # Load ensemble
            print('| loading model(s) from {}'.format(args.path))
            models, _model_args = checkpoint_utils.load_model_ensemble(
                args.path.split(':'),
                arg_overrides=eval(args.model_overrides),
                task=self.task,
            )

            self.model = models[0]

            if self.use_cuda:
                self.model.cuda()

            self.max_positions = utils.resolve_max_positions(
                self.task.max_positions(),
                *[model.max_positions() for model in models]
            )

            self.args = args
        else:
            self.args = args
            self.model = model
            self.task = task
            self.max_positions = task.max_positions()
            self.use_cuda = torch.cuda.is_available() and not args.cpu

    def embed(self, inputs, encoder):
        self.model.eval()

        encoder = getattr(self.model, encoder)

        results = []
        for batch in make_batches(inputs, self.args, self.task, self.max_positions, lambda x: x):
            if self.use_cuda:
                toks, lens = batch[1].cuda(), batch[2].cuda()
            else:
                toks, lens = batch[1], batch[2]
            vecs = encoder(toks, lens, generate=False)
            results.append((batch.ids, vecs['mean'].detach().cpu().numpy()))

        vecs = np.vstack([i[1] for i in results])
        ids = np.hstack([i[0] for i in results])
        vecs = vecs[np.argsort(ids)]

        self.model.train()

        return vecs

def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)

    embed = Embedder(args)

    data = ['I have a dog.', 'How are you?', 'What!', 'I want something to eat.', 'How are youuuu?']
    vecs = embed.embed(data, 'encoder_sem')
    print(vecs)

    return vecs

if __name__ == '__main__':
    cli_main()
