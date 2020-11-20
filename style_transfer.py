#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

import sentencepiece as spm
import fileinput

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders

from collections import namedtuple
from sacremoses import MosesTokenizer

Batch = namedtuple('Batch', 'ids src_tokens src_lengths lang_src_tokens lang_src_lengths')
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
            encode_fn(src_str[0]), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])

    lang_tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str[1]), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lang_lengths = torch.LongTensor([t.numel() for t in lang_tokens])

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_style_transfer(tokens, lengths, lang_tokens, lang_lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
            lang_src_tokens=batch['target'], lang_src_lengths=batch['target_lengths'],
        )


def main(args, inputs):
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

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    start_id = 0
    results = []
    for batch in make_batches(inputs, args, task, max_positions, encode_fn):

        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths

        lang_src_tokens = batch.lang_src_tokens
        lang_src_lengths = batch.lang_src_lengths

        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()
            lang_src_tokens = lang_src_tokens.cuda()
            lang_src_lengths = lang_src_lengths.cuda()

        sample = {
            'en_net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
            'lang_net_input': {
                'src_tokens': lang_src_tokens,
                'src_lengths': lang_src_lengths,
            },
        }
        translations = task.inference_step(generator, models, sample)
        for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
            src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
            lang_src_tokens_i = utils.strip_pad(lang_src_tokens[i], tgt_dict.pad())
            results.append((start_id + id, src_tokens_i, lang_src_tokens_i, hypos))

    # sort output to match input order
    for id, src_tokens, lang_src_tokens, hypos in sorted(results, key=lambda x: x[0]):
        if src_dict is not None:
            src_str = src_dict.string(src_tokens, args.remove_bpe)
            print('S-{}\t{}'.format(id, src_str))

            lang_src_str = src_dict.string(lang_src_tokens, args.remove_bpe)
            print('(style) S-{}\t{}'.format(id, lang_src_str))

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'],
                src_str=src_str,
                alignment=hypo['alignment'] if hypo['alignment'] is not None else None,
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            hypo_str = decode_fn(hypo_str)
            print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
            print('P-{}\t{}'.format(
                id,
                ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
            ))
            if args.print_alignment:
                print('A-{}\t{}'.format(
                    id,
                    ' '.join(map(lambda x: str(utils.item(x)), alignment))
                ))

    # update running id counter
    start_id += len(inputs)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)

    sp = spm.SentencePieceProcessor()
    sp.Load(args.sentencepiece)

    entok = MosesTokenizer(lang='en')

    f = open(args.style_transfer_file)
    lines = f.readlines()
    source = []
    style = []

    def process_line(l):
        if args.tokenize:
            tok = entok.tokenize(l, escape=False)
            l = " ".join(tok)
        l = l.strip().lower()
        l = sp.EncodeAsPieces(l)
        return " ".join(l)

    for i in lines:
        src, sty = i.split('\t')
        src = process_line(src)
        sty = process_line(sty)
        source.append(src)
        style.append(sty)

    main(args, list(zip(source, style)))


if __name__ == '__main__':
    cli_main()
