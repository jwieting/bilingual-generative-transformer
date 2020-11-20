# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)

from fairseq.modules import LayerNorm, MultiheadAttention

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('bgt')
class BGTModel(BaseFairseqModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self,  args, encoder_sem=None, encoder_en=None, encoder_fr=None,
                 decoder_fr=None, decoder_en=None, decoder_trans_en=None, decoder_trans_fr=None,
                 src_dict=None, tgt_dict=None):
        super().__init__()
        self.args = args
        self.bgt_setting = self.args.bgt_setting

        self.encoder_sem = encoder_sem
        self.encoder_en = encoder_en
        self.encoder_fr = encoder_fr

        self.decoder_fr = decoder_fr
        self.decoder_en = decoder_en

        self.decoder_trans_en = decoder_trans_en
        self.decoder_trans_fr = decoder_trans_fr

        self.epoch_iter = None

        self.num_updates = 0
        self.counter = 0

        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    def max_positions(self):
        """Maximum length supported by the model."""

        if self.encoder_sem is not None:
            mx = self.encoder_sem.max_positions()
        else:
            mx = self.encoder_sem_fr.max_positions()

        if self.decoder_en is not None:
            return (mx, self.decoder_en.max_positions())
        else:
            return (mx, self.decoder_trans_en.max_positions())

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--bgt-setting', type=str, choices=["trans", "bgt"], metavar='STR',
                            help='which experiment to run, choices are: '
                            'trans - translation baseline'
                            'bgt - bgt model')
        parser.add_argument('--latent-size', type=int, default=0,metavar='D',
                            help='size of latent embeddings')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        def get_decoder_embed_tokens():
            return build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        def get_encoder_embed_tokens():
            return build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )

        assert len(src_dict.symbols) == len(tgt_dict.symbols)
        encoder_sem = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        encoder_en = cls.build_encoder(args, src_dict, get_encoder_embed_tokens())
        encoder_fr = cls.build_encoder(args, tgt_dict, get_encoder_embed_tokens())
        decoder_trans_fr = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, do_trans=True)
        decoder_trans_en = cls.build_decoder(args, src_dict, get_decoder_embed_tokens(), do_trans=True)
        decoder_fr = None
        decoder_en = None

        if args.bgt_setting == "bgt":
            decoder_fr = cls.build_decoder(args, tgt_dict, get_decoder_embed_tokens(), do_trans=False)
            decoder_en = cls.build_decoder(args, src_dict, get_decoder_embed_tokens(), do_trans=False)

        return BGTModel(args, encoder_sem=encoder_sem, encoder_en=encoder_en, encoder_fr=encoder_fr,
                        decoder_fr=decoder_fr, decoder_en=decoder_en, decoder_trans_en=decoder_trans_en,
                        decoder_trans_fr=decoder_trans_fr, src_dict=src_dict, tgt_dict=tgt_dict)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, do_trans=True):
        return TransformerDecoder(args, tgt_dict, embed_tokens, do_trans=do_trans)

    def forward(self, sample):

        en_src_tokens = sample['en_net_input']['src_tokens']
        en_src_lengths = sample['en_net_input']['src_lengths']
        en_prev_output_tokens = sample['en_net_input']['prev_output_tokens']

        fr_src_tokens = sample['fr_net_input']['src_tokens']
        fr_src_lengths = sample['fr_net_input']['src_lengths']
        fr_prev_output_tokens = sample['fr_net_input']['prev_output_tokens']

        sem_encoder_out_en = self.encoder_sem(en_src_tokens, en_src_lengths)
        sem_encoder_out_fr = self.encoder_sem(fr_src_tokens, fr_src_lengths)

        if self.counter % 2 == 0:
            sem_encoder_out = sem_encoder_out_en
        else:
            sem_encoder_out = sem_encoder_out_fr

        self.counter += 1

        decoder_out = {}

        if self.bgt_setting == "bgt":
            en_encoder_out = self.encoder_en(en_src_tokens, en_src_lengths)
            fr_encoder_out = self.encoder_fr(fr_src_tokens, fr_src_lengths)

            # get english decoder sentence embedding
            en_z = torch.cat((en_encoder_out['mean'], sem_encoder_out['mean']), dim=1)
            # get french decoder sentence embedding
            fr_z = torch.cat((fr_encoder_out['mean'], sem_encoder_out['mean']), dim=1)

            # get total z, mean, logv
            z = torch.cat((sem_encoder_out['z'], en_encoder_out['z'], fr_encoder_out['z']), dim=1)
            logv = torch.cat((sem_encoder_out['logv'], en_encoder_out['logv'], fr_encoder_out['logv']), dim=1)
            lv_logv = torch.cat((en_encoder_out['logv'], fr_encoder_out['logv']), dim=1)

            mean = torch.cat((sem_encoder_out['mean'], en_encoder_out['mean'], fr_encoder_out['mean']), dim=1)
            lv_mean = torch.cat((en_encoder_out['mean'], fr_encoder_out['mean']), dim=1)

            en_decoder_out = self.decoder_en(fr_prev_output_tokens, {'sent_emb': en_z,
                                                                     'encoder_out': None, 'encoder_padding_mask': None})
            fr_decoder_out = self.decoder_fr(en_prev_output_tokens, {'sent_emb': fr_z,
                                                                     'encoder_out': None, 'encoder_padding_mask': None})

            en_lv_logits = en_decoder_out[0] #use french target
            fr_lv_logits = fr_decoder_out[0] #use english target

            decoder_out['z'] = z
            decoder_out['logv'] = logv
            decoder_out['lv_logv'] = lv_logv
            decoder_out['mean'] = mean
            decoder_out['lv_mean'] = lv_mean

            decoder_out['en_lv_logits'] = en_lv_logits
            decoder_out['fr_lv_logits'] = fr_lv_logits
            decoder_out['sem_en'] = sem_encoder_out_en['mean']
            decoder_out['sem_fr'] = sem_encoder_out_fr['mean']

        #translation
        sent_emb = sem_encoder_out_en['mean']
        trans_decoder_out = self.decoder_trans_en(en_prev_output_tokens, {'sent_emb': sent_emb,
                                                                     'encoder_out': None, 'encoder_padding_mask': None})
        fr_trans_logits = trans_decoder_out[0]  # use english target
        decoder_out['fr_trans_logits'] = fr_trans_logits

        sent_emb = sem_encoder_out_fr['mean']
        trans_decoder_out = self.decoder_trans_fr(fr_prev_output_tokens, {'sent_emb': sent_emb,
                                                                                      'encoder_out': None,
                                                                                      'encoder_padding_mask': None})
        en_trans_logits = trans_decoder_out[0]  # use fremch target
        decoder_out['en_trans_logits'] = en_trans_logits

        return decoder_out

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1)
        else:
            return utils.softmax(logits, dim=-1)

class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))
        self.args = args
        self.dropout = args.dropout
        self.bgt_setting = self.args.bgt_setting

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.hidden2mean = nn.Linear(embed_dim, self.args.latent_size, bias=False)

        if self.bgt_setting == "bgt":
            self.hidden2logv = nn.Linear(embed_dim, self.args.latent_size, bias=False)
            self.latent2hidden = nn.Linear(self.args.latent_size, embed_dim, bias=False)

    def forward(self, src_tokens, src_lengths, generate=False):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        # if not encoder_padding_mask.any():
        #    encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        #sample z
        z = None
        if self.bgt_setting == "bgt" and not generate:
            z = torch.randn([x.size()[1], self.args.latent_size])

        sent_emb, mean, logv = self.get_sentence_embs(x, encoder_padding_mask, z)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T,
            'sent_emb': sent_emb,
            'mean': mean,
            'logv': logv,
            'z': z,
        }

    def get_sentence_embs(self, encoder_out, encoder_padding_mask, z=None):

        if not self.args.cpu:
            mean_pool = torch.where(
                encoder_padding_mask.unsqueeze(2).cuda(),
                torch.Tensor([float(0)]).cuda(),
                encoder_out.transpose(1, 0).float()
            ).type_as(encoder_out)
        else:
            mean_pool = torch.where(
                encoder_padding_mask.unsqueeze(2),
                torch.Tensor([float(0)]),
                encoder_out.transpose(1, 0).float()
            ).type_as(encoder_out)

        den = encoder_padding_mask.size()[1] - encoder_padding_mask.sum(dim=1)
        mean_pool =  mean_pool.sum(dim=1) / den.float().unsqueeze(1)

        mean = self.hidden2mean(mean_pool)
        logv = None
        if self.bgt_setting == "bgt":
            logv = self.hidden2logv(mean_pool)
            if z is not None:
                std = torch.exp(0.5 * logv)
                if not self.args.cpu:
                    z = z.cuda()
                z = z * std + mean
                sent_emb = self.latent2hidden(z)
            else:
                sent_emb = self.latent2hidden(mean)
        else:
            sent_emb = mean

        return sent_emb, mean, logv

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if encoder_out['sent_emb'] is not None:
            encoder_out['sent_emb'] = \
                encoder_out['sent_emb'].index_select(0, new_order)
        if encoder_out['mean'] is not None:
            encoder_out['mean'] = \
                encoder_out['mean'].index_select(0, new_order)
        if encoder_out['logv'] is not None:
            encoder_out['logv'] = \
                encoder_out['logv'].index_select(0, new_order)
        if encoder_out['z'] is not None:
            encoder_out['z'] = \
                encoder_out['z'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, do_trans=True):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.args = args
        self.do_trans = do_trans
        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerSentenceEmbeddingDecoderLayer(args, no_encoder_attn, do_trans=do_trans)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            if self.do_trans:
                self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), embed_dim + self.args.latent_size))
            else:
                self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.args.latent_size * 2 + embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state)
        x = self.output_layer(x, encoder_out['sent_emb'])
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        sent_emb = encoder_out['sent_emb']

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x, sent_emb,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, sent_emb, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                sent_emb = sent_emb.unsqueeze(1).expand((features.size()[0], features.size()[1], sent_emb.size()[1]))
                features = torch.cat((sent_emb, features), dim=2)
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict

class TransformerSentenceEmbeddingDecoderLayer(TransformerDecoderLayer):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, add_bias_kv=False, add_zero_attn=False, do_trans=True):
        super().__init__(args)
        self.embed_dim = args.decoder_embed_dim
        self.args = args
        self.do_trans = do_trans
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

        if do_trans:
            self.decoder_fc1 = Linear(self.embed_dim + self.args.latent_size, self.embed_dim)
        else:
            self.decoder_fc1 = Linear(self.embed_dim + self.args.latent_size * 2, self.embed_dim)

    def forward(
        self,
        x,
        sent_emb = None,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        prev_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
        size = (x.size()[0], x.size()[1], sent_emb.size()[-1])

        concat_sent_emb = torch.cat((x, sent_emb.expand(size)), dim=2)
        x = self.decoder_fc1(concat_sent_emb)
        F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state["prev_key"], saved_state["prev_value"]
            return x, attn, self_attn_state
        return x, attn

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('bgt', 'base')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

@register_model_architecture('bgt', 'bgt-emnlp')
def bgt(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)

