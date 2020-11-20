# bilingual-generative-transformer

Code to train models from "A Bilingual Generative Transformer for Semantic Sentence Embedding". Our code is based on the 58e43cb3ff18f1f47fd62926f00c70cb5920a66f commit from Fairseq https://github.com/pytorch/fairseq from Facebook AI Research.

To get started, follow the installation and setup instructions below.

If you use our code for your work please cite:

    @article{wieting2019bilingual,
        title={A Bilingual Generative Transformer for Semantic Sentence Embedding},
        author={Wieting, John and Neubig, Graham and Berg-Kirkpatrick, Taylor},
        booktitle={Proceedings of the Empirical Methods in Natural Language Processing},
        url={https://arxiv.org/abs/1911.03895},
        year={2019}
    }

Installation and setup instructions:

1. Clone the repository and install the code:

        git clone https://github.com/jwieting/bilingual-generative-transformer.git
        cd bilingual-generative-transformer
        pip install --editable .

2. Download the data files, including training data, and saved models from http://www.cs.cmu.edu/~jwieting:

        wget http://www.cs.cmu.edu/~jwieting/bgt.zip .
        unzip bgt.zip
        rm bgt.zip
        
3. Download the STS evaluation data:

        wget http://www.cs.cmu.edu/~jwieting/STS.zip .
        unzip STS.zip
        rm STS.zip

To train the (Bilingual Generative Transformer) BGT model (on French OpenSubtitles 2018 and Gigaword data) other choices include (OpenSubtitles 2018 ar, es, fr, ja, and tr):

    python -u train.py bgt/fr-os-giga/data-joint-bin -a bgt-emnlp --bgt-setting bgt --optimizer adam --lr 0.0005 -s en -t fr \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 500 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion bgt_loss --max-epoch 20 --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9, 0.98)' \
    --save-dir checkpoints/bgt --distributed-world-size 1 --latent-size 1024 --update-freq 50 --task bgt \
    --save-interval-updates 0 --sentencepiece bgt/fr-os-giga/fr-en.1m.sp.20k.model --x0 65536 --translation-loss 1.0 \
    --sentence-avg --tokenize 1 --num-workers 0 --find-unused-parameters

To train the translation (Trans) baseline model (on French OpenSubtitles 2018 and Gigaword data) other choices include (OpenSubtitles 2018 ar, es, fr, ja, and tr):

    python -u train.py bgt/fr-os-giga/data-joint-bin -a bgt-emnlp --bgt-setting trans --optimizer adam --lr 0.0005 -s en -t fr \
    --label-smoothing 0.1 --dropout 0.3 --max-tokens 500 --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --criterion bilingual_label_smoothed_cross_entropy --max-epoch 20 --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --adam-betas '(0.9, 0.98)' --save-dir checkpoints/trans --distributed-world-size 1 --latent-size 1024 --update-freq 50 \
    --task bgt --save-interval-updates 0 --sentencepiece bgt/fr-os-giga/fr-en.1m.sp.20k.model --sentence-avg --tokenize 1\
    --num-workers 0 --find-unused-parameters

To evaluate a model on the STS tasks:

    python -u evaluate.py bgt/fr-os-giga/data-joint-bin -s en -t fr --path bgt/checkpoints/bgt/checkpoint_best.pt \
    --tokenize 1 --sentencepiece bgt/fr-os-giga/fr-en.1m.sp.20k.model

To score a list of sentence pairs in tab-separated (tsv) format:

    python -u evaluate_list.py bgt/fr-os-giga/data-joint-bin -s en -t fr --path bgt/checkpoints/bgt/checkpoint_best.pt \
    --sentencepiece bgt/fr-os-giga/fr-en.1m.sp.20k.model --tokenize 1 --sim-file bgt/sentences.txt

To generate outputs following our "style-transfer" setting:

    python -u style_transfer.py bgt/fr-os-giga/data-joint-bin -s en -t fr --path bgt/checkpoints/bgt/checkpoint_best.pt \
    --sentencepiece bgt/fr-os-giga/fr-en.1m.sp.20k.model --tokenize 1 --task bgt --remove-bpe sentencepiece \
    --style-transfer-file bgt/style_transfer.txt
