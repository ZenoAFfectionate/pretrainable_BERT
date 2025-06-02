# BERT Model Reimplementation with PyTorch

This project is a PyTorch-based reimplementation of the BERT (Bidirectional Encoder Representations from Transformers) model. It supports pretraining using two key tasks: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).

> [NAACL-2018] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
> Paper URL : https://arxiv.org/abs/1810.04805


## Introduction

The BERT (Bidirectional Encoder Representations from Transformers) paper, introduced by Jacob Devlin et al. in 2018, presents a novel approach to pretraining language models that significantly improves performance on a wide range of natural language processing (NLP) tasks.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

The introduction of BERT marked a significant shift in NLP research, leading to the development of many variations (such as RoBERTa, DistilBERT, and ALBERT). Its ability to handle a wide range of language tasks with minimal architecture adjustments has made it a go-to model for many language understanding applications.

To have a better understanding of this classical model, we decide to implement it from strach based on some previous projects. 

Currently this project is almost done, the code is almost fully verified now!
But we are still putting our hands on fine-tuning the BERT to reproduce the full results of the original paper.


## Features

### Pretraining Task
* Masked Language Modeling (MLM): The model randomly masks some tokens in the input and trains to predict these masked tokens.
* Next Sentence Prediction (NSP): The model predicts if two given sentences follow each other in the original text.

### Training Visualization
* The model outputs the MLM loss and NSP loss during training.
* It also displays the average accuracy for both MLM and NSP tasks at each epoch.

### Wikitext-2 Demo
* The project includes a demonstration on the Wikitext-2 dataset for pretraining the BERT model.
* You can run the pretraining process directly on the Wikitext-2 dataset to observe the training pipeline.

### Extensible for Larger Datasets:
* This BERT implementation can be adapted to larger pretraining datasets.
* Ensure the dataset follows the required format (randomly split each sentence into two parts using \t) and build a corresponding vocabulary file for the dataset (using dataset/vocab.py).


## Quickstart

**NOTICE : Your corpus should be prepared with two sentences in one line with tab(\t) separator**

Example of the corpus is as follow:
```
We live in a \t twilight word.\n
There is \t no friend at dust.\n
```

### 1. Clone the repository:
```shell
git clone https://github.com/ZenoAFfectionate/pretrain_BERT
cd pretrain_BERT
```

### 2. Install the required dependencies:
```shell
pip install -r requirements.txt
```

### 3. Building vocab based on your corpus
**NOTICE : the wikitext-2 data has been preprocessed, please see `preprocess.py` for more details.**
```shell
cd dataset
python vocab.py -c ./wikitext-2/data.txt -o ./wikitext-2/vocabulary -m 5
```
The parameter '-m' is used to set the minimum frequency threshold for words to be included in the vocabulary. Words that appear in the dataset fewer times than this value will be excluded.


### 4. Train your own BERT model
```shell
cd ..
python pretrain.py -c ./dataset/wikitext-2/train.txt -t ./dataset/wikitext-2/test.txt -v ./dataset/wikitext-2/vocabulary -o ./checkpoint/wikitext-2

```
Remember to adjust the parameter of the model if needed!

## Language Model Pre-training

In the paper, authors shows the new language model training methods, 
which are "masked language model" and "predict next sentence".


### Masked Language Model 

> Original Paper : 3.3.1 Task #1: Masked LM 

#### Rules:
Randomly 15% of input token will be changed into something, based on under sub-rules

```
Input  Sequence : The man went to [MASK] store with [MASK] dog
Target Sequence :                  the                his
```

#### Rules:
Randomly 15% of input token will be changed into something, based on under sub-rules

1. Randomly 80% of tokens, gonna be a `[MASK]` token (special token)
2. Randomly 10% of tokens, gonna be a `[RAND]` token (another word)
3. Randomly 10% of tokens, will be remain as same. But need to be predicted.

### Predict Next Sentence

> Original Paper : 3.3.2 Task #2: Next Sentence Prediction

```
Input : [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
Label : Is Next

Input = [CLS] the man heading to the store [SEP] penguin [MASK] are flight ##less birds [SEP]
Label = Not Next
```

#### Rules:

1. Randomly 50% of next sentence, gonna be continuous sentence.
2. Randomly 50% of next sentence, gonna be unrelated sentence.


## Author
Original: Junseong Kim, Scatter Lab (codertimo@gmail.com / junseong.kim@scatterlab.co.kr)

Modifier: Zeno Pang(SCUT), Leeyyi(SYSU)

## Reference

[1] [The Annotated Trasnformer](https://github.com/harvardnlp/annotated-transformer)

[2] [GitHub: BERT PyTorch Implementation](https://github.com/codertimo/BERT-pytorch)

[3] [Building BERT Model from Sratch](https://medium.com/data-and-beyond/complete-guide-to-building-bert-model-from-sratch-3e6562228891)

## License
This project following MIT License as written in LICENSE file

## Copyright
Copyright 2018 Junseong Kim : [GitHub: BERT PyTorch Implementation](https://github.com/codertimo/BERT-pytorch)

Copyright 2018 Alexander Rush : [The Annotated Trasnformer](https://github.com/harvardnlp/annotated-transformer)
