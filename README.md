# icd-classifier
Classification of international disease codes (ICD)

## Requirements
* Python 3.9.x

## To install the repo:

1. Clone the git repository from the source

1. create a new virtual environment and install
requirements.txt

    ```console
    $ python3 -m virtualenv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt
    ```

1. install the `icd_classifier` package
```pip install -m .```

1. to install PECOS with python version > 3.9, install from source: `pip install git+https://github.com/amzn/pecos.git`


## Repository Overview
```
  icd_classifier
    ├── data
    │   ├── build_vocab.py
    │   ├── concat_and_split.py    <-- splitting into train/dev/test
    │   ├── dataset.py             
    │   ├── data_utils.py          <-- Batch dataloader for DL training,
                                   lookup methods for ICD9 codes + descriptions
    │   ├── extract_wvs.py
    │   ├── get_discharge_summaries.py
    │   └── text_embeddings.py     <-- Word2Vec and Sentence BERT embeddings
    ├── data_preprocessing_and_embedding.ipynb  <-- notebook to preprocess data
    ├── modeling
    │   ├── concepts.py
    │   ├── dl_clf.py              <-- deep learning train-test script
    │   ├── models.py              <-- deep learning CNN, CAML, RNN models
    │   ├── evaluation.py          <-- metrics, evaluation
    │   ├── tools.py               <-- tools for saving models/metrics
    │   ├── traditional_clf.py     <-- LogReg, SVM
    │   └── xmc_clf.py             <-- Extreme Multilabel classification with Pecos
    └── settings.py               <-- global variables like data paths

``` 

# Datasets

1. Download MIMIC-III data files from https://physionet.org/works/MIMICIIIClinicalDatabase/files/
put `.csv.gz` files in `data/raw/` directory


1. Data preprocessing:
start jupyter notebook
```jupyter-notebook icd_classifier```
and execute `data_preprocessing_and_embedding.ipynb` notebook (cell-by-cell or all at once [15 min])
- cleaned up datasets will end up in `data/processed/` directory
- if not, move them manually. We write and load a lot of files as the dataset is being processed, train-dev split, embedded or vocabulary built.

# Running models
## Traditional classifiers
1. Logistic regression

```
python icd_classifier/modeling/traditional_clf.py --train_file data/processed/train_50.csv --test_file data/processed/dev_50.csv --vocab data/processed/vocab.csv --number_labels 50 --model log_reg --ngram 0
```
1. SVC linear
```
python icd_classifier/modeling/traditional_clf.py --train_file data/processed/train_50.csv --test_file data/processed/dev_50.csv --vocab data/processed/vocab.csv --model linear_svc --number_labels 50 --ngram 0
```

## Deep learning classifiers
- the following models and their training procedure is a re-implementation from Mullenbach 2018 paper [] and repo https://github.com/jamesmullenbach/caml-mimic
- might have to remove `--gpu` flag if no GPU available

1. CNN
```
python icd_classifier/modeling/dl_clf.py --data_path data/processed/train_50.csv --vocab data/processed/vocab.csv --number_labels 50 --model basic_cnn --n_epochs 100 --filter_size 4 --filter_maps 50 --dropout 0.2 --lr 0.003 --embeddings_file data/processed/processed_full.embed --early_stopping_metric prec_at_5 --gpu
```

1. CNN CAML
```
python icd_classifier/modeling/dl_clf.py --data_path data/processed/train_50.csv --vocab data/processed/vocab.csv --number_labels 50 --model caml --n_epochs 100 --filter_size 4 --filter_maps 50 --dropout 0.2 --lr 0.003 --embeddings_file data/processed/processed_full.embed --early_stopping_metric prec_at_5 --gpu
```
1. RNN
```
python icd_classifier/modeling/dl_clf.py --data_path data/processed/train_50.csv --vocab data/processed/vocab.csv --number_labels 50 --model rnn --n_epochs 100 --dropout 0 --lr 0.003 --rnn_dim 100 --rnn_cell_type gru --rnn_layers 1 --embeddings_file data/processed/processed_full.embed --early_stopping_metric prec_at_5 --rnn_bidirectional
```

## XR-Linear
Data preparation and XR-Linear and XR-Transformer models can be ran from `icd_classifier/modeling/xmc_clf.py`
```
python icd_classifier/modeling/xmc_clf.py --train_file data/processed/train_full.csv --test_file data/processed/dev_full.csv --number_labels full --topk 8 --label_feat_path data/processed/Z.emb.all.BioBERT-mnli-snli-scinli-scitail-mednli-stsb.npy --prepare_text_files
```


### Transformer embeddings
Must install `sentence-transformers` library from https://www.sbert.net/docs/installation.html
Then, use `icd_classifier.data.text_embeddings.sentence_embeddings()` method to encode label descriptions Z


Sentence emebddings tested:
https://huggingface.co/sentence-transformers/all-mpnet-base-v2
https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
https://huggingface.co/gsarti/biobert-nli


Pecos example embeddings:
for dataset embeddings, use https://github.com/amzn/pecos/tree/mainline/examples/xr-transformer-neurips21#getting-xr-transformer-embeddings

```
model_dir="pecos/encoders/wiki10-31k/roberta"
python3 -m pecos.xmc.xtransformer.encode --text-path pecos/mimic3/X.trn.txt --model-folder ${model_dir} --batch-gen-workers 16 --save-emb-path pecos/mimic3/X.emb.trn.npy --batch-size 128 --use-gpu True --verbose-level 3
```

label encoding
```
python3 -m pecos.xmc.xtransformer.encode --text-path pecos/mimic3/Z.all.txt --model-folder ${model_dir} --batch-gen-workers 16 --save-emb-path pecos/mimic3/Z.emb.all.npy --batch-size 128 --use-gpu True --verbose-level 3
```

