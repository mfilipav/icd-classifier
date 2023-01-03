# icd-classifier
Classification of international disease codes (ICD)

## Requirements
* Python 3.10.x

# To run the project:

1. Clone the git repository from the source

1. create a new virtual environment and install
requirements.txt

    ```console
    $ python3 -m virtualenv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt
    ```

1. install the package
```pip install -m .```


1. Download MIMIC-III data files from https://physionet.org/works/MIMICIIIClinicalDatabase/files/
put `.csv.gz` files in `data/raw/` directory


1. Data preprocessing:
start jupyter notebook
```jupyter-notebook icd_classifier```
and execute `data_preprocessing_and_embedding.ipynb` notebook (cell-by-cell or all at once [15 min])


1. Running models

Logistic regression
- one-vs-one (not one-vs-all)
- with max pooling vs averaging strategies

```
# much better results
python icd_classifier/modeling/traditional_clf.py --train_file data/processed/train_50.csv --dev_file data/processed/dev_50.csv --vocab data/processed/vocab.csv --Y 50 --model log_reg --ngram 0

# TODO: resolve issues with this implementation
python icd_classifier/modeling/train.py --data_path data/processed/train_50.csv --vocab data/processed/vocab.csv --Y 50 --model log_reg --n_epochs 100 --pool avg --batch_size 16 --lr 0.003 --embeddings_file data/processed/processed_full.embed --gpu
```

SVC linear
```
python icd_classifier/modeling/traditional_clf.py --train_file data/processed/train_50.csv --dev_file data/processed/dev_50.csv --vocab data/processed/vocab.csv --model linear_svc --Y 50 --ngram 0
```

CNN
- basic
- no attention mechanism yet
```
python icd_classifier/modeling/train.py --data_path data/processed/train_50.csv --vocab data/processed/vocab.csv --Y 50 --model basic_cnn --n_epochs 100 --filter_size 4 --filter_maps 500 --dropout 0.2 --lr 0.003 --embeddings_file data/processed/processed_full.embed --gpu
```