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
put .csv.gz files in `data/raw/` directory
