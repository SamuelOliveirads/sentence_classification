# sentence_classification

## Overview

This is your new Kedro project, which was generated using `Kedro 0.18.12`.

Take a look at the [Kedro documentation](https://kedro.readthedocs.io) to get started.

## How to install dependencies

Declare any dependencies in `src/requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r src/requirements.txt
```

Download this embedding for data preparation in the notebook:

http://nlp.stanford.edu/data/glove.6B.zip

Extract the files and place them in the /data/06_models/ folder.

> Note: You will need the raw dataset that contains the phrases and classifications, as this data may be sensitive it will not be made available directly by this project.

## How to run on a notebook

You can run the latest notebook `2.1-api-call.ipynb` to view the final project, the other notebooks are earlier versions.


## How to run the api

Run the following command to start the API locally:

```
python src/sentence_classification/handler.py
```

Then make a request to the api, a request option is at the end of the api-call notebook.