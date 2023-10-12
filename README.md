# commonlit-summaries-competition

## Install package for dev

```bash
python -m pip install -e ./package[dev]
```

## Install package locally

```bash
python -m pip install ./package
```

## Build package

```bash
python -m pip install build
python -m build package
```

## Run tests
First download the test data, for example with the Kaggle API:
```
kaggle competitions download -c commonlit-evaluate-student-summaries
```
Unzip the data and place it in a directory called `data` at the top level of the repo. Then run:

```bash
pytest package
```
