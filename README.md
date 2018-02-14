# MultimodalGame

## Installing

git clone --recurse-submodules https://github.com/lgraesser/MultimodalGame.git

## Dependencies

- Python3
- Pytorch
- scikit-image

You should install Pytorch using instructions from [here](http://pytorch.org/). Otherwise, can install dependencies using pip: `pip install -r requirements.txt`

## Building the Datasets

This model used ShapeWorld datasets. To generate a demo dataset run the following command.

```
mkdir data
./ShapeWorld/build_datasets.sh
```

This model also depends on pretrained word embeddings. We recommend using the `6B.100d` GloVe embeddings available [here](https://nlp.stanford.edu/projects/glove/).

## Running the Code

Here is an example command for running the agents in a "Fixed" setting. Alternatively run the following command to train this model.

```
./run.sh
```

```
python model_symmetric.py \
-experiment_name test_training \
-exchange_samples 0 \
-model_type Fixed \
-max_exchange 1 \
-batch_size 8 \
-batch_size_dev 8 \
-m_dim 50 \
-h_dim 100 \
-desc_dim 100 \
-num_classes 10 \
-learning_rate 1e-4 \
-entropy_agent1 0.01 \
-entropy_agent2 0.01 \
-use_binary \
-max_epoch 100 \
-top_k_dev 1 \
-top_k_train 1 \
-dataset_path ./data/oneshape_simple_textselect \
-dataset_name oneshape_simple_textselect \
-dataset_size_train 250 \
-dataset_size_dev 100  \
-wv_dim 100 \
-glove_path ./glove.6B/glove.6B.100d.txt \
-log_path ./logs \
-debug_log_level INFO
```
