# MultimodalGame

## Dependencies

- Python2.7
- Pytorch

You should install Pytorch using instructions from [here](http://pytorch.org/). Otherwise, can install dependencies using pip: `pip install -r requirements.txt`

## Building the Datasets

This model requires an hdf5 file containing image features and csv file containing class descriptions. To build such a dataset using images from Imagenet, you can simply run the following script:

```
cd ./utils
bash build_datasets.sh
```

This will download image urls from Imagenet (~300mb compressed), save urls from 30 classes, split them into train/dev/test, download the relevant images, extract the necessary features using a pretrained ResNet-34, and build the descriptions file.

This model also depends on pretrained word embeddings. We recommend using the `6B.100d` GloVe embeddings availabe [here](https://nlp.stanford.edu/projects/glove/).

## Running the Code

Here is an example command for running the agents in an "Adaptive" setting, where the Receiver has the option to terminate the conversation and make a prediction before the maximum number of exchange steps have been exhausted.

```
python model.py \
-experiment_name demo \ # used to save various log files
-exchange_samples 5 \ # print samples of the communication
-model_type Adaptive \ # the receiver will determine when to stop the conversation
-max_exchange 10 \ # max number of exchange steps in the agents' conversation
-batch_size 64 \
-rec_w_dim 32 \ # message dimension of the receiver (this should match the sender)
-sender_out_dim 32 \ # message dimension of the sender (this should match the receiver)
-img_h_dim 256 \ # hidden dimension of the sender
-rec_hidden 64 \ # hidden dimension of the receiver
-learning_rate 1e-4 \ # learning rate for gradient descent
-entropy_rec 0.01 \ # regularize the receiver's messages
-entropy_sen 0.01 \ # regularize the sender's messages
-entropy_s 0.08 \ # regularize the stop bit 
-use_binary \ # specify binary communication (continuous values are also an option)
-max_epoch 500 \ # number of epochs to train
-top_k_dev 6 \ # specify tok-k for dev
-top_k_train 6 \ # specify top-k for train
-descr_train ./utils/descriptions.csv \
-descr_dev ./utils/descriptions.csv \
-train_file ./utils/train.hdf5 \
-dev_file ./utils/dev.hdf5 \
-wv_dim 100 \ # dimension of word vector
-glove_path ~/data/glove/glove.6B.100d.txt
```

## Message Analysis

After training a model, it's desirable to examine the binary messages used in the communication between the Sender and Receiver. These can be retrieved with a command along the lines of the following:

```
EXPERIMENT_NAME="demo"; \
    python model.py \
    -log_load ./logs/${EXPERIMENT_NAME}.json \ # load model configuration from here
    -binary_only \ # specify to only extract binary messages (`eval_only` is also an option)
    -experiment_name demo-binary \ # write output to a log file different from training
    -checkpoint ./logs/${EXPERIMENT_NAME}.pt_best \ # load this checkpoint
    -binary_output ./logs/${EXPERIMENT_NAME}.bv.hdf5 \ # save messages as an hdf5
    -fixed_exchange # use `fixed_exchange` since the adaptive length can be determined with the stop bits
```
