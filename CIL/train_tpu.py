#Code Adapted from : https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
#Code Adapted from : https://github.com/kpe/bert-for-tf2/blob/master/examples/gpu_movie_reviews.ipynb

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import os
from datetime import datetime

import bert
from bert.tokenization.bert_tokenization import FullTokenizer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert import BertModelLayer

from tqdm import tqdm
import numpy as np

from sklearn.utils import shuffle

EXPERIMENT = 'first_experiment'

CHECKPOINT_DIR = './checkpoints'
CHECKPOINT = './bert/checkpoints/bert_base'
CHECKPOINT_CKPT = os.path.join(CHECKPOINT, 'bert_model.ckpt')
CHECKPOINT_VOCAB = os.path.join(CHECKPOINT, 'vocab.txt')
CHECKPOINT_CONFIG = os.path.join(CHECKPOINT, 'bert_config.json')

DATASET_DIR = './'

DATASET_FILE_TRAIN_NEG = os.path.join(DATASET_DIR, 'twitter-datasets/train_neg_full.txt')
DATASET_FILE_TRAIN_POS = os.path.join(DATASET_DIR, 'twitter-datasets/train_pos_full.txt')
DATASET_FILE_TEST = os.path.join(DATASET_DIR, 'twitter-datasets/test_data.txt')

FILE_PATHS = [DATASET_FILE_TRAIN_POS, DATASET_FILE_TRAIN_NEG]

#402, 324
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_TRAIN_EPOCHS = 3.0
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

label_list = [0, 1]

def load_data(file_path):
    data = {}
    data["sentence"] = []
    with open(file_path, "r") as f:
        data["sentence"] = f.readlines()

    #longest_string = max(data["sentence"], key=len)
    #print(longest_string)
    #print(len(longest_string))

    return pd.DataFrame.from_dict(data)

def load_dataset(pos_directory, neg_directory):
    pos_df = load_data(pos_directory)
    neg_df = load_data(neg_directory)

    pos_df["sentiment"] = 1
    neg_df["sentiment"] = 0

    return pd.concat([pos_df, neg_df])

'''def create_dataset():
    train = []
    test = []

    import pdb
    pdb.set_trace()

    return train, test, 256'''

class MovieReviewData:
    DATA_COLUMN = "sentence"
    LABEL_COLUMN = "sentiment"

    def __init__(self, tokenizer= FullTokenizer, sample_size=None, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_seq_len = 0
        trainset = load_dataset(DATASET_FILE_TRAIN_POS, DATASET_FILE_TRAIN_NEG)

        trainset = shuffle(trainset, random_state=5)

        train = trainset.head(2250000)
        test = trainset.tail(250000)

        train = shuffle(train)
        test = shuffle(test)

        trainset.reset_index(inplace=True, drop=True)

        #train, test = map(lambda df: df.reindex(df[MovieReviewData.DATA_COLUMN].str.len().sort_values().index),
        #                 [train, test])

        if sample_size is not None:
            assert sample_size % 128 == 0
            train, test = train.head(sample_size), test.head(250000)
            # train, test = map(lambda df: df.sample(sample_size), [train, test])

        ((self.train_x, self.train_y),
         (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = max_seq_len
        #self.max_seq_len = min(self.max_seq_len, max_seq_len)
        ((self.train_x, self.train_x_token_types),
         (self.test_x, self.test_x_token_types)) = map(self._pad,
                                                       [self.train_x, self.test_x])

    def _prepare(self, df):
        x, y = [], []
        with tqdm(total=df.shape[0], unit_scale=True) as pbar:
            for ndx, row in df.iterrows():
                text, label = row[MovieReviewData.DATA_COLUMN], row[MovieReviewData.LABEL_COLUMN]
                tokens = self.tokenizer.tokenize(text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                self.max_seq_len = max(self.max_seq_len, len(token_ids))
                x.append(token_ids)
                y.append(int(label))
                pbar.update()
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x, t = [], []
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)

def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer


def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler


def create_model(max_seq_len, adapter_size=64):
    """Creates a classification model."""

    # adapter_size = 64  # see - arXiv:1902.00751

    # create the bert layer
    with tf.io.gfile.GFile(CHECKPOINT_CONFIG, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = adapter_size
        bert = BertModelLayer.from_params(bert_params, name="bert")

    input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    # token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
    # output         = bert([input_ids, token_type_ids])
    output = bert(input_ids)

    print("bert shape", output.shape)
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=2, activation="softmax")(logits)

    # model = keras.Model(inputs=[input_ids, token_type_ids], outputs=logits)
    # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])
    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))

    # load the pre-trained model weights
    load_stock_weights(bert, CHECKPOINT_CKPT)

    # freeze weights if adapter-BERT is used
    if adapter_size is not None:
        freeze_bert_layers(bert)

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

    return model

if __name__== "__main__":

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        raise BaseException(
            'ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


    tokenizer = FullTokenizer(vocab_file=CHECKPOINT_VOCAB, do_lower_case=True)

    #data = MovieReviewData(tokenizer, sample_size=10*128*19531, max_seq_len=128)
    data = MovieReviewData(tokenizer, sample_size=10*128*1757, max_seq_len=128)
    #data = MovieReviewData(tokenizer, sample_size=10*128*1000, max_seq_len=128)

    #train, test, max_seq_len = create_dataset()

    '''
    print("            train_x", data.train_x.shape)
    print("train_x_token_types", data.train_x_token_types.shape)
    print("            train_y", data.train_y.shape)

    print("             test_x", data.test_x.shape)

    print("        max_seq_len", data.max_seq_len)
    '''

    adapter_size = None # use None to fine-tune all of BERT or 32,64

    with tpu_strategy.scope():
        model = create_model(data.max_seq_len, adapter_size=adapter_size)
        #model = create_model(max_seq_len, adapter_size=adapter_size)

    model.summary()
    #model.load_weights("twitter_second.h5")

    log_dir = "./log/twitter/" + datetime.now().strftime("%Y%m%d-%H%M%s")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    total_epoch_count = 1

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_DIR,
                                                     save_weights_only=False,
                                                     verbose=1,
                                                     save_freq=32*3*1000)

    dataset_train = tf.data.Dataset.from_tensor_slices((data.train_x, data.train_y))
    dataset_train = dataset_train.batch(32, drop_remainder=True)

    dataset_test = tf.data.Dataset.from_tensor_slices((data.test_x, data.test_y))
    dataset_test = dataset_test.batch(32, drop_remainder=True)

    model.fit(dataset_train,
              validation_data=dataset_test,
              shuffle=True,
              epochs=total_epoch_count,
              callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                        end_learn_rate=1e-7,
                                                        warmup_epoch_count=20,
                                                        total_epoch_count=total_epoch_count),
                         cp_callback,
                         keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                         tensorboard_callback])

    '''model.fit(x=data.train_x, y=data.train_y,
             validation_data=(data.test_x, data.test_y),
             batch_size=32,
             shuffle=True,
             epochs=total_epoch_count,
             callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                      end_learn_rate=1e-7,
                                                      warmup_epoch_count=3,
                                                      total_epoch_count=total_epoch_count),
                       cp_callback,
                       keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                       #tensorboard_callback
                     ])'''


    model.save_weights('./twitter_long.h5', overwrite=True)