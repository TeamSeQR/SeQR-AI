import re 
import time 
import datetime
import os
import pdb 
import pickle
import argparse 
import numpy as np 
from tqdm import tqdm
from bisect import bisect_left  
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer  # Updated Import
from TextCNN import * 
from utils import *  

parser = argparse.ArgumentParser(description="Train URLNet model") 

# Data arguments
default_max_len_words = 200
parser.add_argument('--data.max_len_words', type=int, default=default_max_len_words, metavar="MLW",
  help="maximum length of url in words (default: {})".format(default_max_len_words))
default_max_len_chars = 200
parser.add_argument('--data.max_len_chars', type=int, default=default_max_len_chars, metavar="MLC",
  help="maximum length of url in characters (default: {})".format(default_max_len_chars))
default_max_len_subwords = 20 
parser.add_argument('--data.max_len_subwords', type=int, default=default_max_len_subwords, metavar="MLSW",
  help="maximum length of word in subwords/characters (default: {})".format(default_max_len_subwords))
default_min_word_freq = 1
parser.add_argument('--data.min_word_freq', type=int, default=default_min_word_freq, metavar="MWF",
  help="minimum frequency of word in training population to build vocabulary (default: {})".format(default_min_word_freq))
default_dev_pct = 0.001
parser.add_argument('--data.dev_pct', type=float, default=default_dev_pct, metavar="DEVPCT",
  help="percentage of training set used for dev (default: {})".format(default_dev_pct))
parser.add_argument('--data.data_dir', type=str, default='train_10000.txt', metavar="DATADIR",
  help="location of data file")
default_delimit_mode = 1 
parser.add_argument("--data.delimit_mode", type=int, default=default_delimit_mode, metavar="DLMODE",
  help="0: delimit by special chars, 1: delimit by special chars + each char as a word (default: {})".format(default_delimit_mode))

# Model arguments
default_emb_dim = 32
parser.add_argument('--model.emb_dim', type=int, default=default_emb_dim, metavar="EMBDIM",
  help="embedding dimension size (default: {})".format(default_emb_dim))
default_filter_sizes = "3,4,5,6"
parser.add_argument('--model.filter_sizes', type=str, default=default_filter_sizes, metavar="FILTERSIZES",
  help="filter sizes of the convolution layer (default: {})".format(default_filter_sizes))
default_emb_mode = 1 
parser.add_argument('--model.emb_mode', type=int, default=default_emb_mode, metavar="EMBMODE",
  help="1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN (default: {})".format(default_emb_mode))

# Train arguments
default_nb_epochs = 5
parser.add_argument('--train.nb_epochs', type=int, default=default_nb_epochs, metavar="NEPOCHS",
  help="number of training epochs (default: {})".format(default_nb_epochs))
default_batch_size = 128
parser.add_argument('--train.batch_size', type=int, default=default_batch_size, metavar="BATCHSIZE",
  help="Size of each training batch (default: {})".format(default_batch_size))
parser.add_argument('--train.l2_reg_lambda', type=float, default=0.0, metavar="L2LREGLAMBDA",
  help="l2 lambda for regularization (default: 0.0)")
default_lr = 0.001
parser.add_argument('--train.lr', type=float, default=default_lr, metavar="LR",
  help="learning rate for optimizer (default: {})".format(default_lr))

# Log arguments
parser.add_argument('--log.output_dir', type=str, default="runs/10000/", metavar="OUTPUTDIR",
  help="directory of the output model")
parser.add_argument('--log.print_every', type=int, default=50, metavar="PRINTEVERY",
  help="print training result every this number of steps (default: 50)")
parser.add_argument('--log.eval_every', type=int, default=500, metavar="EVALEVERY",
  help="evaluate the model every this number of steps (default: 500)")
parser.add_argument('--log.checkpoint_every', type=int, default=500, metavar="CHECKPOINTEVERY",
  help="save a model every this number of steps (default: 500)")

FLAGS = vars(parser.parse_args())

for key, val in FLAGS.items():
  print("{}={}".format(key, val))

# Read the data
urls, labels = read_data(FLAGS["data.data_dir"]) 

# Initialize the tokenizer
tokenizer = Tokenizer()  # You can set num_words or other parameters as needed
tokenizer.fit_on_texts(urls)

# Convert texts to sequences
urls_sequences = tokenizer.texts_to_sequences(urls)

# Create word index from the tokenizer
word_index = tokenizer.word_index
print("Found {} unique tokens.".format(len(word_index)))

# Prepare the data
max_len_words = FLAGS["data.max_len_words"]
x = pad_sequences(urls_sequences, maxlen=max_len_words)  # Adjust the maxlen as needed

# Splitting the data for training and testing
pos_x = []
neg_x = []
for i in range(len(labels)):
    label = labels[i] 
    if label == 1: 
        pos_x.append(i)
    else: 
        neg_x.append(i)
print("Overall Mal/Ben split: {}/{}".format(len(pos_x), len(neg_x)))
pos_x = np.array(pos_x) 
neg_x = np.array(neg_x) 

x_train, y_train, x_test, y_test = prep_train_test(pos_x, neg_x, FLAGS["data.dev_pct"])

# Use the updated `x` for training and testing
x_train = x[pos_x]
x_test = x[neg_x]

###################################### Training #########################################################

def train_dev_step(x, y, emb_mode, is_train=True):
    if is_train: 
        p = 0.5
    else: 
        p = 1.0
    if emb_mode == 1: 
        feed_dict = {
            cnn.input_x_char_seq: x[0],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}  
    elif emb_mode == 2: 
        feed_dict = {
            cnn.input_x_word: x[0],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == 3: 
        feed_dict = {
            cnn.input_x_char_seq: x[0],
            cnn.input_x_word: x[1],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == 4: 
        feed_dict = {
            cnn.input_x_word: x[0],
            cnn.input_x_char: x[1],
            cnn.input_x_char_pad_idx: x[2],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    elif emb_mode == 5:  
        feed_dict = {
            cnn.input_x_char_seq: x[0],
            cnn.input_x_word: x[1],
            cnn.input_x_char: x[2],
            cnn.input_x_char_pad_idx: x[3],
            cnn.input_y: y,
            cnn.dropout_keep_prob: p}
    if is_train:
        _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
    else: 
        step, loss, acc = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
    return step, loss, acc

def make_batches(x_train_char_seq, x_train_word, x_train_char, y_train, batch_size, nb_epochs, shuffle=False):
    if FLAGS["model.emb_mode"] == 1:  
        batch_data = list(zip(x_train_char_seq, y_train))
    elif FLAGS["model.emb_mode"] == 2:  
        batch_data = list(zip(x_train_word, y_train))
    elif FLAGS["model.emb_mode"] == 3:  
        batch_data = list(zip(x_train_char_seq, x_train_word, y_train))
    elif FLAGS["model.emb_mode"] == 4:
         batch_data = list(zip(x_train_char, x_train_word, y_train))
    elif FLAGS["model.emb_mode"] == 5:  
        batch_data = list(zip(x_train_char, x_train_word, x_train_char_seq, y_train))
    batches = batch_iter(batch_data, batch_size, nb_epochs, shuffle)

    if nb_epochs > 1: 
        nb_batches_per_epoch = int(len(batch_data)/batch_size)
        if len(batch_data)%batch_size != 0:
            nb_batches_per_epoch += 1
        nb_batches = int(nb_batches_per_epoch * nb_epochs)
        return batches, nb_batches_per_epoch, nb_batches
    else:
        return batches 

def prep_batches(batch):
    if FLAGS["model.emb_mode"] == 1:
        x_char_seq, y_batch = zip(*batch)
    elif FLAGS["model.emb_mode"] == 2:
        x_word, y_batch = zip(*batch)
    elif FLAGS["model.emb_mode"] == 3:
        x_char_seq, x_word, y_batch = zip(*batch)
    elif FLAGS["model.emb_mode"] == 4:
        x_char, x_word, y_batch = zip(*batch)
    elif FLAGS["model.emb_mode"] == 5:
        x_char, x_word, x_char_seq, y_batch = zip(*batch)

    x_batch = []
    if FLAGS["model.emb_mode"] in [1, 3, 5]:
        x_char_seq = pad_seq_in_word(x_char_seq, FLAGS["data.max_len_chars"])
        x_batch.append(x_char_seq)
    if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
        x_word = pad_seq_in_word(x_word, FLAGS["data.max_len_words"])
        x_batch.append(x_word)
    if FLAGS["model.emb_mode"] in [4, 5]:
        x_char, x_char_pad_idx = pad_seq(x_char, FLAGS["data.max_len_words"], FLAGS["data.max_len_subwords"], FLAGS["model.emb_dim"])
        x_batch.extend([x_char, x_char_pad_idx])
    return x_batch, y_batch

# TensorFlow 2.x compatible training loop
tf.compat.v1.disable_eager_execution()  # Disable eager execution for TensorFlow 2.x
sess = tf.compat.v1.Session()

with sess.as_default():
    cnn = TextCNN(
        char_ngram_vocab_size=len(word_index) + 1,
        word_ngram_vocab_size=len(word_index) + 1,
        char_vocab_size=len(word_index) + 1,
        embedding_size=FLAGS["model.emb_dim"],
        word_seq_len=FLAGS["data.max_len_words"],
        char_seq_len=FLAGS["data.max_len_chars"],
        l2_reg_lambda=FLAGS["train.l2_reg_lambda"],
        mode=FLAGS["model.emb_mode"],
        filter_sizes=list(map(int, FLAGS["model.filter_sizes"].split(",")))
    )

    global_step = tf.Variable(0, name="global_step", trainable=False) 
    optimizer = tf.compat.v1.train.AdamOptimizer(FLAGS["train.lr"])
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    print("Writing to {}\n".format(FLAGS["log.output_dir"]))
    if not os.path.exists(FLAGS["log.output_dir"]): 
        os.makedirs(FLAGS["log.output_dir"])

    # Save dictionary files 
    word_index_dir = FLAGS["log.output_dir"] + "word_index.p"
    pickle.dump(word_index, open(word_index_dir, "wb"))

    # Save training and validation logs 
    train_log_dir = FLAGS["log.output_dir"] + "train_logs.csv"
    with open(train_log_dir, "w") as f:
        f.write("step,time,loss,acc\n") 
    val_log_dir = FLAGS["log.output_dir"] + "val_logs.csv"
    with open(val_log_dir, "w") as f:
        f.write("step,time,loss,acc\n")

    # Save model checkpoints 
    checkpoint_dir = FLAGS["log.output_dir"] + "checkpoints/" 
    if not os.path.exists(checkpoint_dir): 
        os.makedirs(checkpoint_dir) 
    checkpoint_prefix = checkpoint_dir + "model"
    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=5) 

    sess.run(tf.compat.v1.global_variables_initializer())

    # Corrected make_batches call
    train_batches, nb_batches_per_epoch, nb_batches = make_batches(x_train, x_train, x_train, y_train, FLAGS["train.batch_size"], FLAGS['train.nb_epochs'], True)

    min_dev_loss = float('Inf') 
    dev_loss = float('Inf')
    dev_acc = 0.0 
    print("Number of batches in total: {}".format(nb_batches))
    print("Number of batches per epoch: {}".format(nb_batches_per_epoch))

    it = tqdm(range(nb_batches), desc="emb_mode {} delimit_mode {} train_size {}".format(FLAGS["model.emb_mode"], FLAGS["data.delimit_mode"], x_train.shape[0]), ncols=0)
    for idx in it:
        batch = next(train_batches)
        x_batch, y_batch = prep_batches(batch) 
        step, loss, acc = train_dev_step(x_batch, y_batch, emb_mode=FLAGS["model.emb_mode"], is_train=True)
        
        # Enhanced logging for debugging
        print("Step: {}, Loss: {:.5f}, Accuracy: {:.5f}".format(step, loss, acc))                     
        
        if step % FLAGS["log.print_every"] == 0: 
            with open(train_log_dir, "a") as f:
                f.write("{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), loss, acc)) 
            it.set_postfix(
                trn_loss='{:.3e}'.format(loss),
                trn_acc='{:.3e}'.format(acc),
                dev_loss='{:.3e}'.format(dev_loss),
                dev_acc='{:.3e}'.format(dev_acc),
                min_dev_loss='{:.3e}'.format(min_dev_loss))
        if step % FLAGS["log.eval_every"] == 0 or idx == (nb_batches - 1): 
            total_loss = 0
            nb_corrects = 0
            nb_instances = 0
            test_batches = make_batches(x_test, x_test, x_test, y_test, FLAGS['train.batch_size'], 1, False)
            for test_batch in test_batches:
                x_test_batch, y_test_batch = prep_batches(test_batch)
                step, batch_dev_loss, batch_dev_acc = train_dev_step(x_test_batch, y_test_batch, emb_mode=FLAGS["model.emb_mode"], is_train=False)
                nb_instances += x_test_batch[0].shape[0]
                total_loss += batch_dev_loss * x_test_batch[0].shape[0]
                nb_corrects += batch_dev_acc * x_test_batch[0].shape[0]
            
            dev_loss = total_loss / nb_instances 
            dev_acc = nb_corrects / nb_instances 
            with open(val_log_dir, "a") as f: 
                f.write("{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), dev_loss, dev_acc))
            if step % FLAGS["log.checkpoint_every"] == 0 or idx == (nb_batches - 1): 
                if dev_loss < min_dev_loss: 
                    path = saver.save(sess, checkpoint_prefix, global_step=step) 
                    min_dev_loss = dev_loss  
                    print("Saved model checkpoint to {}\n".format(path))
