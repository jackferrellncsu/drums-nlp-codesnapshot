import datetime
import nltk
import tensorflow.keras as keras
import pandas
import random
import numpy as np
import numpy.random as R
import tensorflow.keras as keras
import tensorflow_hub as hub
import matplotlib.pyplot as mpl
import tensorflow as tf
nltk.download('punkt')
from tensorflow.keras import layers
from tqdm import tqdm
from numpy.random import seed
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from tensorflow.keras import backend as K
from bert.tokenization import FullTokenizer
from bisect import bisect
seed(1)

bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
#If this create_tokenizer_from_hub_module() is frozen comment above and uncomment below
#bert_path = '/HUB'

def trainclaibtestsplit(arr, p ,q, seed):
    '''Defined the splitting of data (no longer used)'''
    random.seed(seed)
    ind1 = random.sample(range(1,len(arr)), int(p * len(arr)))
    notind1 = list(np.setdiff1d(range(0,len(arr)), ind1))
    train = [arr[i] for i in ind1]

    ind2 = random.sample(notind1, int((q / (1-p)) * len(notind1)))
    notind2 = list(np.setdiff1d(notind1, ind2))
    calib = [arr[i] for i in ind2]
    test = [arr[i] for i in notind2]
    return [train,calib,test]

def reformatRaw(array):
    '''
    Function which formats the raw data into a tuple of ((word,pos), ...) for each sentence for our BERT model
    '''
    sentences = [None] * len(array)
    for i in tqdm(range(0, len(array))):
        splx = array[i].split(' ')
        tempsent = [None] * len(splx)
        for ii in range(0, len(splx)):
            splind = splx[ii].rfind('/')
            word = splx[ii][:splind]
            pos = splx[ii][(splind + 1):]
            pos = pos.replace("-tl", "")
            pos = pos.replace("-hl", "")
            pos = pos.replace("fw-", "")
            pos = pos.replace("-nc", "")
            pos = pos.replace("bez", "bbb")
            tempsent[ii] = (word, pos)
        sentences[i] = tempsent
    return sentences

def tag_sequence(sentences):
    '''
    Grabs the tags from the tuple sentence and outputs an array of POS tags
    '''
    return [[t for w, t in sentence] for sentence in sentences]

def text_sequence(sentences):
    '''
    Grabs the words from the tuple sentence and outputs an array of words
    '''
    return [[w for w, t in sentence] for sentence in sentences]

def  split(sentences, max):
    '''
    Splitting sentences that are to long into sentences of length max
    '''
    new=[]
    for data in tqdm(sentences):
        new.append(([data[x:x+max] for x in range(0, len(data), max)]))
    new = [val for sublist in new for val in sublist]
    return new

class PaddedExample(object):
    '''
    If training on a TPU this is required
    '''

class InputExample(object):
    '''
    A single training/test example for simple sequence classification.
    '''

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def convert_single_example(tokenizer, example, tag2int, max_seq_length=256):
    '''
    :param tokenizer: The tokenizer from tensorflow hub
    :param example: the input example
    :param tag2int: dictionary which maps tags => indexes
    :param max_seq_length: max sequence length
    :return: Input Tokens, Masked (ones for words, zeros for pads), Sentence Identification, labels for index of POS

    Changes a single example into a input into BERT
    '''

    if isinstance(example, PaddedExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label_ids = [0] * max_seq_length
        return input_ids, input_mask, segment_ids, label_ids

    tokens_a = example.text_a
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0: (max_seq_length - 2)]

    # Token map will be an int -> int mapping between the `orig_tokens` index and last tokens for word
    orig_to_tok_map = []
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    orig_to_tok_map.append(len(tokens) - 1)

    for token in tokens_a:
        tokens.extend(tokenizer.tokenize(token))
        orig_to_tok_map.append(len(tokens) - 1)
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)
    orig_to_tok_map.append(len(tokens) - 1)
    input_ids = tokenizer.convert_tokens_to_ids([tokens[i] for i in orig_to_tok_map])

    # The mask has 1 for real tokens and 0 for padding tokens.
    input_mask = [1] * len(input_ids)

    label_ids = []
    labels = example.label
    label_ids.append(0)
    label_ids.extend([tag2int[label] for label in labels])
    label_ids.append(0)

    #Pad with zeros up to the sequence length
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)

    return input_ids, input_mask, segment_ids, label_ids

def convert_examples_to_features(tokenizer, examples, tag2int, max_seq_length=256):
    '''
    :param tokenizer: The tokenizer from tensorflow hub
    :param example: vector of input examples
    :param tag2int: dictionary which maps tags => indexes
    :param max_seq_length: max sequence length
    :return: vectors of Input Tokens, Masked (ones for words, zeros for pads), Sentence Identification, labels for index of POS

    Changes vectors of examples into a inputs into BERT
    '''

    input_ids, input_masks, segment_ids, labels = [], [], [], []

    for example in examples:
        input_id, input_mask, segment_id, label = convert_single_example(tokenizer, example, tag2int, max_seq_length)
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)

    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels),
    )


def convert_text_to_examples(texts, labels):
    '''

    :param texts: the vector of vectors of words
    :param labels: the vector of vectors of POS tags
    :return: Training data for BERT

    Creates the input data for our model for training
    '''
    InputExamples = []

    for text, label in zip(texts, labels):
        InputExamples.append(InputExample(guid=None, text_a=text, text_b=None, label=label))
    return InputExamples


class BertLayer(layers.Layer):

    def __init__(self, output_representation='sequence_output', trainable=True, **kwargs):
        '''

        Creating the BertLayer, and setting its parameters to be fine-tuned
        '''
        self.bert = None
        super(BertLayer, self).__init__(**kwargs)

        self.trainable = trainable
        self.output_representation = output_representation

    def build(self, input_shape):
        '''

        :param input_shape: the length of IDs + length of mask vector, length of segment vector
        '''

        #Drawing from the hub model
        self.bert = hub.Module(bert_path,
                               trainable=self.trainable,
                               name="{}_module".format(self.name))

        # Assign module's trainable weights to model
        s = ["/cls/", "/pooler/"]
        self._trainable_weights += [var for var in self.bert.variables[:] if not any(x in var.name for x in s)]

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            self.output_representation
        ]
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs[0], 0.0)

    def compute_output_shape(self, input_shape):
        if self.output_representation == 'pooled_output':
            return (None, 768)
        else:
            return (None, None, 768)

def build_model(max_seq_length, n_tags):
    '''

    :param max_seq_length: The maximum sequence length
    :param n_tags: the number of types of POS tags
    :return: the Keras model

    [in_id, in_mask, in_seg] -> BERT -> Dense(768,190) -> Softmax

    '''

    seed = 1
    in_id = layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = layers.Input(shape=(max_seq_length,), name="input_masks")
    in_seg = layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_in = [in_id, in_mask, in_seg]

    np.random.seed(seed)
    bert_out = BertLayer()(bert_in)

    np.random.seed(seed)
    POSoutputs = layers.Dense(n_tags, activation=keras.activations.softmax)(bert_out)

    np.random.seed(seed)
    model = keras.models.Model(inputs=bert_in, outputs=POSoutputs)

    np.random.seed(seed)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00004), loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    return model

def create_tokenizer_from_hub_module(sess):
    '''

    :return: The tokenizer from the BERT uncased model saved in the local directory HUB
    '''

    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def initialize_vars(sess):
    '''

    Initializing Tensorflow Session
    '''
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


def findPvals(list, calibList):
    '''

    :param list: a vector of softmax output from the BERT Model
    :param calibList: The list of the calibration scores from the calibration set
    :return: the vector of vector of p-values
    '''
    pList = []
    for l in list:
        pList.append(1 - (bisect(calibList, 1 - l) / len(calibList)))
    return pList

def makeIntervals(pvals, epsilon):
    '''

    :param list: vector pvals
    :param epsilon: epsilon cutoff for error rate p>e => in interval
    :return: the conformal prediction set
    '''
    intervals = []
    for i, l in enumerate(pvals):
        if l >= epsilon:
            intervals.append(i)

    return intervals


def Cred(pVals):
    '''

    :param pVals: the vector of vector of pvals
    :return: Credibility for the model
    '''
    sum = 0
    for p in tqdm(pVals):
        sum += max(p)
    return sum / len(pVals)


def OP(pVals, trueLabels):
    '''

    :param pVals: the vector of vector of pvals
    :param trueLabels: The true labels for the POS in form of a vector of vector of ints
    :return: Observed Perceptivness of the prediction sets
    '''
    sum = 0
    for p, x in zip(tqdm(pVals), trueLabels):
        sum += p[x]
    return sum / len(trueLabels)


def OF(pVals, trueLabels):
    '''

    :param pVals: the vector of vector of pvals
    :param trueLabels: The true labels for the POS in form of a vector of vector of ints
    :return: Obsevrved Fuzziness of the prediction sets
    '''
    sum = 0
    for p, x in zip(tqdm(pVals), trueLabels):
        sum += np.sum(p) - p[x]

    return sum / len(trueLabels)


def confNPISACDS(pVals, trueLabels, epsilon):
    '''

    :param pVals: the vector of vector of pvals
    :param trueLabels: The true labels for the POS in form of a vector of vector of ints
    :param epsilon: epsilon cutoff for error rate p>e => in interval
    :return: [Nominal Confidence, N-Criterion, Prop Indecisive Set, Average Confidence of Decisive Sets]
    '''
    allInts = []
    for p in pVals:
        allInts.append(makeIntervals(p, epsilon))

    sum = 0
    count = 0
    above1 = 0
    abovecount = 0
    for x, y in zip(allInts, trueLabels):
        sum += len(x)
        if len(x) > 1:
            above1 += 1
            if y in x:
                abovecount += 1
        if y in x:
            count += 1
    if above1 > 0 and above1 != len(allInts):
        return count / len(allInts), sum / len(allInts), above1 / len(allInts), (count - abovecount) / (
                    len(allInts) - above1)
    #If there are no sets of size >1
    else:
        return count / len(allInts), sum / len(allInts), above1 / len(allInts), -1


def ClassificationAccuracy(pVals, trueLabels):
    '''

    :param pVals: the vector of vector of pvals
    :param trueLabels: The true labels for the POS in form of a vector of vector of ints
    :return: Classification accuracy for the model
    '''
    sum = 0
    for p, i in zip(pVals, trueLabels):
        if np.argmax(p) == i:
            sum += 1
    return sum / len(trueLabels)

def intervalDist(pVals, epsilon):
    '''

    :param pVals: the vector of vector of pvals
    :param epsilon: epsilon cutoff for error rate p>e => in interval
    :return: Distribution of the sizes of intervals
    '''
    allInts = []
    for p in pVals:
        allInts.append(len(makeIntervals(p, epsilon)))

    return allInts


