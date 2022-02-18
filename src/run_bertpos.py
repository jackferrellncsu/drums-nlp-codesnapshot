#The code for training the BERT POS model is adopted from https://github.com/soutsios/pos-tagger-bert

from routine_bertpos import *

Total_Raw = pandas.io.parsers.read_csv("Data/brown_pos_bert.csv")

for zzz in range(5):
    #imporing and cleaning data
    Data = Total_Raw['Shuffle' + str(zzz+1)]
    '''
    train_sentences = reformatRaw(np.array(Data[:2]))
    calib_sentences = reformatRaw(np.array(Data[2: 4]))
    test_sentences = reformatRaw(np.array(Data[4:6]))
    '''
    train_sentences = reformatRaw(np.array(Data[:45872]))
    calib_sentences = reformatRaw(np.array(Data[45872: 51606]))
    test_sentences = reformatRaw(np.array(Data[51606:]))


    #Creating set of tags for output vector
    tags = set([item for sublist in train_sentences+calib_sentences+test_sentences for _, item in sublist])

    tag2int = {}
    int2tag = {}

    #making dictionary to easily find index of tags in vector
    for i, tag in enumerate(sorted(tags)):
        tag2int[tag] = i+1
        int2tag[i+1] = tag

    #Adding padding tag
    tag2int['-PAD-'] = 0
    int2tag[0] = '-PAD-'

    n_tags = len(tag2int)
    list(tag2int)

    #Redicing size of each sentence
    M = 100
    train_sentences = split(train_sentences, M)
    calib_sentences = split(calib_sentences, M)
    test_sentences = split(test_sentences, M)

    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    # Params for bert model and tokenization

    #Getting the text and tags from the sentences
    train_text = text_sequence(train_sentences)
    test_text = text_sequence(test_sentences)
    cal_text = text_sequence(calib_sentences)

    train_label = tag_sequence(train_sentences)
    test_label= tag_sequence(test_sentences)
    cal_label= tag_sequence(calib_sentences)

    #Hub model in directory HUB
    tokenizer = create_tokenizer_from_hub_module(sess)

    #Converting Text and labels into examples
    train_examples = convert_text_to_examples(train_text, train_label)
    calib_examples = convert_text_to_examples(cal_text, cal_label)
    test_examples = convert_text_to_examples(test_text, test_label)

    #Converting the examples into BERT features for inputs to our model
    (train_input_ids, train_input_masks, train_segment_ids, train_labels_ids) = convert_examples_to_features(tokenizer, train_examples, tag2int, max_seq_length=M+2)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels_ids) = convert_examples_to_features(tokenizer, test_examples, tag2int, max_seq_length=M+2)
    (calib_input_ids, calib_input_masks, calib_segment_ids, calib_labels_ids) = convert_examples_to_features(tokenizer, calib_examples, tag2int, max_seq_length=M+2)

    #Creating one-hot labels for our POS tags
    train_labels = tf.keras.utils.to_categorical(train_labels_ids, num_classes=n_tags)
    test_labels = tf.keras.utils.to_categorical(test_labels_ids, num_classes=n_tags)
    calib_labels = tf.keras.utils.to_categorical(calib_labels_ids, num_classes=n_tags)

    #Instantiation, training and saving weights for the model
    model = build_model(M+2, n_tags)

    initialize_vars(sess)

    t_ini = datetime.datetime.now()

    cp = ModelCheckpoint(filepath=("bert_tagger" + str(zzz)  + ".h5"),
                         monitor='val_acc',
                         save_best_only=True,
                         save_weights_only=True,
                         verbose=1)

    early_stopping = EarlyStopping(monitor = 'val_acc', patience = 5)

    history = model.fit([train_input_ids, train_input_masks, train_segment_ids],
                        train_labels,
                        validation_data=([train_input_ids, train_input_masks, train_segment_ids], train_labels),
                        #validation_split=0.3,
                        epochs=3,
                        batch_size=16,
                        shuffle=True,
                        verbose=1,
                        callbacks=[cp, early_stopping]
                       )

    t_fin = datetime.datetime.now()
    print('Training completed in {} seconds'.format((t_fin - t_ini).total_seconds()))


