from routine_bertpos import *

Total_Raw = pandas.io.parsers.read_csv("brown_pos.csv")

#Making the DataFrame for our results
DD = {}
DD['CA'] = R.rand(5)
DD['Cred'] = R.rand(5)
DD['OP'] = R.rand(5)
DD['OF'] = R.rand(5)
DD['Conf99'] = R.rand(5)
DD['N99'] = R.rand(5)
DD['PIS99'] = R.rand(5)
DD['ACDS99'] = R.rand(5)
DD['Conf999'] = R.rand(5)
DD['N999'] = R.rand(5)
DD['PIS999'] = R.rand(5)
DD['ACDS999'] = R.rand(5)
DD['Conf95'] = R.rand(5)
DD['N95'] = R.rand(5)
DD['PIS95'] = R.rand(5)
DD['ACDS95'] = R.rand(5)
FINALRESULTS = pandas.DataFrame(DD)

for zzz in range(0,5):
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

    model = build_model(M+2, n_tags)

    initialize_vars(sess)
    # Reloading Weights into Model
    model.load_weights('bert_tagger' + str(zzz) + '.h5')

    y_pred_calib = []
    y_true_calib = []

    #Calibration Predictions
    for i in tqdm(range(int(len(calib_input_ids)))):
        YP = model.predict([calib_input_ids[range(i, i+1)], calib_input_masks[range(i, i+1)], calib_segment_ids[range(i, i+1)]], batch_size= 1)
        y_pred_calib.append(YP[0])
        y_true_calib.append(calib_labels[range(i,i+1)].argmax(-1)[0])

    #Temp Function for softmaxes with batches
    def make_prediction(i=16):
        calibr = []
        for w, true, pred in zip(calib_input_ids[i], y_true_calib[i], y_pred_calib[i]):
            if tokenizer.convert_ids_to_tokens([w])[0]!='[PAD]' and \
                tokenizer.convert_ids_to_tokens([w])[0]!='[CLS]' and \
                tokenizer.convert_ids_to_tokens([w])[0]!='[SEP]':
                calibr.append(1 - pred[true])

        return calibr

    #Calculating all the calibration non-conformity scores
    bigCalibr = []
    for ii in range(len(y_pred_calib)):
        bigCalibr.append(make_prediction(i=ii))

    #Flattening and sorting them into vector
    flat_calib = [item for sublist in bigCalibr for item in sublist]

    flat_calib.sort()

    #Predicting for the testing set
    y_pred_test = []
    y_true_test = []
    for i in tqdm(range(int(len(test_input_ids)))):
        YP = model.predict([test_input_ids[range(i, i+1)], test_input_masks[range(i, i+1)], test_segment_ids[range(i, i+1)]], batch_size= 1)
        y_pred_test.append(YP[0])
        y_true_test.append(test_labels[range(i,i+1)].argmax(-1)[0])

    #flattening our predictions and true labels for easy of calculation
    flat_proba = [item for sublist in y_pred_test for item in sublist]
    flat_labs = [item for sublist in y_true_test for item in sublist]

    #Calculating p-Vals and saving true labels
    pVals = []
    trueLabels = []
    for x, y in zip(tqdm(flat_proba), flat_labs):
        if y != 0:
            pVals.append(findPvals(x, flat_calib))
            trueLabels.append(y)

    #Saving our results of the given model
    FINALRESULTS['CA'][zzz] = ClassificationAccuracy(pVals, trueLabels)
    FINALRESULTS['Cred'][zzz] = Cred(pVals)
    FINALRESULTS['OP'][zzz] = OP(pVals, trueLabels)
    FINALRESULTS['OF'][zzz] = OF(pVals, trueLabels)
    FINALRESULTS['Conf99'][zzz],FINALRESULTS['N99'][zzz],FINALRESULTS['PIS99'][zzz],FINALRESULTS['ACDS99'][zzz] = confNPISACDS(pVals, trueLabels, .01)
    FINALRESULTS['Conf999'][zzz],FINALRESULTS['N999'][zzz], FINALRESULTS['PIS999'][zzz],FINALRESULTS['ACDS999'][zzz] = confNPISACDS(pVals, trueLabels, .001)
    FINALRESULTS['Conf95'][zzz],FINALRESULTS['N95'][zzz],FINALRESULTS['PIS95'][zzz],FINALRESULTS['ACDS95'][zzz] = confNPISACDS(pVals, trueLabels, .05)
    FINALRESULTS.to_csv('out/out_bertpos_results/RESULTS.csv')

#Distribution of prediction set sizes
NSIZES = intervalDist(pVals, .01)
mpl.hist(NSIZES, bins = range(0,max(NSIZES)+1), color = 'red')
mpl.xlim((0,max(NSIZES)))
mpl.xticks([0,5,10,15])
mpl.savefig("out/out_bertpos_results/PredSets99.png")

mpl.close()

#Comparison of nominal and empirical confidence levels
nomconf = []
empconf = []
for i in tqdm(range(1, 100)):
    nomconf.append(1- i/100)
    empconf.append(confNPISACDS(pVals, trueLabels, i/100)[0])

mpl.plot(nomconf, empconf, label = "BERT")
mpl.plot(np.array(range(1,100))/100, np.array(range(1,100))/100, label = "Nominal")
mpl.savefig("out/out_bertpos_results/NominalVEmpirical.png")

mpl.close()

#Distribution of Non-Conformity Scores
mpl.hist(flat_calib, color = 'red', bins = int(np.sqrt(len(flat_calib))/7.5))
mpl.savefig("out/out_bertpos_results/NonConfHist.png")

mpl.close()

#Zooming into the small nonconformity scores
trim_calib = []
for x in flat_calib:
    if x < .0002:
        trim_calib.append(x)

mpl.hist(trim_calib, color = 'red', bins = int(np.sqrt(len(trim_calib))))
mpl.xlim(0,.0002)
mpl.xticks([0,.0002])
mpl.savefig("out/out_bertpos_results/NonConfZoom.png")

