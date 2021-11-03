# Helper functions
include("routine_blstm.jl")

# Seed for random operations
Random.seed!(24)

# Creates empty vectors to save results from model
results = []
all_nonconf = []
nom_conf = []

# For loop that iterates through 5 different data splits
for i in 1:5

    # -------------------------- Dealing With Data -------------------------- #

    # Read in raw data
    brown_raw = CSV.read("Data/brown_pos.csv", DataFrame)[!, i]

    # Splits words in each sentence
    raw_sentences = split.(brown_raw, " ")

    # Finds unique POS tags, setences split by words, and the tags for each sentence
    unique_pos, sentences, sentence_tags = data_cleaner(raw_sentences)

    # Creates a vector containing every word and finds the unique ones
    words = get_word_vec(sentences)
    unique_words = convert(Vector{String},unique(words))

    # Finds embeddings for each unique word and creates embedding dictionsary
    embeddings_glove = load_embeddings(GloVe{:en},4, keep_words=Set(unique_words))
    embtable = Dict(word=>embeddings_glove.embeddings[:,ii] for (ii,word) in
                                                enumerate(embeddings_glove.vocab))

    # Creates embeddings for every word in every sentence
    sent_vec = sentence_embed(sentences, embtable)

    # Creates onehot vectors for every word in every sentence, regarding the POS
    onehot_vec = get_onehot(unique_pos, sentence_tags)

    # Train/Test/Calib Split
    train = sent_vec[1:45872]
    train_class = onehot_vec[1:45872]
    test = sent_vec[45873:51606]
    test_class = onehot_vec[45873:51606]
    calib = sent_vec[51607:end]
    calib_class = onehot_vec[51607:end]

    # Creates DataLoader classes
    dl_calib = Flux.Data.DataLoader((calib, calib_class))
    dl_test = Flux.Data.DataLoader((test, test_class))
    dl_train = Flux.Data.DataLoader((train, train_class),
                                        batchsize = 1000, shuffle = true)

    # ---------------- Begin Training and Testing Model Loop ---------------- #

    # Model Architecture
    forward = LSTM(300, 150)
    backward = LSTM(300, 150)
    predictor =  Chain(Dense(300, 250, relu), Dense(250, 190, x->x), softmax)

    function BiLSTM(x)
        Flux.reset!(forward)
        Flux.reset!(backward)
        fw_pred = forward.(x)
        bw_pred = reverse(backward.(reverse(x)))
        final_pred = vcat.(fw_pred, bw_pred)
        return final_pred
    end

    # Predicts embedding for chosen word
    model(x) = predictor.(BiLSTM(x))

    # Optimizer
    opt = Flux.Optimiser(ExpDecay(0.01, 0.1, 1, 1e-4), RADAM())

    # Model parameters
    ps = Flux.params((forward, backward, predictor))

    function loss(x, y)
        l = 0
        for i in length(x)
            l += sum(Flux.Losses.crossentropy.(model(x[i]),y[i]))
        end
        return l
    end

    # Model Training
    for i in ProgressBar(1:700)
        Flux.train!(loss, ps, dl_train, opt)
    end

    # --------------------------- Model Analysis ---------------------------- #

    # Calculates nonconformity scores for trained model on calibration set
    α_i = find_nonconf_vals(calib, calib_class, model)

    # Finds the actual POS onehots for each word and the p-values
    act_pos, pvals = get_pvals(test, test_class, model, α_i)

    # Finds model's prediction for chosen word's POS and actual POS for chosen word
    args_mod, args_class = pred_class(test, test_class, model)

    # ------------------------ Performance Criterion ------------------------ #

    # Classification Accuracy
    CA = class_accuracy(act_pos, pvals)

    # Credibility
    cred = credibility(pvals)

    # Observed Perceptiveness and Observed Fuzziness
    OP, OF = find_OP_OF(pvals, act_pos)

    # Nominal confidence levels at each ϵ from 0.01 to 0.5 with step of 0.05
    nominal = nominal_conf(pvals, act_pos)

    # ----------------------- Stats for 99% Conf Sets ----------------------- #

    acds_99 = find_ACDS(pvals, args_mod, args_class, .01)
    pis_99 = find_PIS(pvals, .01)
    sizes_99, ncrit_99 = avg_set_sizes(pvals, .01)
    empconf_99 = empconf(pvals, act_pos, .01)

    # ----------------------- Stats for 95% Conf Sets ----------------------- #

    acds_95 = find_ACDS(pvals, args_mod, args_class, .05)
    pis_95 = find_PIS(pvals, .05)
    sizes_95, ncrit_95 = avg_set_sizes(pvals, .05)
    empconf_95 = empconf(pvals, act_pos, .05)

    # ----------------------- Stats for 99.9% Conf Sets ----------------------- #

    acds_999 = find_ACDS(pvals, args_mod, args_class, .001)
    pis_999 = find_PIS(pvals, .001)
    sizes_999, ncrit_999 = avg_set_sizes(pvals, .001)
    empconf_999 = empconf(pvals, act_pos, .001)

    # ----------------------- Saving Model Results ----------------------- #

    temp_results = []
    push!(temp_results, acds_99, pis_99, ncrit_99, empconf_99,
                        acds_95, pis_95, ncrit_95, empconf_95,
                        acds_999, pis_999, ncrit_999, empconf_999,
                        OP, OF, cred, CA, sizes_99, sizes_999)

    push!(results, temp_results)

    push!(all_nonconf, α_i)

    push!(nom_conf, nominal)

end

# --------------------- Preparing results for out file --------------------- #

# Loads in criterion and set size results from training and testing loop
criterion, sizes_ninety_nine, sizes_ninety_nine_nine = get_results(results)

# Loads in nominal confidence results from training and testing loop
nominal_confidence, confs = get_nominal(nom_conf)

# Loads in the nonconformity values from the 5th split of the train/test loop
α_vals = all_nonconf[5]

# ----------------------- Saving results for out file ----------------------- #

# Stores plot requirements
plot_requirements = Dict("α_i" => α_vals,
                         "nominal" => nominal_confidence,
                         "confs" => confs,
                         "sizes_99" => sizes_ninety_nine,
                         "sizes_999" => sizes_ninety_nine_nine)

# Stores "Table 1 (blstm)"
t1_data = [(conf_level = "99.9%", emp_conf = criterion[12], acds = criterion[9],
                           pis = criterion[10], N = criterion[11]),
           (conf_level = "99%", emp_conf = criterion[4], acds = criterion[1],
                                      pis = criterion[2], N = criterion[3]),
           (conf_level = "95%", emp_conf = criterion[8], acds = criterion[5],
                                      pis = criterion[6], N = criterion[7])]

# Stores "Table 2 (blstm)"
t2_data = [(CA = criterion[16], cred = criterion[15], OP = criterion[13],
                                                            OF = criterion[14])]

# Saves results and writes them to a JLD file for the out file to read in
JLD.save("out/out_blstm_results/run_blstm_output.jld",
                                 "plot_requirements", plot_requirements,
                                 "t1_data", t1_data,
                                 "t2_data", t2_data)
