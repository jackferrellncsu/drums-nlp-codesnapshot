# -------------------- Importing all necessary packages -------------------- #

using JLD
using LinearAlgebra
using Statistics
using Embeddings
using Flux
using Random
using DataFrames
using Plots
using StatsBase
using BSON
using ProgressBars
using CSV
using TypedTables
using Zygote

# -------------------------------- Functions -------------------------------- #

"""
    word_masker(sentences, tags)

Mask a randomly selected word in every sentence.

Parameter sentences (Vector{Vector{String}}) - Vector of all sentences where
    each word in each sentence vector is stored as a string.
Parameter tags (Vector{Vector{String}}) - Vector of all tags for each word in
    every sentence. Corresponds with the sentences vector of vectors.

Return act_word (Vector{Any}) - Vector of words that is masked in each sentence.
Return act_pos (Vector{Any}) - Vector of masked words corresponding parts of
    speech.
Return sentences (Vector{Vector{String}}) - New sentences vector with each
    masked word being replaced with the string "/MASK/".
"""
function word_masker(sentences, tags)

    act_word = []
    act_pos = []
    for i in 1:length(sentences)
        samp = sample(1:length(sentences[i]))
        push!(act_word, sentences[i][samp])
        push!(act_pos, tags[i][samp])
        sentences[i][samp] = "/MASK/"
    end
    return act_word, act_pos, sentences
end

"""
    data_cleaner(raw_sentences)

Clean Brown Corpus by removing the tags for titles, headings, foreign words and
emphasized words. Reduce number of unique tags to 190.

Parameter raw_sentences ((Vector{Vector{SubString{String}}})) - Vector of all
    sentences where each word/tag combination in each sentence vector is stored
    as a string.

Return tagger (Vector{Any}) - Vector of unique parts of speech stored as strings
Return raw_words (Vector{Any}) - Vector of all sentences where each word in each
    sentence vector is stored as a string.
Return raw_tags (Vector{Any}) - Vector of all tags for each word in every
    sentence. Corresponds with the raw_words vector.
"""
function data_cleaner(raw_sentences)

    raw_tags = []
    raw_words = []
    for sent in raw_sentences
        raw_tags_temp = []
        raw_words_temp = []
        for word in sent
            ind = findlast(x -> x == '/', word)
            pos = word[ind+1:end]
            pos = replace(pos, "-tl" => "")
            pos = replace(pos, "-hl" => "")
            pos = replace(pos, "fw-" => "")
            pos = replace(pos, "-nc" => "")
            pos = replace(pos, "bez" => "bbb")
            push!(raw_tags_temp, convert(String, pos))
            push!(raw_words_temp, lowercase(word[1:ind-1]))
        end
        push!(raw_tags, raw_tags_temp)
        push!(raw_words, raw_words_temp)
    end

    individual_tags = []
    for i in 1:length(raw_tags)
        for j in 1:length(raw_tags[i])
            push!(individual_tags, (raw_tags[i][j]))
        end
    end

    tagger = unique!(individual_tags)

    return tagger, raw_words, raw_tags
end

"""
    get_word_vec(sentences)

Create a vector filled with every element (word) in sentences.

Parameter sentences (Vector{Any}) - Vector of all sentences where
    each word in each sentence vector is stored as a string.

Return words (Vector{String}) - Vector of every word in sentences, stored as
    strings.
"""
function get_word_vec(sentences)

    words = Vector{String}()
    for i in 1:length(sentences)
        for j in 1:length(sentences[i])
            push!(words, sentences[i][j])
        end
    end
    return words
end

"""
    sent_embeddings(sentences, sentence_tags, num_embed, num_words, dict)

Create a tensor of word embeddings (num_embed x num_words x length(sentences)).
Truncates and pads with zeros accordingly.

Parameter sentences (Vector{Any}) - Vector of all sentences where
    each word in each sentence vector is stored as a string.
Parameter sentence_tags (Vector{Any}) - Vector of all tags for each word in
    every sentence. Corresponds with the sentences vector.
Parameter num_embed {Int64} - Length of embedding vectors.
Parameter num_words {Int64} - Number of spaces every sentence will take up.
Parameter dict (Dict{String, Vector{Float32}}) - Embedding dictionary that has
    each word as a key and each words embedding as a value.

Return tens (Array{Float64, 3}) - Tensor of word embeddings for each word in
    each sentence (num_embed x num_words x length(sentences)).
Return sent (Vector{Array{String, 1}}) - Vector of possibly truncated
    sentences, depending on the num_words value. If num_words is greater than
    the length of the longest sentence, then sent will just be the same as the
    parameter sentences.
Return tags (Vector{Array{String, 1}}) - Vector of tags for each word in each
    sentence, corresponds with the sent vector of vectors.
"""
function sent_embeddings(sentences, sentence_tags, num_embed, num_words, dict)

    # Embeddings for each sentence
    tens = zeros(num_embed, num_words, length(sentences))

    sent = []
    tags = []
    for i in 1:length(sentences)
        temp = []
        temp_tag = []
        for j in 1:length(sentences[i])
            if length(sentences[i]) < num_words
                tens[:, j, i] = get(dict, sentences[i][j], zeros(num_embed))
                push!(temp, sentences[i][j])
                push!(temp_tag, sentence_tags[i][j])
            elseif length(sentences[i]) >= num_words
                if j == (num_words + 1)
                    break
                else
                    tens[:, j, i] = get(dict, sentences[i][j], zeros(num_embed))
                    push!(temp, sentences[i][j])
                    push!(temp_tag, sentence_tags[i][j])
                end
            end
        end
        # Pre-sentences with < "num_words" are padded with zeros
        if length(sentences[i]) < num_words

            vecs_needed = 0
            vecs_needed = num_words - length(sentences[i])

            for j in 1:vecs_needed
                tens[:, length(sentences[i]) + j, i] .= 0.0
            end
        end
        push!(sent, temp)
        push!(tags, temp_tag)
    end

    sent = convert(Vector{Vector{String}}, sent)
    tags = convert(Vector{Vector{String}}, tags)
    tens = convert(Array{Float32, 3}, tens)
    return tens, sent, tags
end

"""
    masked_embeddings(new_sentences, sent_tens)

Find indices for the masked word in each sentence. Go through embedded tensor
and replace the masked embedding with a num_embed length vector filled with
Float64 values -20.0.

Parameter new_sentences (Vector{Vector{String}}) - Sentence vector with the
    masked word in each sentence being replaced with the string "/MASK/".
Parameter sent_tens (Array{Float64, 3}) - Tensor of word embeddings for each
    word in each sentence.
Parameter num_embed {Int64} - Length of embedding vectors.

Return masked_ind (Vector{Any}) - Vector of the indices for masked word in each
    sentence.
Return sent_tens (Array{Float64, 3}) - Tensor of word embeddings for each word
    in each sentence (num_embed x num_words x length(sentences)) with each
    masked embedding being replaced with a vector of -20.0's.
"""
function masked_embeddings(new_sentences, sent_tens, num_embed)

    # Finds indices of masked words for each sentence
    masked_ind = []
    for i in 1:length(new_sentences)
        for j in 1:length(new_sentences[i])
            if new_sentences[i][j] == "/MASK/"
                push!(masked_ind, j)
            end
        end
    end

    # Embeddings for each sentence
    for i in 1:length(sent_tens[1, 1, :])
        temp = []
        mask = masked_ind[i]
        sent_tens[:, mask, i] = fill(-20.0, num_embed)
    end


    return masked_ind, sent_tens
end

"""
    create_window(sent_tens_emb, window_size)

Modify an embedding tensor by placing the masked word in the center of each
"sentence" and truncating/padding on either side of the word to meet the
window size.

Parameter sent_tens_emb (Array{Float64, 3}) - Tensor of word embeddings for each
    word in each sentence with each masked embedding being replaced with a
    vector of -20.0's.
Parameter window_size {Int64} - Number of words of either side of the masked
    word.

Return new_tens (Array{Float64, 3}) - Tensor of word embeddings for each word in
    each sentence with each masked embedding being in the center of each
    sentence.
"""
function create_window(sent_tens_emb, window_size)

    mask_ind = 0
    num_wind = (window_size * 2) + 1
    new_tens = zeros(300, num_wind, length(sent_tens_emb[1, 1, :]))
    for i in 1:length(sent_tens_emb[1, 1, :])
        mask_ind = argmin(sent_tens_emb[1, :, i])
        if mask_ind == 16
            new_tens[:, :, i] = sent_tens_emb[:, 1:num_wind, i]
        elseif mask_ind > 16
            num_start = (mask_ind - 16) + 1
            num_end = num_wind + (num_start - 1)
            new_tens[:, :, i] = sent_tens_emb[:, num_start:num_end, i]
        elseif mask_ind < 16
            num_zero = window_size - (mask_ind - 1)
            new_mat = zeros(300, num_zero)
            stopper = mask_ind + (window_size)
            new_tens[:, :, i] = hcat(new_mat, sent_tens_emb[:, 1:stopper, i])
        end
    end

    new_tens = convert(Array{Float32, 3}, new_tens)
    return new_tens
end

"""
    SampleMats(x_mat, y_mat, prop = 0.9)

Split data into training, training class, test, and test class for model input.

Parameter x_mat (Array{Float32, 3}) - Tensor of word embeddings for each
    word in each sentence with each masked embedding being replaced with a
    vector of -20.0's.
Parameter y_mat (Matrix{Float32}) - Matrix of one hot vectors corresponding to
    the tensor of word embeddings.

Return train_x (Array{Float32, 3}) - Training tensor for model.
Return test_x  (Array{Float32, 3}) - Testing tensor for model.
Return train_y (Matrix{Float32}) - Training class for model.
Return test_y (Matrix{Float32}) - Testing class for model.
"""
function SampleMats(x_mat, y_mat, prop = 0.9)

    inds = [1:size(x_mat)[3];]
    length(inds)
    trains = sample(inds, Int(floor(length(inds) * prop)), replace = false)
    inds = Set(inds)
    trains = Set(trains)
    tests = setdiff(inds, trains)

    train_x = x_mat[:, :, collect(trains)]
    train_y = y_mat[:, collect(trains)]

    test_x = x_mat[:, :, collect(tests)]
    test_y = y_mat[:, collect(tests)]


    return train_x, test_x, train_y, test_y
end

"""
    tag_freq(tag)

Testing method that counts how many times any given tag appears in the
prediction sets.

Parameter tag (String) - Tag that is counted.

Return counter (Int64) - The total amount of times the tag appears in the
    prediction sets.
"""
function tag_freq(tag)

    counter = 0
    for i in sets
        for j in i
            if j == tag
                counter += 1
            end
        end
    end
    return counter
end

"""
    BLSTM(x)

Bidirectional Long Short-Term Memory layer.

Parameter x (Array{Float32, 2}) - X value from the training dataloader.

Return res (Array{Float32, 2}) - Matrix of the forward LSTM predictions
    concatenated with backward LSTM predictions.
"""
function BLSTM(x)

    #Flux.reset!((forward, backward))
    fw = forward.([x[:, 1:15, i] for i in 1:size(x, 3)])
    fw_mat = hcat.(f[:,15] for f in fw)

    bw = backward.([x[:, end:-1:17, i] for i = size(x, 3):-1:1])
    bw_mat = hcat.(b[:,15] for b in bw)

    fw_temp = fw_mat[1]
    for i in 2:length(fw_mat)
        fw_temp = hcat(fw_temp, fw_mat[i])
    end

    bw_temp = bw_mat[1]
    for i in 2:length(bw_mat)
        bw_temp = hcat(bw_temp, bw_mat[i])
    end
    #@show fw_temp
    res = vcat(fw_temp, bw_temp)
    #@show res
    return res
end

"""
    loss(x, y)

Loss function for model.

Parameter x (Array{Float32, 2}) - X value from the training dataloader.
Parameter y (Array{Float32, 2}) - Y value from the training dataloader.

Return l (Float64) - Loss value.
"""
function loss(x, y)
    Flux.reset!(forward)
    Flux.reset!(backward)
    l = Flux.Losses.crossentropy(model(x), y)
    return l
end

"""
    load_parameters!(weights)

Loads weights from the "weights" vector that was saved after training model.

Parameter weights (Vector{Any}) - Vector of updated parameters for each layer
    of the neural network.
"""
function load_parameters!(weights)
    Flux.loadparams!(forward, weights[1])
    Flux.loadparams!(backward, weights[2])
    Flux.loadparams!(embedding, weights[3])
    Flux.loadparams!(predictor, weights[4])
end

"""
    find_nonconf_vals(dl_calib, model)

Finds nonconformity scores for all points in the calibration dataloader.

Parameter dl_calib (Dataloader) - Calibration dataloader.
Parameter model (trained model) - Trained model for set prediction.

Return α_i (Vector{Float64}) - Sorted nonconformity scores for points in
    dl_calib.
"""
function find_nonconf_vals(dl_calib, model)
    α_i = Vector{Float64}()
    α = 0.0
    cor = 0.0
    for (x, y) in dl_calib
        cor = maximum(y .* model(x))
        α = 1 - cor
        push!(α_i, α)
    end
    sort!(α_i, rev = true)
    return α_i
end

"""
    test_blstm(dl_test, model, α_i)

Finds the p-values for the points in the testing set and the corresponding
    onehot classes.

Parameter dl_test (Dataloader) - Testing dataloader.
Parameter model (trained model) - Trained model for set prediction.
Parameter α_i (Vector{Float64}) - Sorted nonconformity scores for points in
    dl_calib.

Return correct_onehot (Vector{Any}) - Vector of onehots for each point in
    dl_test.
Return pvals (Vector{Any}) - Vector of p-values for each point in dl_test.
"""
function test_blstm(dl_test, model, α_i)
    correct_onehot = []
    pvals = []
    amount_right = 0
    for (x, y) in dl_test
        V = model(x)
        push!(pvals, toPval(V, α_i))
        Y = []
        for i in y
            push!(Y, i)
        end
        push!(correct_onehot, Y)
    end
    return correct_onehot, pvals
end

"""
    empirical_proposed(pvals, actWords)

Finds the empirical and proposed confidence levels for epsilons between .01 -
    .50.

Parameter pvals (Vector{Any}) - Vector of p-values for each point in dl_test.
Parameter actWords (Vector{Any}) - Vector of onehots for each point in dl_test.

Return valid (Vector{Any}) - Vector of empirical confidence levels.
Return confs (Vector{Any}) - Vector of proposed confidence levels.
"""
function empirical_proposed(pvals, actWords)
    confs = []
    valid = []
    for i in 0:.05:.5
        if i == 0
            i = 0.01
        end
        a = mean(returnIndex.(pvals, argmax.(actWords)) .> i)
        conf = 1 - i
        push!(confs, conf)
        push!(valid, a)
    end
    return valid, confs
end

"""
    find_ACDS(pvals, args_mod, args_class)

Finds the ACDS performance criterion.

Parameter pvals (Vector{Any}) - Vector of p-values for each point in dl_test.
Parameter args_mod (Vector{Any}) - Vector of indices representing the model's
    prediction for the hidden words POS.
Parameter args_class (Vector{Any}) - Vector of the correct indices for the
    hidden words POS.

Return acds (Float64) - ACDS performance criterion.
"""
function find_ACDS(pvals, args_mod, args_class)
    oners = sum.(greatorVec.(pvals)) .== 1
    counter = 0
    for i in 1:length(oners)
        if oners[i] == 1
            if args_mod[i] == args_class[i]
                counter += 1
            end
        end
    end
    acds = counter / sum(oners)
    return acds
end

"""
    find_PIS(pvals)

Finds the PIS performance criterion.

Parameter pvals (Vector{Any}) - Vector of p-values for each point in dl_test.

Return PIS (Fleat64) - PIS performance criterion.
"""
function find_PIS(pvals)
    pis = sum(sum.(greatorVec.(pvals)) .>= 2)/length(pvals)
    return pis
end

"""
    find_OP_OF(pvals, actWords)

Finds the OP and OF performance criterion.

Parameter pvals (Vector{Any}) - Vector of p-values for each point in dl_test.
Parameter actWords (Vector{Any}) - Vector of onehots for each point in dl_test.

Return OP (Float64) - OP performance criterion.
Return OF (Float64) - OF performance criterion.
"""
function find_OP_OF(pvals, actWords)
    OP = mean(dot.(pvals,actWords))
    OF = mean(dot.(pvals, notVec.(actWords)) / length(pvals[1]))
    return OP, OF
end

"""
    pred_class(dl_test, trained_model)

Finds the models point prediction for the chosen words POS and find the actual
    POS.

Parameter dl_test (Dataloader) - Testing dataloader.
Parameter trained_model (trained model) - Trained model for set prediction.

Return args_mod (Vector{Any}) - Vector of indices representing the model's
    prediction for the hidden words POS.
Return args_class (Vector{Any}) - Vector of the correct indices for the
    hidden words POS.
"""
function pred_class(dl_test, trained_model)
    args_mod = []
    args_class = []
    for (x, y) in dl_test
        push!(args_mod, argmax(trained_model(x))[1])
        push!(args_class, argmax(y)[1])
    end
    return args_mod, args_class
end

"""
    toPval(scores,a_i)

Finds p-values given the nonconformity scores and a model prediction

Parameter scores (Vector{Any}) - Softmax POS prediction vector from trained
    model.
Parameter a_i (Vector{Float64}) - Nonconformity scores for calibration set.

Return pvaltemp (Vector{Any}) - P-values for given parameters.
"""
function toPval(scores, a_i)
    a_i = sort(a_i)
    L = length(a_i)
    pvaltemp = []
    for x in scores
        push!(pvaltemp,1-((searchsortedfirst(a_i, 1 - x)/length(a_i))))
    end
    return pvaltemp
end

"""
    greatorVec(vec)

Finds BitVector with values over epsilon being 1's and values under epsilon
    being 0's.

Paramter vec (Vector{Float64}) - Vector of Float64 values.

Return vec .> epsilon (BitVector) - Vector with values over epsilon being 1's
    and values under epsilon being 0's.
"""
function greatorVec(vec)
    return vec .> epsilon
end

"""
    returnIndex(vec, ind)

Finds the value of a vector at any given index.

Paramter vec (Vector{Any}) - Vector of any values.
Paramter ind (Int64) - Index of vector.

Return vec[ind] (Any) - Value of vector at index ind.
"""
function returnIndex(vec, ind)
    return vec[ind]
end

"""
    notVec(vec)

Finds vector with every value in the vector being the absolute value of 1 minus
    the original value.

Paramter vec (Vector{Float64}) - Vector of Float64 values.

Return abs.(1 .- vec) (Vector{Float64}) - Vector with every value in the vector
    being the absolute value of 1 minus the original value.
"""
function notVec(vec)
    return abs.(1 .- vec)
end
