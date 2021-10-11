# -------------------- Importing all necessary packages -------------------- #

# Adding all necessary packages (correct versions)
using Pkg
Pkg.add(Pkg.PackageSpec(;name="JLD", version="0.12.3"))
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add(Pkg.PackageSpec(;name="Embeddings", version="0.4.2"))
Pkg.add(Pkg.PackageSpec(;name="Flux", version="0.12.1"))
Pkg.add("Random")
Pkg.add(Pkg.PackageSpec(;name="DataFrames", version="0.21.8"))
Pkg.add(Pkg.PackageSpec(;name="Plots", version="0.29.9"))
Pkg.add(Pkg.PackageSpec(;name="StatsBase", version="0.33.10"))
Pkg.add(Pkg.PackageSpec(;name="BSON", version="0.3.3"))
Pkg.add(Pkg.PackageSpec(;name="ProgressBars", version="1.4.0"))
Pkg.add(Pkg.PackageSpec(;name="CSV", version="0.8.4"))
Pkg.add(Pkg.PackageSpec(;name="TypedTables", version="1.4.0"))
Pkg.add(Pkg.PackageSpec(;name="Zygote", version="0.6.12"))
Pkg.add(Pkg.PackageSpec(;name="BisectPy", version="1.0.0"))

# Package using statements
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
using BisectPy

# -------------------------------- Functions -------------------------------- #

"""
    data_cleaner(raw_sentences)

Clean Brown Corpus by removing the tags for titles, headings, foreign words and
emphasized words. Reduce number of unique tags to 190.

Parameter raw_sentences (Array{Array{SubString{String},1},1}) - Array of all
    sentences where each word/tag combination in each sentence vector is stored
    as a string.

Return tagger (Array{Any,1}) - Vector of unique parts of speech stored as
    strings.
Return raw_words (Array{Any,1}) - Vector of all sentences where each word in
    each sentence vector is stored as a string.
Return raw_tags (Array{Any,1}) - Vector of all tags for each word in every
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

Parameter sentences (Array{Any,1}) - Vector of all sentences where
    each word in each sentence vector is stored as a string.

Return words (Array{String,1}) - Vector of every word in sentences, stored as
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
    sentence_embed(sentences, embtable)

Create a vector of vectors containing the word embeddings vectors for every
    word in every sentence.

Parameter sentences (Array{Any,1}) - Vector of all sentences where
    each word in each sentence vector is stored as a string.
Parameter embtable (Dict{String,Array{Float32,1}}) - Embeddings table.

Return vec_of_vec (Array{Array{Array{Float32,1},1},1}) - Vector of vectors
    containing the word embeddings vectors for every word in every sentence.
"""
function sentence_embed(sentences, embtable)
    vec_of_vec = []
    for i in sentences
        push!(vec_of_vec, get_embeddings(i, embtable))
    end
    vec_of_vec = convert(Vector{Vector{Vector{Float32}}}, vec_of_vec)
    return vec_of_vec
end

"""
    get_embeddings(sentence, embtable)

Find a vector of word embeddings for any given sentence.

Parameter sentence (Array{Any,1}) - Vector of a sentence where each word in the
    sentence is stored as a string.
Parameter embtable (Dict{String,Array{Float32,1}}) - Embeddings table.

Return sent_vec (Array{Any,1}) - Vector of word embedding vectors corresponding
    to each word in the given sentence.
"""
function get_embeddings(sentence, embtable)
    sent_vec = []
    for i in sentence
        push!(sent_vec, get(embtable, i, Vector{Float32}(zeros(300))))
    end
    return sent_vec
end

"""
    get_onehot(unique_pos, sentence_tags)

Create a vector of vectors containing the onehot POS vectors for every
    word in every sentence.

Parameter sentence_tags (Array{Any,1}) - Vector of vectors that contain the
    correct part of speech for each corresponding word in the "sentences"
    vector.
Parameter unique_pos (Array{Any,1}) - Vector containing all unique POS labels.

Return onehot (Array{Array{Array{Bool,1},1},1}) - Vector of vectors
    containing the onehot POS vectors for every word in every sentence.
"""
function get_onehot(unique_pos, sentence_tags)
    onehot = []
    for i in sentence_tags
        temp = []
        for ii in i
            push!(temp, Flux.onehot(ii, unique_pos))
        end
        push!(onehot, temp)
    end
    onehot = convert(Vector{Vector{Vector{Bool}}}, onehot)
    return onehot
end

"""
    BiLSTM(x)

Bidirectional Long Short-Term Memory layer.

Parameter x (Array{Array{Float32,1},1}) - Word embedding vector for a sentence.

Return final_pred (Array{Array{Float32,1},1}) - Vector of vectors of the
    forward LSTM predictions concatenated with backward LSTM predictions.
"""
function BiLSTM(x)
    Flux.reset!(forward)
    Flux.reset!(backward)
    fw_pred = forward.(x)
    bw_pred = reverse(backward.(reverse(x)))
    final_pred = vcat.(fw_pred, bw_pred)
    return final_pred
end

"""
    loss(x, y)

Loss function for model.

Parameter x (Array{Float32, 2}) - X value from dataloader.
Parameter y (Array{Float32, 2}) - Y value from dataloader.

Return l (Float64) - Loss value.
"""
function loss(x, y)
    l = 0
    for i in length(x)
        l += sum(Flux.Losses.crossentropy.(model(x[i]),y[i]))
    end
    return l
end

"""
    load_parameters!(weights)

Load weights from the weights vector that is saved after training model.

Parameter weights (Array{Any,1}) - Vector of updated parameters for each layer
    of the neural network.
"""
function load_parameters!(weights)
    Flux.loadparams!(forward, weights[1])
    Flux.loadparams!(backward, weights[2])
    Flux.loadparams!(predictor, weights[3])
end

"""
    find_nonconf_vals(calib, calib_class, model)

Find nonconformity scores for all points in the calibration set.

Parameter calib (Array{Array{Array{Float32,1},1},1}) - Calibration sentences.
Parameter calib_class (Array{Array{Array{Bool,1},1},1}) - Calibration onehots.
Parameter model (trained model) - Trained model for set prediction.

Return α_i (Array{Float32,1}) - Sorted nonconformity scores for points in
    the calibration set.
"""
function find_nonconf_vals(calib, calib_class, model)
    α_i = Vector{Float32}()
    calib_class_num = convert(Vector{Vector{Vector{Float32}}}, calib_class)
    for (x, y) in zip(calib, calib_class)
        mod_pred = copy(model(x))
        class = copy(y)
        cor = dot.(mod_pred, class)
        α = 1 .- cor
        append!(α_i, α)
    end
    sort!(α_i)

    return α_i
end

"""
    get_pvals(test, test_class, model, α_i)

Find the p-value for every possible POS in each words softmax vector. Also
    find the boolean onehots for every POS in each words softmax vector.

Parameter test (Array{Array{Array{Float32,1},1},1}) - Testing sentences.
Parameter test_class (Array{Array{Array{Bool,1},1},1}) - Testing onehots.
Parameter model (trained model) - Trained model for set prediction.

Return correct_onehot (Array{Any,1}) - Boolean onehots for every POS in each
    words softmax vector.
Reutrn pvals (Array{Any,1}) - P-values for every possible POS in each words
    softmax vector.
"""
function get_pvals(test, test_class, model, α_i)

    #=
    test_flat = []
    for sent in test
        for word in sent
            push!(test_flat, word)
        end
    end

    test_class_flat = []
    for sent in test_class
        for word in sent
            push!(test_class_flat, word)
        end
    end
    =#

    correct_onehot = []
    pvals = []
    amount_right = 0
    for (x, y) in zip(test, test_class)
        mod_pred = model(x)
        push!(pvals, pval_maker(α_i, mod_pred))
        class = []
        for i in y
            push!(class, i)
        end
        push!(correct_onehot, class)
    end
    return correct_onehot, pvals
end

"""
    pval_maker(nonconf, output)

Calculate p-values given the nonconformity scores and a model prediction.

Parameter nonconf (Array{Float32,1}) - Nonconformity scores for calibration set.
Parameter output (Array{Array{Float32,1},1}) - Softmax POS prediction vectors
    from trained model for any given sentence.

Return pVal (Array{Any,1}) - P-values for given POS prediction values for any
    given sentence.
"""
function pval_maker(nonconf, output)
    pVal = []
    L = length(nonconf)
    for i in 1:length(output)
        word_pval = []
        for ii in 1:length(output[i])
            push!(word_pval, 1-(bisect_left(nonconf, 1 - output[i][ii])/L))
        end
        push!(pVal, word_pval)
    end
    return pVal
end

"""
    pred_class(test, test_class, model)

Finds model's point prediction for each words POS and along with actual POS.

Parameter test (Array{Array{Array{Float32,1},1},1}) - Testing sentences.
Parameter test_class (Array{Array{Array{Bool,1},1},1}) - Testing onehots.
Parameter model (trained model) - Trained model for set prediction.

Return args_mod (Array{Any,1}) - Vector of indices representing the model's
    prediction for the hidden words POS.
Return args_class (Array{Any,1}) - Vector of the correct indices for the
    hidden words POS.
"""
function pred_class(test, test_class, model)

    args_mod_temp = []
    args_class_temp = []
    for (x, y) in zip(test, test_class)
        mod_pred = model(x)
        push!(args_mod_temp, argmax.(mod_pred))
        push!(args_class_temp, argmax.(y))
    end

    args_mod = []
    args_class = []
    for i in 1:length(args_mod_temp)
        for ii in 1:length(args_mod_temp[i])
            push!(args_mod, args_mod_temp[i][ii])
            push!(args_class, args_class_temp[i][ii])
        end
    end

    return args_mod, args_class
end

"""
    class_accuracy(act_pos, pvals)

Finds the classification accuracy of the trained model.

Parameter act_pos (Array{Any,1}) - Boolean onehots for every POS in each
    words softmax vector.
Parameter pvals (Array{Any,1}) - P-values for every possible POS in each words
    softmax vector.

Return CA (Float64) - Classification accuracy for model on testing set.
"""
function class_accuracy(act_pos, pvals)
    count = 0
    tot = 0
    for i in 1:length(pvals)
        count += sum(argmax.(act_pos[i]) .== argmax.(pvals[i]))
        tot += length(act_pos[i])
    end
    CA = count / tot
    return CA
end

"""
    credibility(pvals)

Finds the average credibility of model.

Parameter pvals (Array{Any,1}) - P-values for every possible POS in each words
    softmax vector.

Return mean_cred (Float64) - Average credibility of model.
"""
function credibility(pvals)
    cred = []
    for i in 1:length(pvals)
        push!(cred, mean(maximum.(pvals[i])))
    end
    mean_cred = mean(cred)
    return mean_cred
end

"""
    find_OP_OF(pvals, act_pos)

Finds the OP and OF performance criterion.

Parameter pvals (Array{Any,1}) - P-values for every possible POS in each words
    softmax vector.
Parameter act_pos (Array{Any,1}) - Boolean onehots for every POS in each
    words softmax vector.

Return OP (Float64) - OP performance criterion.
Return OF (Float64) - OF performance criterion.
"""
function find_OP_OF(pvals, act_pos)

    p = []
    a = []
    for i in 1:length(pvals)
        for ii in 1:length(pvals[i])
            push!(p, pvals[i][ii])
            push!(a, act_pos[i][ii])
        end
    end

    OP = mean(dot.(p,a))
    OF = (mean(dot.(p, notVec.(a)) / (length(p[1])))) * 190
    return OP, OF
end

"""
    nominal_conf(pvals, act_pos)

Finds the nominal confidence level at each significance level from 0.01 to 1.0,
    with a step size of 0.05.

Parameter pvals (Array{Any,1}) - P-values for every possible POS in each words
    softmax vector.
Parameter act_pos (Array{Any,1}) - Boolean onehots for every POS in each
    words softmax vector.

Return nom (Array{Any,1}) - Vector of nominal confidence levels at each
    significance level from 0.01 to 1.0, with a step size of 0.05.
"""
function nominal_conf(pvals, act_pos)

    nom = []
    for i in 0:0.05:1.0
        if i == 0.0
            i = 0.01
        end

        emp = empconf(pvals, act_pos, i)
        push!(nom, emp)
    end

    return nom
end

"""
    find_ACDS(pvals, args_mod, args_class)

Finds the ACDS performance criterion.

Parameter pvals (Array{Any,1}) - P-values for every possible POS in each words
    softmax vector.
Parameter args_mod (Array{Any,1}) - Vector of indices representing the model's
    prediction for the hidden words POS.
Parameter args_class (Array{Any,1}) - Vector of the correct indices for the
    hidden words POS.
Parameter epsilon (Float64) - Signifiance level for prediction set creation.

Return acds (Float64) - ACDS performance criterion.
"""
function find_ACDS(pvals, args_mod, args_class, epsilon)

    oners_temp = []
    for i in 1:length(pvals)
        push!(oners_temp, sum.(greatorVec.(pvals[i], epsilon)) .== 1)
    end

    oners = []
    for i in oners_temp
        for ii in i
            push!(oners, ii)
        end
    end

    counter = 0
    for i in 1:length(oners)
        if oners[i] == true
            if args_mod[i] == args_class[i]
                counter += 1
            end
        end
    end
    acds = counter / sum(oners)
    return acds
end

"""
    find_PIS(pvals, epsilon)

Finds the PIS performance criterion.

Parameter pvals (Array{Any,1}) - P-values for every possible POS in each words
    softmax vector.
Parameter epsilon (Float64) - Signifiance level for prediction set creation.

Return PIS (Float64) - PIS performance criterion.
"""
function find_PIS(pvals, epsilon)

    twos_temp = []
    for i in 1:length(pvals)
        push!(twos_temp, sum.(greatorVec.(pvals[i], epsilon)) .>= 2)
    end

    twos = []
    for i in twos_temp
        for ii in i
            push!(twos, ii)
        end
    end

    counter = 0
    for i in twos
        if i == true
            counter += 1
        end
    end
    return counter / length(twos)
end

"""
    avg_set_sizes(pvals, epsilon)

Finds the average set sizes for any given epsilon.

Parameter pvals (Array{Any,1}) - P-values for every possible POS in each words
    softmax vector.
Parameter epsilon (Float64) - Signifiance level for prediction set creation.

Return tot_sizes (Array{Any,1}) - Vector of all POS prediction set sizes for
    each word in the testing set.
Return n_crit (Float64) - N performance criterion, average set size.
"""
function avg_set_sizes(pvals, epsilon)

    sizes = []
    for i in 1:length(pvals)
        push!(sizes, sum.(greatorVec.(pvals[i], epsilon)))
    end

    tot_sizes = []
    for i in sizes
        for ii in i
            push!(tot_sizes, ii)
        end
    end
    n_crit = mean(tot_sizes)
    return tot_sizes, n_crit
end

"""
    empconf(pvals, act_pos, epsilon)

Finds the nominal confidence for the model at any given epsilon.

Parameter pvals (Array{Any,1}) - P-values for every possible POS in each words
    softmax vector.
Parameter act_pos (Array{Any,1}) - Boolean onehots for every POS in each
    words softmax vector.
Parameter epsilon (Float64) - Signifiance level for prediction set creation.

Return conf (Float64) - nominal confidence for the model at any given epsilon.
"""
function empconf(pvals, act_pos, epsilon)

    p = []
    a = []
    for i in 1:length(pvals)
        for ii in 1:length(pvals[i])
            push!(p, pvals[i][ii])
            push!(a, act_pos[i][ii])
        end
    end
    conf = mean(returnIndex.(p, argmax.(a)) .> epsilon)

    return conf
end

"""
    greatorVec(vec, epsilon)

Finds BitArray with values in any given vector that are over epsilon represented
    as 1's and values under epsilon as 0's.

Paramter vec (Array{Any,1}) - Vector of any values (in this case, numerical).
Parameter epsilon (Float64) - Signifiance level for prediction set creation.

Return comp (BitArray{1}) - Vector with values over epsilon being 1's
    and values under epsilon being 0's.
"""
function greatorVec(vec, epsilon)
    comp = vec .> epsilon
    return comp
end

"""
    returnIndex(vec, ind)

Finds the value of a vector at any given index.

Paramter vec (Array{Any,1}) - Vector of any values (in this case, numerical).
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

Paramter vec (Array{Any,1}) - Vector of any values (in this case, numerical).

Return diff (Array{Any,1}) - Vector with every value in the vector
    being the absolute value of 1 minus the original value.
"""
function notVec(vec)
    diff = abs.(1 .- vec)
    return diff
end

"""
    get_nominal(nom_conf)

Finds the average nominal confidence (of all 5 splits) for each significance
    level as well as the proposed confidence.

Parameter nom_conf (Array{Any,1}) - Vector containing the nominal confidence
    level at each significance level from 0.01 to 1.0, with a step size of 0.05
    for all 5 splits.

Return nom_avg (Array{Float64,1}) - Average nominal confidence (of all 5 splits)
    for each significance level.
Return confs (Array{Float64,1}) - Proposed confidence for each significance
    level.
"""
function get_nominal(nom_conf)

    new = []
    for i in 1:length(nom_conf[1]) #1:11
        new_vec = []
        for ii in 1:length(nom_conf) #1:5
            push!(new_vec, nom_conf[ii][i])
        end
        push!(new, new_vec)
    end

    confs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
             0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    nom_avg = mean.(new)
    confs = reverse(confs)
    return nom_avg , confs
end

"""
    get_results(results)

Finds the average of each criterion across all 5 splits, as well as the 5th
    split's sorted set sizes for significance levels 0.01 and 0.001.

Parameter results (Array{Any,1}) - Vector of all results for the run file.

Return avg (Array{Float64,1}) - Average of each criterion across all 5 splits.
Return double_nine (Array{Any,1}) - 5th split's sorted set sizes for SL 0.01.
Return triple_nine (Array{Any,1}) - 5th split's sorted set sizes for SL 0.001.
"""
function get_results(results)

    new = []
    for i in 1:length(results[1]) #1:17
        new_vec = []
        for ii in 1:length(results) #1:5
            push!(new_vec, results[ii][i])
        end
        push!(new, new_vec)
    end

    newer_99 = new[end-1][5]
    newer_999 = new[end][5]

    double_nine = sort(newer_99)
    triple_nine = sort(newer_999)
    avg = mean.(new[1:16])
    return avg, double_nine, triple_nine
end

"""
    ranged_α(nonconf)

Find all nonconformity values that are less than or equal to 0.0002.

Parameter nonconf (Array{Float32,1}) - Calibration set nonconformity values.

Return ranger (Array{Any,1}) - Nonconformity values that are less than or equal
    to 0.0002.
"""
function ranged_α(nonconf)

    ranger = []
    for i in nonconf
        if i <+ .0002
            push!(ranger, i)
        end
    end
    return ranger
end
