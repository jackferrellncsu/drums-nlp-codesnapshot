# Helper functions
include("routine_blstm.jl")

# Cleaned data
include("datagen_blstm.jl")

# Seed for random
Random.seed!(24)

# Loads in data from "datagen_blstm.jl"
brown_data = datagen_blstm_res.blstm_data
sent_tens_emb = get(brown_data, "sentence_tensor", 0)
onehot_vecs = get(brown_data, "onehots", 0)
unique_pos = get(brown_data, "unique_pos", 0)

# Splits masked sentence embeddings and their corresponding one hot matrices
# into test/train/calib
temp_train, test, temp_train_class, test_class = SampleMats(sent_tens_emb,
                                                                onehot_vecs)
train, calib, train_class, calib_class = SampleMats(temp_train,
                                                            temp_train_class)

# Creates DataLoader classes
dl_calib = Flux.Data.DataLoader((calib, calib_class))
dl_test = Flux.Data.DataLoader((test, test_class))
dl_train = Flux.Data.DataLoader((train, train_class),
                                    batchsize = 100, shuffle = true)

# Neural net architecture
forward = LSTM(300, 150)
backward = LSTM(300, 150)
embedding = Dense(300, 300)
predictor = Chain(Dropout(0.2, dims=1), Dense(300, 250, relu),
                                                Dense(250,190), softmax)

# Predicts embedding for chosen word
vectorizer(x) = embedding(BLSTM(x))

# Predicts POS tag based on vectorizer prediction
model(x) = predictor(vectorizer(x))

# Optimizer
opt = Flux.Optimiser(ExpDecay(0.01, 0.1, 1, 1e-4), RADAM())

# Model parameters
ps = Flux.params((forward, backward, embedding, predictor))

# Trains the neural net, tracks loss progression
epochs = 10
traceY = []
for i in ProgressBar(1:epochs)
    Flux.reset!(forward)
    Flux.reset!(backward)
    Flux.train!(loss, ps, dl_train, opt)
    for (x, y) in dl_train
        push!(traceY, loss(x, y))
        break
    end
end

# Stores parameters from trained model
weights = []
push!(weights, Flux.params(forward))
push!(weights, Flux.params(backward))
push!(weights, Flux.params(embedding))
push!(weights, Flux.params(predictor))

# -------- Ensuring that the model is recreated with updated params -------- #

# Sets the weights to the architecture
load_parameters!(weights)

# Rebuilds trained model
vectorizer(x) = embedding(BLSTM(x))
trained_model(x) = predictor(vectorizer(x))

# --------------------------------------------------------------------------- #
# -------------------------- Conformal Predictions -------------------------- #
# --------------------------------------------------------------------------- #

# Calculates nonconformity scores for trained model
α_i = find_nonconf_vals(dl_calib, trained_model)

# Finds the actual POS onehots for each word and the p-values
actWords, pvals = test_blstm(dl_test, trained_model, α_i)

# Finds model's prediction for chosen word's POS and actual POS for chosen word
args_mod, args_class = pred_class(dl_test, trained_model)

# -------------------------- Performance Criterion -------------------------- #

# Classification Accuracy
CA = sum(argmax.(actWords) .== argmax.(pvals)) / length(pvals)

# Credibility
cred = mean(maximum.(pvals))

# Observed Perceptiveness and Observed Fuzziness
OP, OF = find_OP_OF(pvals, actWords)

# ---------------------- Stats for 99% confidence sets ---------------------- #

global epsilon = .01

acds_99 = find_ACDS(pvals, args_mod, args_class)
pis_99 = find_PIS(pvals)
sizes_99 = sum.(greatorVec.(pvals))
ncrit_99 = mean(sizes_99)
empconf_99 = mean(returnIndex.(pvals, argmax.(actWords)) .> epsilon)


# ---------------------- Stats for 95% confidence sets ---------------------- #

global epsilon = .05

acds_95 = find_ACDS(pvals, args_mod, args_class)
pis_95 = find_PIS(pvals)
sizes_95 = sum.(greatorVec.(pvals))
ncrit_95 = mean(sizes_95)
empconf_95 = mean(returnIndex.(pvals, argmax.(actWords)) .> epsilon)

# ---------------------- Stats for 90% confidence sets ---------------------- #

global epsilon = .10

acds_90 = find_ACDS(pvals, args_mod, args_class)
pis_90 = find_PIS(pvals)
sizes_90 = sum.(greatorVec.(pvals))
ncrit_90 = mean(sizes_90)
empconf_90 = mean(returnIndex.(pvals, argmax.(actWords)) .> epsilon)

# --------------------- Saving all results for out file --------------------- #

plot_requirements = Dict("α_i" => α_i,
                         "pvals" => pvals,
                         "actWords" => actWords,
                         "sizes_99" => sizes_99,)

# Table 1 (blstm)
t1_data = [(conf_level = "99%", emp_conf = empconf_99, acds = acds_99,
                                                                pis = pis_99),
           (conf_level = "95%", emp_conf = empconf_95, acds = acds_95,
                                                                pis = pis_95),
           (conf_level = "90%", emp_conf = empconf_90, acds = acds_90,
                                                                pis = pis_90)]

# Table 2 (blstm)
t2_data = [(CA = CA, cred = cred, OP = OP, OF = OF, N99 = ncrit_99,
                                            N95 = ncrit_95, N90 = ncrit_90)]

JLD.save("out/out_blstm_results/run_blstm_output.jld",
                                 "plot_requirements", plot_requirements,
                                 "t1_data", t1_data,
                                 "t2_data", t2_data)
