module datagen_blstm_res

export blstm_data

# Helper functions
include("routine_blstm.jl")

# Seed for random
Random.seed!(24)

# Reads in Brown Corpus through CSV file
brown_df = CSV.read("Data/brown.csv", DataFrame)

# Takes the column of the dataframe that has each word in each sentence, along
# with each corresponding POS tag
brown_data = brown_df[4]

# Splits words in each sentence
raw_sentences = split.(brown_data, " ")

# Finds unique POS tags, setences split by words, and the tags for each sentence
unique_pos, sentences, sentence_tags = data_cleaner(raw_sentences)

# Creates a vector containing every word and finds the unique ones
words = get_word_vec(sentences)
unique_words = convert(Vector{String},unique(words))

# Finds embeddings for each unique word and creates embedding dictionsary
embeddings_glove = load_embeddings(GloVe{:en},4, keep_words=Set(unique_words))
embtable = Dict(word=>embeddings_glove.embeddings[:,ii] for (ii,word) in
                                            enumerate(embeddings_glove.vocab))

# Creates a tensor of word embeddings along with the modified sentences and tags
sent_tens, new_sent, new_tags = sent_embeddings(sentences, sentence_tags,
                                                            300, 180, embtable)

# Masks random word in each sentence
masked_word, masked_pos, masked_sent = word_masker(new_sent, new_tags)

# Finds indices of each masked word and fills tensor with -20.0's at those spots
mask_ind, sent_tens_emb = masked_embeddings(masked_sent, sent_tens, 300)

# Centers masked word in tensor and truncates length of sentence if necessary
sent_tens_emb = create_window(sent_tens_emb, 15)

# Creates one hot matrix for the tensor, regarding the POS of masked word
onehot_vecs = zeros(length(unique_pos), length(masked_pos))
for i in 1:length(masked_pos)
    onehot_vecs[:, i] = Flux.onehot(masked_pos[i], unique_pos)
end
onehot_vecs = convert(Array{Float32, 2}, onehot_vecs)

blstm_data = Dict("onehots" => onehot_vecs,
                           "sentence_tensor" => sent_tens_emb,
                           "unique_pos" => unique_pos)

end
