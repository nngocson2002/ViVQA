__DATASET__ = 'data/dataset.csv'
__VOCAB__ = 'data/vocab.json'
__IMAGES__ = 'data/images'
__FEATURES__ = 'data/features.npz'


embedding_dim = 300
image_size = 448
output_size = image_size // 32 # 2^5
num_features_output = 2048
central_fraction = 0.875

question_features = 1024
num_attention_maps = 2
max_answers = 353
max_vocab_size = 3592

#train
batch_size = 128