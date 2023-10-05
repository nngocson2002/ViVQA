__DATASET__ = 'data/dataset.csv'
__VOCAB__ = 'data/vocab.json'
__IMAGES__ = 'data/images'
__FEATURES__ = 'data/features.npz'


embedding_dim = 300
image_size = 448
output_size = image_size // 32 # 2^5
num_features_output = 2048
central_fraction = 0.875