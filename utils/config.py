__DATASET_TRAIN__ = 'data/ViVQA-csv/train.csv'
__DATASET_TEST__ = 'data/ViVQA-csv/test.csv'
__VOCAB__ = 'data/vocab.json'
__IMAGES__ = 'data/images'

VISUAL_MODEL = {

    'CLIP-ViT': {
        'visual_features': 512,
        'feature_shape': (512,),
        'path': 'data/clip-vit.hdf5'
    },

    'Resnet152': {
        'visual_features': 2048,
        'feature_shape': (2048, 14, 14), # target_size = 448 / 2^5 = 14
        'path': 'data/resnet152.hdf5'
    }
}

TEXT_MODEL = {
    'PhoBert': {
        'text_features': 768
    }
}

max_answers = 353
batch_size = 128