# Descriptions
```
usage: main.py [-h] [--output-dir OUTPUT_DIR]
               [--log-level {debug,info,warning,error,critical,passive}]
               [--lr-scheduler-type {cosine,linear}]
               [--warmup-ratio WARMUP_RATIO]
               [--logging-strategy {no,epoch,steps}]
               [--save-strategy {no,epoch,steps}]
               [--save-total-limit SAVE_TOTAL_LIMIT] [-tb TRAIN_BATCH_SIZE]
               [-eb EVAL_BATCH_SIZE] [-e EPOCHS] [-lr LEARNING_RATE]
               [--weight-decay WEIGHT_DECAY] [--workers WORKERS]
               [--image-path IMAGE_PATH] [--ans-path ANS_PATH]
               [--train-path TRAIN_PATH] [--test-path TEST_PATH]
               [--feature-paths FEATURE_PATHS]
               [--efficientnet-b {0,1,2,3,4,5,6,7}]
               [--drop-path-rate DROP_PATH_RATE]
               [--encoder-layers ENCODER_LAYERS]
               [--encoder-attention-heads-layers ENCODER_ATTENTION_HEADS_LAYERS]
               [--classes CLASSES]

options:
  -h, --help            show this help message and exit
  --output-dir OUTPUT_DIR
  --log-level {debug,info,warning,error,critical,passive}
  --lr-scheduler-type {cosine,linear}
  --warmup-ratio WARMUP_RATIO
  --logging-strategy {no,epoch,steps}
  --save-strategy {no,epoch,steps}
  --save-total-limit SAVE_TOTAL_LIMIT
  -tb TRAIN_BATCH_SIZE, --train-batch-size TRAIN_BATCH_SIZE
  -eb EVAL_BATCH_SIZE, --eval-batch-size EVAL_BATCH_SIZE
  -e EPOCHS, --epochs EPOCHS
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
  --weight-decay WEIGHT_DECAY
  --workers WORKERS
  --image-path IMAGE_PATH
  --ans-path ANS_PATH
  --train-path TRAIN_PATH
  --test-path TEST_PATH
  --feature-paths FEATURE_PATHS
  --efficientnet-b {0,1,2,3,4,5,6,7}
  --drop-path-rate DROP_PATH_RATE
  --encoder-layers ENCODER_LAYERS
  --encoder-attention-heads-layers ENCODER_ATTENTION_HEADS_LAYERS
  --classes CLASSES
```
# Example
Change directory.
```bash
cd beit3/HCMUS
```
Install.
```bash
pip install salesforce-lavis
pip install torchscale underthesea mlflow efficientnet_pytorch
```
Run training.
```bash
!python main.py --log-level 'info'\
                --image-path './data/images' \
                --train-path './data/ViVQA-csv/train.csv'\
                --test-path './data/ViVQA-csv/test.csv' \
                --ans-path './data/vocab.json'\
                --feature-paths './features'\
                --output-dir './output'\
                --epoch 15
```