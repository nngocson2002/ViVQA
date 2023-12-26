# Descriptions
```
usage: main.py [-h] [--output-dir OUTPUT_DIR]
               [--log-level {debug,info,warning,error,critical,passive}]
               [--lr-scheduler-type {cosine,linear}] [--warmup-ratio WARMUP_RATIO]
               [--logging-strategy {no,epoch,steps}] [--save-strategy {no,epoch,steps}]
               [--save-total-limit SAVE_TOTAL_LIMIT] [-tb TRAIN_BATCH_SIZE] [-eb EVAL_BATCH_SIZE]
               [-e EPOCHS] [-lr LEARNING_RATE] [--weight-decay WEIGHT_DECAY] [--workers WORKERS]
               [--image-path IMAGE_PATH] [--ans-path ANS_PATH] [--train-path TRAIN_PATH]
               [--test-path TEST_PATH] [--drop-path-rate DROP_PATH_RATE] [--classes CLASSES]

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
  --drop-path-rate DROP_PATH_RATE
  --classes CLASSES
```
# Example
Change directory.
```bash
cd beit3/UIT
```
Run training.
```bash
python main.py --log-level 'info'\
               --image-path './data/images' \
               --train-path './data/ViVQA-csv/train.csv'\
               --test-path './data/ViVQA-csv/test.csv' \
               --ans-path './data/vocab.json'\
               --epoch 30
```