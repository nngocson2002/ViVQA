from transformers import TrainingArguments, Trainer
import mlflow
import argparse
import os
import glob
import numpy as np
from timm.models import create_model
from sklearn.metrics import accuracy_score
import model
from dataset import get_dataset
from extract_vision import extract_features
from VisionEmbedding import Blip2ViTExtractor, EfficientnetExtractor

os.environ['MLFLOW_EXPERIMENT_NAME'] = 'mlflow-vivqa'

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}

def get_options():
    args = argparse.ArgumentParser()

    # Training Argument
    args.add_argument("--output-dir", type=str, default="./output")
    args.add_argument("--log-level", choices=["debug", "info", "warning", "error", "critical", "passive"], default="passive")
    args.add_argument("--lr-scheduler-type", choices=["cosine", "linear"], default="cosine")
    args.add_argument("--warmup-ratio", type=float, default=0.1)
    args.add_argument("--logging-strategy", choices=["no", "epoch", "steps"], default="epoch")
    args.add_argument("--save-strategy", choices=["no", "epoch", "steps"], default="epoch")
    args.add_argument("--save-total-limit", type=int, default=1)
    args.add_argument("-tb", "--train-batch-size", type=int, default=75)
    args.add_argument("-eb", "--eval-batch-size", type=int, default=75)
    args.add_argument("-e", "--epochs", type=int, default=15)
    args.add_argument("-lr", "--learning-rate", type=float, default=3e-5)
    args.add_argument("--weight-decay", type=float, default=0.01)
    args.add_argument("--workers", type=int, default=2)

    # Varriables setting
    args.add_argument("--image-path", type=str, default="./data/images")
    args.add_argument("--ans-path", type=str, default="./data/vocab.json")
    args.add_argument("--train-path", type=str, default="./data/ViVQA-csv/train.csv")
    args.add_argument("--test-path", type=str, default="./data/ViVQA-csv/train.csv")
    args.add_argument("--feature-paths", type=str, default="./features")


    # Model setting
    args.add_argument("--efficientnet-b", choices=[0, 1, 2, 3, 4, 5, 6, 7], default=7)
    args.add_argument("--drop-path-rate", type=float, default=0.3)
    args.add_argument("--encoder-layers", type=int, default=4)
    args.add_argument("--encoder-attention-heads-layers", type=int, default=4)
    args.add_argument("--classes", type=int, default=353)

    opt = args.parse_args()
    return opt

def main():
    opt = get_options()

    if not os.path.exists(f'{opt.feature_paths}'):
        os.makedirs(f'{opt.feature_paths}')

    print('Extracting image features...')
    print('*****Extract features from Blip-2*****')
    vision_embed = Blip2ViTExtractor()
    if not os.path.exists(f'{opt.feature_paths}/{vision_embed.model_name}.hdf5'):
        extract_features(vision_embed, opt, model_name=vision_embed.model_name)
    else:
        print('Features from Blip-2 already exist!')

    print(f'*****Extract features from Efficientnet-{opt.efficientnet_b}*****')
    vision_embed = EfficientnetExtractor(model_name=f'efficientnet-b{opt.efficientnet_b}')
    if not os.path.exists(f'{opt.feature_paths}/{vision_embed.model_name}.hdf5'):
        extract_features(vision_embed, opt, model_name=vision_embed.model_name)
    else:
        print(f'Features from Efficientnet-b{opt.efficientnet_b} already exist!')
    print('Extract completed!')
    
    feature_paths = glob.glob(os.path.join(opt.feature_paths, '*.hdf5'))

    train_dataset, test_dataset = get_dataset(opt, feature_paths)

    model = create_model('vivqa_model', 
                        num_classes=opt.classes, 
                        drop_path_rate=opt.drop_path_rate,
                        encoder_layers=opt.encoder_layers,
                        encoder_attention_heads=opt.encoder_attention_heads_layers)

    args = TrainingArguments(
        output_dir=opt.output_dir,
        log_level=opt.log_level,
        lr_scheduler_type=opt.lr_scheduler_type,
        warmup_ratio=opt.warmup_ratio,
        logging_strategy=opt.logging_strategy,
        save_strategy=opt.save_strategy,
        save_total_limit=opt.save_total_limit,
        per_device_train_batch_size=opt.train_batch_size,
        per_device_eval_batch_size=opt.train_batch_size,
        num_train_epochs=opt.epochs,
        learning_rate=opt.learning_rate,
        weight_decay=opt.weight_decay,
        dataloader_num_workers=opt.workers,
        report_to='mlflow'
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test = trainer.evaluate(test_dataset)
    print(f'Test Accuracy: {test["eval_accuracy"]}')

    mlflow.end_run()

if __name__ == '__main__':
    main()
