from transformers import TrainingArguments, Trainer
import mlflow
import argparse
import os
import numpy as np
from timm.models import create_model
from sklearn.metrics import accuracy_score
import bartphobeit
from dataset import get_dataset

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
    args.add_argument("-tb", "--train-batch-size", type=int, default=16)
    args.add_argument("-eb", "--eval-batch-size", type=int, default=16)
    args.add_argument("-e", "--epochs", type=int, default=30)
    args.add_argument("-lr", "--learning-rate", type=float, default=3e-5)
    args.add_argument("--weight-decay", type=float, default=0.01)
    args.add_argument("--workers", type=int, default=2)

    # Varriables setting
    args.add_argument("--image-path", type=str, default="./data/images")
    args.add_argument("--ans-path", type=str, default="./data/vocab.json")
    args.add_argument("--train-path", type=str, default="./data/ViVQA-csv/train.csv")
    args.add_argument("--test-path", type=str, default="./data/ViVQA-csv/train.csv")


    # Model setting
    args.add_argument("--drop-path-rate", type=float, default=0.4)
    args.add_argument("--classes", type=int, default=353)

    opt = args.parse_args()
    return opt

def main():
    opt = get_options()

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
        report_to='mlflow',
        save_safetensors=False
    )

    model = create_model('vivqa_model', num_classes=opt.classes, drop_path_rate=opt.drop_path_rate)
    train_dataset, test_dataset = get_dataset(opt)

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