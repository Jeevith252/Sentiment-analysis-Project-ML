import pandas as pd
import numpy as np

import mlflow
import mlflow.transformers

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset


mlflow.set_experiment("Yelp_Sentiment_Analysis")


df = pd.read_csv("yelp.csv")

df = df[["text", "stars"]]

df = df[df["stars"] != 3]

df["label"] = df["stars"].apply(
    lambda x: 1 if x >= 4 else 0
)


train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )


train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)

test_dataset = test_dataset.rename_column(
    "label",
    "labels"
)

train_dataset = train_dataset.remove_columns(
    ["text", "stars", "__index_level_0__"]
)

test_dataset = test_dataset.remove_columns(
    ["text", "stars", "__index_level_0__"]
)


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary"
        )
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)


with mlflow.start_run():

    mlflow.log_param(
        "model_name",
        "distilbert-base-uncased"
    )

    mlflow.log_param(
        "epochs",
        3
    )

    mlflow.log_param(
        "learning_rate",
        2e-5
    )

    mlflow.log_param(
        "batch_size",
        8
    )

    trainer.train()

    metrics = trainer.evaluate()

    for key, value in metrics.items():

        if isinstance(
            value,
            (int, float)
        ):
            mlflow.log_metric(
                key,
                value
            )

    trainer.save_model(
        "sentiment_model"
    )

    tokenizer.save_pretrained(
        "sentiment_model"
    )

    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer
        },
        artifact_path="sentiment_model"
    )

    run_id = (
        mlflow.active_run()
        .info
        .run_id
    )

    model_uri = (
        f"runs:/{run_id}/sentiment_model"
    )

    mlflow.register_model(
        model_uri,
        "YelpSentimentModel"
    )

    print(metrics)
    print("Model Registered Successfully")