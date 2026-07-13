import os
import boto3

BUCKET = "jeevith-sentiment-model-2026"

FILES = [
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training_args.bin",
    "vocab.txt"
]

os.makedirs("sentiment_model", exist_ok=True)

s3 = boto3.client("s3")

for file in FILES:

    path = f"sentiment_model/{file}"

    if not os.path.exists(path):

        print(f"Downloading {file}")

        s3.download_file(
            BUCKET,
            file,
            path
        )

print("Model Downloaded")