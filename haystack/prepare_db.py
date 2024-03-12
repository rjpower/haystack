import annoy
import os
import itertools
import datasets
import numpy as np
import pandas as pd
import torch
import tqdm
import transformers

from .config import HaystackConfig

CACHE_DIR = "./cache"
DATASET = "allenai/peS2o"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 128
MAX_DOCUMENTS = 1000


def split(tokenizer, config, record):
    tokens = [tokenizer.tokenize(body) for body in record["text"]]
    token_ids = tokenizer(record["text"], return_tensors="np")

    num_docs = token_ids["input_ids"].shape[0]
    result = []
    for i in range(num_docs):
        for j in range(0, token_ids["input_ids"][i].shape[0], config.key_context):
            result.append(
                {
                    "key_tokens": np.array(tokens[i][j : j + config.key_context]),
                    "key_input_ids": token_ids["input_ids"][i][
                        j : j + config.key_context
                    ],
                    "key_token_type_ids": token_ids["token_type_ids"][i][
                        j : j + config.key_context
                    ],
                    "key_attention_mask": token_ids["attention_mask"][i][
                        j : j + config.key_context
                    ],
                    "value_tokens": np.array(tokens[i][j : j + config.value_context]),
                    "value_input_ids": token_ids["input_ids"][i][
                        j : j + config.value_context
                    ],
                    "value_token_type_ids": token_ids["token_type_ids"][i][
                        j : j + config.value_context
                    ],
                    "value_attention_mask": token_ids["attention_mask"][i][
                        j : j + config.value_context
                    ],
                }
            )

    df = pd.DataFrame(result)

    # iterate over columns,  convert to torch tensors, padding to the maximum length
    result_dict = df.to_dict(orient="list")

    for k, v in result_dict.items():
        if "_tokens" in k:
            result_dict[k] = v
            continue

        max_len = max(len(array) for array in v)
        # pad all tensors to max len and convert to 2d array
        result_dict[k] = np.array(
            [np.pad(array, (0, max_len - len(array))) for array in v]
        )

    return result_dict


def encode(tokenizer, model, config, record):
    splits = split(tokenizer, config, record)
    for k, v in splits.items():
        if "_tokens" in k:
            continue
        else:
            splits[k] = torch.tensor(v).to("cuda")

    with torch.no_grad():
        key_embedding = model(
            input_ids=splits["key_input_ids"],
            token_type_ids=splits["key_token_type_ids"],
            attention_mask=splits["key_attention_mask"],
        )[0][:, 0]

    key_embedding = key_embedding.detach().cpu().numpy()

    return {
        "key_embedding": key_embedding,
        "value_input_ids": splits["value_input_ids"].cpu().numpy(),
        "value_token_type_ids": splits["value_token_type_ids"].cpu().numpy(),
        "key_tokens": splits["key_tokens"],
        "value_tokens": splits["value_tokens"],
    }


def create_ann_and_value_db():
    config = HaystackConfig()
    model = transformers.AutoModel.from_pretrained(EMBED_MODEL)

    os.makedirs("data", exist_ok=True)

    if torch.cuda.is_available():
        model = model.to("cuda")
    else:
        model = model.to("mps")

    model.eval()

    ds = datasets.load_dataset(
        DATASET,
        split="train",
        cache_dir=CACHE_DIR,
        streaming=True,
    )
    DS_KEYS = list(next(iter(ds)).keys())

    tokenizer = transformers.AutoTokenizer.from_pretrained(EMBED_MODEL)
    encode_ds = ds.map(
        lambda record: encode(tokenizer, model, config, record),
        batch_size=BATCH_SIZE,
        batched=True,
        remove_columns=DS_KEYS,
    )

    encode_ds = encode_ds.take(MAX_DOCUMENTS)

    encode_iter = tqdm.tqdm(encode_ds, total=MAX_DOCUMENTS)
    index = annoy.AnnoyIndex(config.embedding_size, "angular")
    index.on_disk_build("data/keys.index")
    value_db = np.empty((MAX_DOCUMENTS, config.value_context), dtype=np.int32)

    for i, batch in enumerate(itertools.islice(encode_iter, MAX_DOCUMENTS)):
        if i % 100 == 0:
            title = " ".join(batch["key_tokens"])[:80]
            encode_iter.set_description(f"Indexing {i} {title}")

        value_db[i, :] = batch["value_input_ids"]
        index.add_item(i, batch["key_embedding"])

    np.save("data/value_tokens", value_db)
    del value_db

    index.build(10)


if __name__ == "__main__":
    create_ann_and_value_db()
