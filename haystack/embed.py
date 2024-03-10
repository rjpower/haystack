from angle_emb import AnglE, Prompts

KEY_WINDOW = 32
VALUE_WINDOW = 32

# Load the embedding model and scan across the dataset.  Like Retro, we compose
# a key based on `KEY_WINDOW` tokens and a value based on `VALUE_WINDOW` tokens.
# We slide the window across the dataset and construct an embedding database for
# each key-value pair.
angle = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy="cls").cuda()
angle.set_prompt(prompt=Prompts.C)

# We will load the dataset and use faiss to index the embeddings. During training
# we will use a softmax lookup in order to allow gradient flow through the model.
vec = angle.encode({"text": "hello world"}, to_numpy=True)
