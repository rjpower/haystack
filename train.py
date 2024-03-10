import os
from haystack import model


from transformers import AutoTokenizer

model = model.HaystackForCausalLM.from_pretrained("google/gemma-2b", token=os.environ['HF_ACCESS_TOKEN'])
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b", token=os.environ['HF_ACCESS_TOKEN'])

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
