import string
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

print('Loading the model')
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", cache_dir="../.cache")
print('Loading the tokenizer')
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

my_prompt = [string.ascii_letters] * 1000
my_prompt = " ".join(my_prompt)

print(f'Len of chars in prompt {len(my_prompt)}')

model = model.cuda()

input = tokenizer(
    my_prompt, truncation=False, return_tensors='pt',
)
input = {k: v.cuda() for k, v in input.items()}

print(f'I am the input keys: {input.keys()}')
print(f'I am the shape of input ids: {input["input_ids"].shape}')

output = model(**input)

actual_input = tokenizer.batch_decode(input['input_ids'])

print(f'Actual input: {actual_input}')
print(f'Len of actual input: {len(actual_input[0])}')