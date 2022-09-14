from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("allenai/tk-instruct-3b-def", cache_dir="./.cache")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/tk-instruct-3b-def", cache_dir="./.cache")

input_ids = tokenizer.encode(
        "Definition: return the currency of the given country. Now complete the following example - Input: India. Output:", 
        return_tensors="pt")
output = model.generate(input_ids, max_length=10)
output = tokenizer.decode(output[0], skip_special_tokens=True)   # model should output 'Indian Rupee'
print(output)
