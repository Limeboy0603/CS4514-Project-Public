from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

tokenizer = AutoTokenizer.from_pretrained("hon9kon9ize/yi-1.5-6b-yub-vocab-expanded")
model = AutoModel.from_pretrained("hon9kon9ize/yi-1.5-6b-yub-vocab-expanded")

prompt = "請用以下關鍵詞 按照關鍵詞順序以及邏輯 組成句子: 消息 說 他 香港 身份證 沒 但是 美國 護照"

print("Encoding prompt...")
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cpu")
print("Generating output...")
output = model.generate(input_ids, max_length=100, num_return_sequences=1, repetition_penalty=1.1, do_sample=True, top_k=50, top_p=0.95)
print("Decoding output...")
output = tokenizer.decode(output[0], skip_special_tokens=False)

print(output)