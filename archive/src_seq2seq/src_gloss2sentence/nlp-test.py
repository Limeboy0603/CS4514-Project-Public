from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "stvlynn/Qwen-7B-Chat-Cantonese"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True)


prompt = f'合併以下詞語為一句與時事有關的完整句子: {"今 溫度 一 九 濕 百分比 七 四".replace(" ", ", ")}'.to("cuda")
response, _ = model.chat(tokenizer, prompt, history=None)

print(response)