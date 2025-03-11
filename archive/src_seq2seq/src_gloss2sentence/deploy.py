from transformers import AutoModelForCausalLM, AutoTokenizer
import flask
import time
import threading

model_name = "stvlynn/Qwen-7B-Chat-Cantonese"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", trust_remote_code=True)

app = flask.Flask(__name__)

def update_timer():
    start_time = time.time()
    while not timer_event.is_set():
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds", end='\r')
        time.sleep(1)

@app.route("/combine", methods=["POST"])
def chat():
    data = flask.request.json
    # prompt = data["prompt"]
    # prompt = f'以下句子是從香港手語翻譯而成的，試將以下句子改為廣東話: 「{data["prompt"]}」'
    prompt = f'合併以下詞語為一句與時事有關的完整句子: {data["prompt"].replace(" ", ", ")}'
    
    global timer_event
    timer_event = threading.Event()
    timer_thread = threading.Thread(target=update_timer)
    timer_thread.start()

    try:
        response, _ = model.chat(tokenizer, prompt, history=None)
    finally:
        timer_event.set()
        timer_thread.join()
    
    return flask.jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000)