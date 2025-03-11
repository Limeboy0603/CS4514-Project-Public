import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import pandas as pd
import argparse
import ast
import jieba
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", type=str, default="./results/train_LSTM_weighted_32_512_1024_1/result.csv")
args = parser.parse_args()
result_file_path = args.result_file

result_file = pd.read_csv(result_file_path, delimiter=",")
pred_list = result_file["Decoded Sentence"].tolist()
ref_list = result_file["Target Sentence"].tolist()

pred_list = [ast.literal_eval(pred) for pred in pred_list]
ref_list = [ast.literal_eval(ref) for ref in ref_list]

# remove all "<BAD>" tokens
for i in range(len(pred_list)):
    pred_list[i] = [word for word in pred_list[i] if word != "<BAD>"]
    ref_list[i] = [word for word in ref_list[i] if word != "<BAD>"]

# print(pred_list)
# print(ref_list)

# # Calculate BLEU score
# bleu_score = corpus_bleu(ref_list, pred_list)
# print("List BLEU Score: ", bleu_score)

split_file = pd.read_csv("./dataset/tvb-hksl-news/split/test.csv", delimiter="|") 
split_sentences = split_file["words"].tolist()
split_sentences = [jieba.lcut(sentence) for sentence in split_sentences]

eval_data = []

for i in range(len(split_sentences)):
    split_sentence = split_sentences[i]
    pred_sentence = pred_list[i]
    ref_sentence = ref_list[i]

    # remove all "<BAD>" tokens
    split_sentence = [word for word in split_sentence if word != "<BAD>"]
    pred_sentence = [word for word in pred_sentence if word != "<BAD>"]
    ref_sentence = [word for word in ref_sentence if word != "<BAD>"]

    # Calculate BLEU score
    list_bleu_score = sentence_bleu([ref_sentence], pred_sentence)

    print("Split Sentence: ", split_sentence)
    print("Pred Sentence: ", pred_sentence)
    print("Ref Sentence: ", ref_sentence)
    print("List BLEU Score: ", list_bleu_score)

    # Generate sentence
    pred_joined = " ".join(pred_sentence)
    post_obj = {"prompt": pred_joined}
    # response = requests.post("172.17.0.1:5000", json=post_obj)
    response = requests.post("http://127.0.0.1:5001/combine", json=post_obj)
    gen_sentence = response.json()["response"]
    gen_sentence_tokenized = jieba.lcut(gen_sentence)
    sentence_bleu_score = sentence_bleu([ref_sentence], gen_sentence_tokenized)

    eval_data.append({
        "split_sentence": split_sentence,
        "pred_sentence": pred_sentence,
        "ref_sentence": ref_sentence,
        "list_bleu_score": list_bleu_score,
        "gen_sentence": gen_sentence,
        "gen_sentence_tokenized": gen_sentence_tokenized,
        "sentence_bleu_score": sentence_bleu_score
    })

# export eval_data to csv
eval_data_df = pd.DataFrame(eval_data)
eval_data_df.to_csv("./results/eval_data.csv", index=False)
