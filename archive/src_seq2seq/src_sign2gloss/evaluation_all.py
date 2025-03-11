import nltk
from nltk.translate.bleu_score import corpus_bleu
import pandas as pd
import ast

result_file = pd.read_csv("./results/eval_data.csv")
gen_sentence_tokenized = result_file["gen_sentence_tokenized"].tolist()
split_sentence = result_file["split_sentence"].tolist()

gen_sentence_tokenized = [ast.literal_eval(gen) for gen in gen_sentence_tokenized]
split_sentence = [ast.literal_eval(ref) for ref in split_sentence]

# remove all "<BAD>" tokens
for i in range(len(gen_sentence_tokenized)):
    gen_sentence_tokenized[i] = [word for word in gen_sentence_tokenized[i] if word != "<BAD>"]
    split_sentence[i] = [word for word in split_sentence[i] if word != "<BAD>"]

print(gen_sentence_tokenized)
# print(ref_sentence)

# Calculate BLEU score
bleu_score = corpus_bleu(split_sentence, gen_sentence_tokenized)
print("List BLEU Score: ", bleu_score)