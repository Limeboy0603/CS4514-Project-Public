import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sentence_transformers import SentenceTransformer

m = SentenceTransformer("shibing624/text2vec-base-chinese")
sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']

sentence_embeddings = m.encode(sentences)
print("Sentence embeddings:")
print(sentence_embeddings)

# print the similarity of the two sentences
from sklearn.metrics.pairwise import cosine_similarity
print("Similarity of the two sentences:")
print(cosine_similarity([sentence_embeddings[0]], [sentence_embeddings[1]]))