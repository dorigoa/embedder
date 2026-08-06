# pip install sentence-transformers numpy

import argparse
import numpy as np
from sentence_transformers import SentenceTransformer

usage = "ebmedder --words <word1[,word2[,word3...]]> --sample <sample word>"

parser = argparse.ArgumentParser(usage=usage)

parser.add_argument(
    "-W", "--words", type=str, dest="words", required=True, help="Mandatory argument: word to be embedded wrt the sample, can be a command separated list of words"
)
parser.add_argument(
    "-s", "--sample", type=str, dest="sample", required=True, help="Word to be compared"
)

args = parser.parse_args()

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

corpus = [w.strip() for w in args.words.split(',') if w.strip()]
if not corpus:
    parser.error("--words non contiene parole valide")

emb = model.encode(corpus, convert_to_numpy=True)

emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

print("shape:", emb.shape)                       # (5, 384)
print("norme:", np.linalg.norm(emb, axis=1))     # tutte ~1.0

q = model.encode([args.sample], convert_to_numpy=True)
q = q / np.linalg.norm(q)
sims = emb @ q.T                                  # similarità coseno, in [-1, 1]

ordine = np.argsort(-sims.ravel())
for i in ordine:
    print(f"{corpus[i]:20s} {sims[i,0]:.3f}")
