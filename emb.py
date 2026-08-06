#!/usr/bin/env python3
# pip install sentence-transformers torch
import argparse
import sys
import torch
from sentence_transformers import SentenceTransformer

from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()

#import os
#os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL = "Qwen/Qwen3-Embedding-0.6B"

def pick_device(requested=None):
    """Restituisce il device migliore disponibile, o quello richiesto se valido."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    p = argparse.ArgumentParser(
        usage="emb.py --words <w1[,w2,...]> --sample <parola>"
    )
    p.add_argument("-W", "--words", required=True,
                   help="Lista di parole separate da virgola")
    p.add_argument("-s", "--sample", required=True,
                   help="Parola di confronto")
    p.add_argument("-d", "--device", default="mps", required=False, choices=['cpu','mps','cuda'],
                   help="Forza il device: cpu | mps | cuda (default: auto)")
    args = p.parse_args()

    corpus = [w.strip() for w in args.words.split(",") if w.strip()]
    if not corpus:
        p.error("--words non contiene parole valide")

    device = pick_device(args.device)
    try:
        model = SentenceTransformer(MODEL, 
                                    device=device,
                                    processor_kwargs={"padding_side": "left"},)
    except Exception as e:
        print(f"Errore nel caricamento del modello su '{device}': {e}",
              file=sys.stderr)
        return 1

    #print(f"device: {model.device}", file=sys.stderr)

    with torch.inference_mode():
        # normalize_embeddings=True -> normalizzazione L2 fatta in torch, sul device
        # convert_to_tensor=True    -> nessun rientro in RAM CPU
        emb = model.encode(corpus, 
                           convert_to_tensor=True,
                           normalize_embeddings=True)
        q = model.encode([args.sample], 
                         convert_to_tensor=True,
                         normalize_embeddings=True,
                         prompt_name="query")
        sims = (emb @ q.T).squeeze(1)          # coseno, essendo già normalizzati
        vals, idx = torch.sort(sims, descending=True)

    # unico trasferimento verso CPU: solo per stampare
    for i, v in zip(idx.tolist(), vals.tolist()):
        print(f"{corpus[i]:20s} {v:+.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
