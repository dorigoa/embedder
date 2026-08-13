#!/usr/bin/env python3
# pip install sentence-transformers torch
import argparse
import sys
import torch
from logzero import logger, LogFormatter
import logzero
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
from transformers.utils import logging as hf_logging
hf_logging.disable_progress_bar()

import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

MODEL = {'mini'  : "paraphrase-multilingual-MiniLM-L12-v2",
         'qwen'  : "Qwen/Qwen3-Embedding-0.6B",
         'gemma' : 'google/embeddinggemma-300m',
         'mpnet' : 'paraphrase-multilingual-mpnet-base-v2',
         'bge'   : 'BAAI/bge-m3' }


# def pick_device(requested=None):
#     """Restituisce il device migliore disponibile, o quello richiesto se valido."""
#     if requested:
#         return requested
#     if torch.cuda.is_available():
#         return "cuda"
#     if torch.backends.mps.is_available():
#         return "mps"
#     return "cpu"


FMT = ('%(color)s[%(levelname)1.1s %(asctime)s.%(msecs)03d '
       '%(module)s:%(lineno)d]%(end_color)s %(message)s')
DATEFMT = '%y%m%d %H:%M:%S'

formatter = LogFormatter(fmt=FMT, datefmt=DATEFMT)
logzero.formatter(formatter)


def pick_device(requested=None):
    """Auto-detect del device che performa meglio.

    Nota: PyTorch per ROCm (AMD) espone le GPU tramite
    l'API 'cuda', quindi 'cuda' copre sia NVIDIA che AMD.
    """
    if requested:
        available = {
            "cuda": torch.cuda.is_available,
            "mps":  torch.backends.mps.is_available,
            "cpu":  lambda: True,
        }[requested]()
        if not available:
            raise RuntimeError(f"device '{requested}' requested is not available")
        return requested

    if torch.cuda.is_available():
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        logger.debug(f"GPU {backend}: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    p = argparse.ArgumentParser(
        usage="emb.py --words <w1[,w2,...]> --sample <parola>"
    )
    p.add_argument("-w", "--words", required=True,
                   help="Comma separated words to be ranked (corpus)")
    p.add_argument("-s", "--sample", required=True,
                   help="Word to match against (sample)")
    p.add_argument("-d", "--device", default=None, required=False, choices=['cpu','mps','cuda'],
                   help="Force the device on which execute math operations")
    p.add_argument("-m", "--model", default="mini", required=False, choices=MODEL.keys(),
                   help=f"Select the model.")
    args = p.parse_args()

    corpus = [w.strip() for w in args.words.split(",") if w.strip()]
    if not corpus:
        p.error("--words doesn't contain a valid list of command-separated words/sentences")

    try:
        device = pick_device(args.device)
    except RuntimeError as e:
        logger.error(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)
    
    logger.debug(f"Using device {device}")

    logger.debug("Importing transformer...")
    from sentence_transformers import SentenceTransformer
    logger.debug("Transformer imported.")

    try:
        logger.debug(f"Loading model {MODEL[args.model]}...")
        model = SentenceTransformer(MODEL[args.model], 
                                    device=device,
                                    processor_kwargs={"padding_side": "left"},)
        logger.debug(f"Model load completed.")
    except Exception as e:
        print(f"Errore nel caricamento del modello su '{device}': {e}",
              file=sys.stderr)
        return 1

    logger.debug("Torching...")
    with torch.inference_mode():
        # normalize_embeddings=True -> normalizzazione L2 fatta in torch, sul device
        # convert_to_tensor=True    -> nessun rientro in RAM CPU
        emb = model.encode(corpus, 
                           convert_to_tensor=True,
                           normalize_embeddings=True,
                           show_progress_bar=False)
        q = model.encode([args.sample], 
                         convert_to_tensor=True,
                         normalize_embeddings=True,
                         prompt_name="query",
                         show_progress_bar=False)
        sims = (emb @ q.T).squeeze(1)          # coseno, essendo già normalizzati
        vals, idx = torch.sort(sims, descending=True)

    # unico trasferimento verso CPU: solo per stampare
    logger.debug("Torch done.")
    for i, v in zip(idx.tolist(), vals.tolist()):
        print(f"{corpus[i]:20s} {v:+.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
