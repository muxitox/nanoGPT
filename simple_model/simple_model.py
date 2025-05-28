import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTConfig, GPT

def forward(w, toks):



    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(w @ toks, dim=-1)
    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1)
    # append sampled index to the running sequence and continue
    idx = torch.cat((idx, idx_next), dim=1)



if __name__ == "__name__":
    device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.

    model_name = "gpt2"
    model = GPT.from_pretrained(model_name, dict(dropout=0.0))

    model.to(device)

    # Get embeddings to sample tokens from them
    wte = model.transformer.wte
    emb_size = wte.embedding.weights.size[0]

    num_tokens_sample = 1000
    idx = torch.randint(0, emb_size, (num_tokens_sample,))

    pos = torch.arange(0, num_tokens_sample, dtype=torch.long, device=device)  # shape (t)

    # forward the GPT model itself
    tok_emb = model.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
    pos_emb = model.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)

    tok_0 = tok_emb + pos_emb

    num_running_steps = 10
    for i in range(num_running_steps):
        forward()