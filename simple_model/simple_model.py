import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTConfig, GPT

def forward(w, tok, beta=1.0):

    # apply softmax to get probabilities
    probs = F.softmax(beta * tok @ w, dim=0)
    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1)

    # forward the GPT model itself
    tok_emb = model.transformer.wte(idx_next)  # token embeddings of shape (b, t, n_embd)
    pos_emb = model.transformer.wpe(torch.tensor([tok.size()[0]]))  # position embeddings of shape (t, n_embd)
    i_tok = tok_emb + pos_emb

    tok = torch.cat((tok, i_tok), dim=0)

    return tok, idx_next

def forward_mf(mu, cov, w, beta, t):

    mu_t = mu + beta * cov @ w

    mu_1_t = (mu * t + mu_t) / (t+1)

    return mu_t, mu_1_t



# Seed 1007 2 idxs
torch.manual_seed(1007)

# Load GPT2
device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
model_name = "gpt2"
model = GPT.from_pretrained(model_name, dict(dropout=0.0))
model.to(device)

beta = 10.0

# Get embeddings to sample tokens from them
wte = model.transformer.wte
num_embeddings = wte.num_embeddings
token_size = wte.embedding_dim

num_tokens_sample = 1000
idx = torch.randint(0, num_embeddings, (num_tokens_sample,))

pos = torch.arange(0, num_tokens_sample, dtype=torch.long, device=device)  # shape (t)

# Transform into token representation
tok_emb = model.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
pos_emb = model.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
tok_0 = tok_emb + pos_emb

# Create random w vector
w_mean = 0
w_std = 1
w = torch.normal(w_mean, w_std, (1, token_size))[0]

# Compute the average over multiple trajectories
num_runs = 10
num_running_steps = 5
tok_stats = torch.zeros((num_runs, num_running_steps, token_size))
idxs = torch.zeros((num_runs, num_running_steps))
for r in range(num_runs):

    tok = tok_0.clone()
    for i in range(num_running_steps):
        # Get the next token
        tok, idx_next = forward(w, tok, beta)
        # Accumulate stats
        tok_stats[r, i, :] += tok[-1, :]

        idxs[r, i] = idx_next


tok_stats_avg = torch.mean(tok_stats, dim=0)
print(idxs)
print()

# Now, compute the approximation
mu_1_t = torch.mean(tok_0, dim=0)
cov_1_t = torch.cov(tok_0.T)

tok_stats_mf = torch.zeros((num_running_steps, token_size))
for i in range(num_running_steps):
    mu_t, mu_1_t = forward_mf(mu_1_t, cov_1_t, w, beta, num_tokens_sample + i)

    tok_stats_mf[i, :] = mu_t

print(tok_stats - tok_stats_mf)
