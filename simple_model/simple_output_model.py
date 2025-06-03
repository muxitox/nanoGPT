import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTConfig, GPT

import matplotlib.pyplot as plt


def forward(tok_emb, W_k, q_t_1, beta=1.0):

    # apply softmax to get probabilities
    k =  tok_emb @ W_k
    probs = F.softmax(beta * k @ q_t_1, dim=0)
    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1)

    tok_t = tok_emb[idx_next][0]

    return tok_t, idx_next

def forward_mf(mu, cov, q_t_1, beta):

    mu_t = mu + beta * cov.T @ q_t_1

    return mu_t


#############
# Main start
#############
torch.manual_seed(1005)

# Load GPT2
device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
model_name = "gpt2"
model = GPT.from_pretrained(model_name, dict(dropout=0.0))
model.to(device)

beta = 1000

# Get embeddings to sample tokens from them
wte = model.transformer.wte
num_embeddings = wte.num_embeddings
token_size = wte.embedding_dim

# Select new_emb_size tokens from the embedding to create a smaller vocabulary
perm_idxs = torch.randperm(num_embeddings)
new_emb_size = 150
perm_idxs = perm_idxs[:new_emb_size]

# Transform into token representation
# Equivalent to x_{t+1} in the LaTeX
tok_emb = model.transformer.wte(perm_idxs).detach()


# Create random W patterns
w_mean = 0
w_std = 1/(token_size)
W_k = torch.normal(w_mean, w_std, (token_size, token_size))

q_is_roll = True
if q_is_roll:
    W_q = torch.roll(W_k, -1, 0)
else:
    W_q = torch.normal(w_mean, w_std, (token_size, token_size))

# Choose x_0
idx_0 = 1
tok_0 = tok_emb[idx_0]

# Get mean and covariances from the vocabulary for the approximation
emb_mu = torch.mean(tok_emb @ W_k, dim=0)
emb_cov = torch.cov((tok_emb @ W_k).T)


################################################
# Compute the average over multiple trajectories
################################################
num_runs = 1
num_running_steps = 1
tok_stats = torch.zeros((num_runs, num_running_steps, token_size))
tok_stats_mf = torch.zeros((num_running_steps, token_size))
idxs = torch.zeros((num_runs, num_running_steps))
for r in range(num_runs):

    tok_t_1 = tok_0.clone()
    for i in range(num_running_steps):
        # Pre compute q to share it with both methods
        q_t_1 = tok_t_1 @ W_q

        # Get the next token
        tok_t_1, idx_t_1 = forward(tok_emb, W_k, q_t_1, beta)

        # Accumulate stats
        tok_stats[r, i, :] += tok_t_1
        idxs[r, i] = idx_t_1

        #####
        # Compute approximation
        ########
        if r==0:
            mu_t = forward_mf(emb_mu, emb_cov, q_t_1, beta)
            tok_stats_mf[i, :] = mu_t



tok_stats_avg = torch.mean(tok_stats, dim=0)
print("Selected idxs at each trajectory")
print(idxs)
print()

####################################
# Statistics collection and plotting
####################################
rmse0 = torch.sqrt(torch.mean((tok_stats_avg - tok_stats_mf)**2))
print("Error wrt to selected token", rmse0)

rmse1 = torch.sqrt(torch.mean((tok_0[0] -  tok_stats_mf)**2))
print("Error wrt random token", rmse1)


step_to_plot = 0
feats_to_plot = 150
plt.plot(tok_stats_avg[step_to_plot][:feats_to_plot], label="Real Avg")
plt.plot(tok_stats_mf[step_to_plot][:feats_to_plot], label="MF")
plt.title(f"RMSE {rmse0}")
plt.legend()
plt.show()
plt.close()

plt.figure()
token_comparison_id = min(new_emb_size-1, 3545)
plt.plot(tok_emb[token_comparison_id][:feats_to_plot], label="Random token")
plt.plot(tok_stats_mf[step_to_plot][:feats_to_plot], label="MF")
plt.title(f"Error wrt random vector {rmse1}")
plt.legend()
plt.show()
plt.close()

fig, ax = plt.subplots(2, 2)
ax = ax.ravel()
feats_to_show = [3,5,103,108]
for i in range(0, len(feats_to_show)):
    ax[i].hist((tok_emb @ W_k)[:, feats_to_show[i]])
    ax[i].set_title(f"Feat {feats_to_show[i]}")
fig.show()
plt.close()