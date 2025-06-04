import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTConfig, GPT

import matplotlib.pyplot as plt


def forward(tok_emb_t1, x_t, W_k, W_q, beta=1.0):

    # apply softmax to get probabilities
    k = tok_emb_t1 @ W_k
    q_t = x_t @ W_q
    Wk_Wq_x_t = W_k @ q_t

    probs = F.softmax(beta * k @ q_t, dim=0)

    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1)

    tok_t = tok_emb_t1[idx_next][0]

    return tok_t, idx_next, Wk_Wq_x_t,  q_t

def forward_mf(mu, cov, x_t, beta):

    mu_t = mu + beta * cov.T @ x_t

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

beta = 1

# Get embeddings to sample tokens from them
wte = model.transformer.wte
num_embeddings = wte.num_embeddings
token_size = wte.embedding_dim

# Select new_emb_size tokens from the embedding to create a smaller vocabulary
perm_idxs = torch.randperm(num_embeddings)
new_emb_size = 1000
perm_idxs = perm_idxs[:new_emb_size]

# Transform into token representation
# Equivalent to x_{t+1} in the LaTeX
tok_emb = model.transformer.wte(perm_idxs).detach()


# Create random W patterns
w_mean = 0
w_std = 1/(token_size)
num_patterns = token_size
W_k = torch.normal(w_mean, w_std, (token_size, num_patterns))

q_is_roll = True
if q_is_roll:
    W_q = torch.roll(W_k, -1, 0)
else:
    W_q = torch.normal(w_mean, w_std, (token_size, token_size))

# Choose x_0
idx_0 = 1
tok_0 = tok_emb[idx_0]

# Get mean and covariances from the vocabulary for the approximation
embWk_mu = torch.mean(tok_emb @ W_k, dim=0)
embWk_cov = torch.cov((tok_emb @ W_k).T)

emb_mu = torch.mean(tok_emb, dim=0)
emb_cov = torch.cov((tok_emb).T)


################################################
# Compute the average over multiple trajectories
################################################
num_trials = 100
num_running_steps = 2
tok_stats = torch.zeros((num_trials, num_running_steps, token_size))
tok_stats_mf_Wk = torch.zeros((num_running_steps, num_patterns))
tok_stats_mf = torch.zeros((num_running_steps, token_size))
idxs = torch.zeros((num_trials, num_running_steps))
for r in range(num_trials):

    tok_t_1 = tok_0.clone()
    for t in range(num_running_steps):

        # Get the next token
        tok_t_1, idx_t_1, Wk_Wq_x_t, q_t = forward(tok_emb, tok_t_1, W_k, W_q, beta)

        # Accumulate stats
        tok_stats[r, t, :] += tok_t_1
        idxs[r, t] = idx_t_1


tok_stats_avg = torch.mean(tok_stats, dim=0)


#####
# Compute the approximation
########
q_t = tok_0 @ W_q
Wk_Wq_x_t = W_k @ q_t
for t in range(num_running_steps):
    muWk_t = forward_mf(embWk_mu, embWk_cov, q_t, beta)
    mu_t = forward_mf(emb_mu, emb_cov, Wk_Wq_x_t, beta)
    tok_stats_mf_Wk[t, :] = muWk_t
    tok_stats_mf[t, :] = mu_t

    q_t = tok_stats_avg[t] @ W_q
    Wk_Wq_x_t = W_k @ q_t



print("Selected idxs at each trajectory")
print(idxs)
print()

####################################
# Statistics collection and plotting
####################################
rmse0 = torch.sqrt(torch.mean((tok_stats_avg @ W_k - tok_stats_mf_Wk)**2))
print("Error (Wk) wrt to selected token", rmse0)


token_comparison_id = min(new_emb_size-1, 3545)
rmse1 = torch.sqrt(torch.mean((tok_emb[token_comparison_id] @ W_k -  tok_stats_mf_Wk)**2))
print("Error (Wk) wrt random token", rmse1)

rmse2 = torch.sqrt(torch.mean((tok_stats_avg - tok_stats_mf)**2))
print("Error (X) wrt to selected token", rmse2)


timesteps_to_plot = [0,1]

for t in timesteps_to_plot:
    step_to_plot = 0
    feats_to_plot = 150
    plt.plot((tok_stats_avg[t] @ W_k)[:feats_to_plot], label="K Real Avg")
    plt.plot(tok_stats_mf_Wk[t][:feats_to_plot], label="MF")
    plt.title(f"RMSE {rmse0} T={t}")
    plt.legend()
    plt.show()
    plt.close()

    # plt.figure()
    # plt.plot((tok_emb[token_comparison_id] @ W_k)[:feats_to_plot], label="K Random token")
    # plt.plot(tok_stats_mf_Wk[t][:feats_to_plot], label="MF")
    # plt.title(f"Error wrt random vector {rmse1}")
    # plt.legend()
    # plt.show()
    # plt.close()

    plt.plot(tok_stats_avg[t][:feats_to_plot], label="X Real Avg")
    plt.plot(tok_stats_mf[t][:feats_to_plot], label="MF")
    plt.title(f"RMSE {rmse2}, T={t}")
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