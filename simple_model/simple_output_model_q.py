import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from mpmath import arange
from numpy.f2py.auxfuncs import throw_error
from sympy import ceiling
from torch.nn import functional as F
from model import GPTConfig, GPT

import matplotlib.pyplot as plt

import time

def forward(tok_emb_t_1, x_t, W_k, W_q, beta=1.0):
    # apply softmax to get probabilities
    k_t_1 = tok_emb_t_1 @ W_k.T
    q_t = x_t @ W_q.T
    Wk_Wq_x_t = W_k.T @ q_t  # Compute this also for comparison

    probs = F.softmax(beta * k_t_1 @ q_t, dim=0)

    # sample from the distribution
    idx_next = torch.multinomial(probs, num_samples=1)

    tok_t = tok_emb_t_1[idx_next][0]

    return tok_t, idx_next, Wk_Wq_x_t, q_t

def forward_mf(mu, cov, tok_emb, W_k, W_q, beta):

    # Compute the probability of the text token
    k = tok_emb @ W_k.T
    p_k = beta * k @ mu + beta**2 * torch.einsum('ni,ij,nj->n', k, cov, k) / 2

    p_k = F.softmax(p_k, dim=0)

    mu_x = torch.sum(p_k[:, None] * tok_emb, dim=0)

    mu_k = mu_x @ W_k.T
    mu_q = mu_x @ W_q.T

    q = tok_emb @ W_q.T
    q_centered = q - mu_q
    sigma_q = torch.einsum('n,nm,nk->mk', p_k, q_centered, q_centered)

    return mu_k, mu_q, sigma_q

def plot_2_t_step(stats_1, stats_2, num_feats_to_plot, tag):
  rmse = torch.sqrt(torch.mean((stats_1 - stats_2)**2))


  plt.plot(stats_1[:num_feats_to_plot], label="Sim Avg")
  plt.plot(stats_2[:num_feats_to_plot], label="MF")
  plt.title(f"{tag} RMSE {rmse} t={t}")
  plt.legend()
  plt.show()

def plot_agg_t_step(gold_stat, other_stats, num_feats_to_plot, label, line, title):

    plt.plot(gold_stat[:num_feats_to_plot], label="Sim Avg")

    for i in range(len(other_stats)):
        rmse = torch.sqrt(torch.mean((gold_stat - other_stats[i])**2))
        plt.plot(other_stats[i][:num_feats_to_plot], line[i], label=f"{label[i]} E{rmse:.6f}")
        plt.xlabel("Patterns/Features")

    plt.title(title)
    plt.legend()
    plt.show()

def plot_error_traj(gold_stat, other_stats, label, line, title):

    for i in range(len(other_stats)):
        rmse = torch.sqrt(torch.mean((gold_stat - other_stats[i]) ** 2, dim=1))
        print(title)
        print(rmse)
        plt.plot(rmse, line[i], label=f"{label[i]}")

    plt.ylabel("RMSE")
    plt.xlabel("t")
    plt.title(title)
    plt.legend()
    plt.show()

def return_subplot(num_feats):
    if num_feats == 1:
        fig, ax = plt.subplots(1, 1)
    elif num_feats == 2:
        fig, ax = plt.subplots(2, 1)
    elif num_feats == 3:
        fig, ax = plt.subplots(3, 1)
    else:
        fig, ax = plt.subplots(min(4, math.floor((num_feats)/2)), 2)

    return fig, ax

def subplot_trajectories(stats, label, line, title, domain_label, random_feats=False, max_feats_show=4):

    T, num_feats = stats[0].shape
    num_feats_show = min(num_feats, max_feats_show)

    fig, ax = return_subplot(num_feats_show)
    ax = ax.ravel()

    if max_feats_show > 4 and random_feats:
        feats = torch.randperm(stats[0].shape[1])[:num_feats_show]
    else:
        feats = torch.arange(num_feats_show)

    for i in range(0, num_feats_show):
        for s in range(len(stats)):
            ax[i].plot(stats[s][:,feats[i]], line[s], label=label[s])

            ax[i].set_ylabel(f"${domain_label}_{{{feats[i]}}}$")
            ax[i].set_xlabel(f"$t$")

        if i==num_feats_show - 1:
            ax[i].legend()
    plt.suptitle(title)
    fig.show()
    plt.close()



#############
# Main start
#############


# Experiment settings
num_trials = 1000
num_running_steps = 50

##############
# Network settings
################
# In this case instead of initializing from random variables, use GPT2 embeddings
# as initialization (They are distributed in a Gaussian way for a large number of tokens).

# Load GPT2
device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
model_name = "gpt2"
model = GPT.from_pretrained(model_name, dict(dropout=0.0))
model.to(device)

beta = 10
binary_vars = False
q_is_roll = True  # If W_q a shift over W_k
patterns_from_vocab = False # If you draw samples from the token vocab to make the W patterns
emb_size = 500 # Number of tokens in the embedding. -1 if you want the full vocab size
num_patterns = 10  # Number of patterns in the W matrices. -1 if you want to match token_size
token_size = 250 # Number of features. If -1, match with the embedding token size, otherwise, force your selection.
random_ini_token = False
ini_token_idx = 0
seed = 150

torch.manual_seed(seed)


####################
# Set up the vocabulary and weights
####################

# Define the tokens or spins
if binary_vars:    # Binary variables

    if token_size == -1:
        raise Exception("token_size must be defined")
    if num_patterns == -1:
        raise Exception("num_patterns must be defined")
    if emb_size == -1:
        raise Exception("emb_size must be defined")

    tok_emb = (torch.rand((emb_size, token_size), device=device) > 0.5).float()
    tok_emb = tok_emb * 2 - 1
else:   # Tokens from GPT2

    # Get embeddings to sample tokens from them
    wte = model.transformer.wte
    num_embeddings = wte.num_embeddings

    if token_size==-1:
        token_size = wte.embedding_dim

    if emb_size == -1:
      emb_size = num_embeddings

    if num_patterns == -1:
      num_patterns = token_size

    # Select emb_size tokens from the embedding to create a smaller vocabulary
    perm_tokens_idxs = torch.randperm(num_embeddings)
    tokens_idxs = perm_tokens_idxs[:emb_size]

    # Transform into token representation
    # Equivalent to x_{t+1} in the LaTeX doc
    tok_emb = model.transformer.wte(tokens_idxs).detach()
    perm_feats_idxs = torch.randperm(wte.embedding_dim)
    tok_emb = tok_emb[:, perm_feats_idxs[:token_size]]

# Define the W patterns

if patterns_from_vocab:

    if num_patterns > emb_size:
        raise Exception("num_patterns > emb_size")

    k_patts = torch.randperm(emb_size)[:num_patterns]
    q_patts = torch.randperm(emb_size)[:num_patterns]

    W_k = tok_emb[k_patts]
    W_q = tok_emb[q_patts]

    if binary_vars:
        W_k = W_k / math.sqrt(token_size)
        W_q = W_q / math.sqrt(token_size)
    else:

        tok_mean = torch.mean(tok_emb, dim=None)
        tok_mean_b = torch.mean(tok_emb, dim=0)
        k_mean = torch.mean(tok_emb[k_patts], dim=None)
        q_mean = torch.mean(tok_emb[q_patts], dim=None)
        k_std = torch.std(tok_emb[k_patts], dim=None)
        q_std = torch.std(tok_emb[q_patts], dim=None)

        tok_emb_std = torch.std(tok_emb, dim=None)
        tok_emb_std_b = torch.std(tok_emb, dim=0)

        scaling = torch.sqrt(tok_emb_std_b**2 + tok_mean_b**2)
        # FPC = 1 - (num_patterns / emb_size)
        # W_k = tok_emb[k_patts] / scaling
        # W_q = tok_emb[q_patts] / scaling
        #
        # J = (1 / math.sqrt(token_size * num_patterns)) * (W_k.T @ W_q)

        W_k = W_k / (scaling * (token_size * num_patterns)**(1/4))
        W_q = W_q / (scaling * (token_size * num_patterns)**(1/4))


    J = (W_k.T @ W_q)
    J_var = torch.var(J, dim=None)
    print("J variance", J_var, 1/token_size)
    print()

else:
    # Create random W patterns
    w_mean = 0
    # w_std = 1/math.sqrt(token_size)
    w_std = 1
    # num_patterns = token_size
    W_k = torch.normal(w_mean, w_std, (num_patterns, token_size))
    W_q = torch.normal(w_mean, w_std, (num_patterns, token_size))

    if binary_vars and not patterns_from_vocab:
        raise Exception("Behavior not programmed yet")

if q_is_roll:
    W_q = torch.roll(W_k, 1, 0)



# Choose initial token
# Choose x_0
if random_ini_token:
    tok_0 = torch.randn((emb_size, token_size))[ini_token_idx]
else:
    # tok_0 = tok_emb[ini_token_idx]
    tok_0 = W_q[ini_token_idx]


################################################
# Compute the average over multiple trajectories (without approximation)
################################################

# Variables to save statistics
tok_stats = torch.zeros((num_trials, num_running_steps, token_size))
idxs = torch.zeros((num_trials, num_running_steps))

startt = time.time()
for r in range(num_trials):

    if r % 100 == 0:
        print(f"Trial {r+1}/{num_trials}")

    tok_t_1 = tok_0.clone()
    for t in range(num_running_steps):

        # Get the next token
        tok_t_1, idx_t_1, Wk_Wq_x_t, q_t = forward(tok_emb, tok_t_1, W_k, W_q, beta)

        # Accumulate stats
        tok_stats[r, t, :] += tok_t_1
        idxs[r, t] = idx_t_1

endt = time.time()
time_elapsed = endt - startt
print(f"Time elapsed: {time_elapsed:.2f} seconds")

print("Selected tokens at each step")
print(idxs[0])
print(idxs[3])
print(idxs[100])
print(idxs[-1])



tok_stats_avg = torch.mean(tok_stats, dim=0)


# Plot average statistics of the original model

tag = "k"
if patterns_from_vocab:
    patterns_str = "VocabPatts"
else:
    patterns_str = "GaussianPatts"

stats = [tok_stats_avg @ W_k.T]
stats = [stat[num_patterns:num_patterns*3] for stat in stats]
label = ["Ori"]
line = ["-"]
title = rf"{tag} $\beta$={beta} n_trials={num_trials} {patterns_str} N={token_size} M={num_patterns} V={emb_size}"
subplot_trajectories(stats, label, line, title, "k", random_feats=False, max_feats_show=3)


#####
# Compute the approximation
########


mu_q_t_1 = tok_0 @ W_q.T
cov_q_t_1 = torch.zeros((num_patterns, num_patterns))

stats_k_mf = torch.zeros((num_running_steps, num_patterns))
for t in range(num_running_steps):

    mu_k, mu_q_t_1, cov_q_t_1 = forward_mf(mu_q_t_1, cov_q_t_1, tok_emb, W_k, W_q, beta)

    stats_k_mf[t] = mu_k

# Plot random trajectories
stats = [tok_stats_avg @ W_k.T,  stats_k_mf]
label = ["Sim", "MFk"]
line = ["-", "-."]
subplot_trajectories(stats, label, line, title, "k", random_feats=True)



# Show random k features
fig, ax = plt.subplots(2, 2)
ax = ax.ravel()
feats_perm = torch.randperm(num_patterns)
feats_to_show = feats_perm[:4]
for i in range(0, len(feats_to_show)):
    ax[i].hist((tok_emb @ W_k.T)[:, feats_to_show[i]])
    ax[i].set_title(f"Feat {feats_to_show[i]}")
fig.suptitle("k")
fig.show()
plt.close()
