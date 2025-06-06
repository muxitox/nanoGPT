import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from model import GPTConfig, GPT

import matplotlib.pyplot as plt


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

def forward_mf(mu, cov, x_t, beta):

    mu_t = mu + beta * cov.T @ x_t

    return mu_t

def plot_2_t_step(stats_1, stats_2, num_feats_to_plot, tag):
  rmse = torch.sqrt(torch.mean((stats_1 - stats_2)**2))


  plt.plot(stats_1[:num_feats_to_plot], label="Real Avg")
  plt.plot(stats_2[:num_feats_to_plot], label="MF")
  plt.title(f"{tag} RMSE {rmse} t={t}")
  plt.legend()
  plt.show()

def plot_agg_t_step(gold_stat, other_stats, num_feats_to_plot, label, line, beta, t, tag):

  plt.plot(gold_stat[:num_feats_to_plot], label="Real Avg")

  for i in range(len(other_stats)):
    rmse = torch.sqrt(torch.mean((gold_stat - other_stats[i])**2))
    plt.plot(other_stats[i][:num_feats_to_plot], line[i], label=f"{label[i]} E{rmse:.6f}")
  plt.title(rf"{tag} $\beta$={beta} t={t}")
  plt.legend()
  plt.show()

#############
# Main start
#############


# Experiment settings
num_trials = 500
num_running_steps = 5

##############
# Network settings
################
# In this case instead of initializing from random varialbes, use GPT2 embeddings
# as initialization (They are distributed in a Gaussian way for a large number of tokens).

# Load GPT2
device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
model_name = "gpt2"
model = GPT.from_pretrained(model_name, dict(dropout=0.0))
model.to(device)

beta = 5
q_is_roll = True  # If W_q a shift over W_k
patterns_from_vocab = True # If you draw samples from the token vocab to make the W patterns
torch.manual_seed(1005)
emb_size = 1000  # Number of tokens in the embedding. -1 if you want the full vocab size
num_patterns = 768  # Number of patterns in the W matrices. -1 if you want to match token_size


# Set up the weights



# Get embeddings to sample tokens from them
wte = model.transformer.wte
num_embeddings = wte.num_embeddings
token_size = wte.embedding_dim

if emb_size == -1:
  emb_size = num_embeddings

if num_patterns == -1:
  num_patterns = token_size

# Select emb_size tokens from the embedding to create a smaller vocabulary
perm_idxs = torch.randperm(num_embeddings)
perm_idxs = perm_idxs[:emb_size]

# Transform into token representation
# Equivalent to x_{t+1} in the LaTeX doc
tok_emb = model.transformer.wte(perm_idxs).detach()

# Define the W patterns

if patterns_from_vocab:
  k_patts = torch.randperm(emb_size)[:num_patterns]
  q_patts = torch.randperm(emb_size)[:num_patterns]

  W_k = tok_emb[k_patts]
  W_q = tok_emb[q_patts]

else:
  # Create random W patterns
  w_mean = 0
  w_std = 1/(token_size)
  # num_patterns = token_size
  W_k = torch.normal(w_mean, w_std, (num_patterns, token_size))
  W_q = torch.normal(w_mean, w_std, (num_patterns, token_size))

if q_is_roll:
    W_q = torch.roll(W_k, 1, 0)



# Choose initial token
# Choose x_0
idx_0 = 1
tok_0 = tok_emb[idx_0]
tok_0 = W_q[idx_0]



################################################
# Compute the average over multiple trajectories (without approximation)
################################################

# Variables to save statistics
tok_stats = torch.zeros((num_trials, num_running_steps, token_size))
idxs = torch.zeros((num_trials, num_running_steps))

for r in range(num_trials):

    tok_t_1 = tok_0.clone()
    for t in range(num_running_steps):

        # Get the next token
        tok_t_1, idx_t_1, Wk_Wq_x_t, q_t = forward(tok_emb, tok_t_1, W_k, W_q, beta)

        # Accumulate stats
        tok_stats[r, t, :] += tok_t_1
        idxs[r, t] = idx_t_1


print(idxs[0])
tok_stats_avg = torch.mean(tok_stats, dim=0)



#####
# Compute the approximation
########


# Compute least squares to reverse the k approx
W_lsq, residuals, rank, s = torch.linalg.lstsq(tok_emb @ W_k.T, tok_emb, rcond=None, driver="gels")

# COmpute sufficient statistics
# Get mean and covariances from the vocabulary for the approximation
# For x
tok_emb_mu = torch.mean(tok_emb, dim=0)
tok_emb_cov = torch.cov((tok_emb).T)


# For k it's faster just projecting the statistics by the linearity
tok_emb_Wk_mu = tok_emb_mu @  W_k.T
tok_emb_Wk_cov =  W_k @ tok_emb_cov @ W_k.T


# Simulation  loop
tok_stats_mf_Wk_traj = torch.zeros((num_running_steps, num_patterns))
tok_stats_mf_Wk_mf_lsq = torch.zeros((num_running_steps, num_patterns))
tok_stats_mf_x_mf_lsq = torch.zeros((num_running_steps, token_size))
tok_stats_mf_traj = torch.zeros((num_running_steps, token_size))
tok_stats_mf_mf = torch.zeros((num_running_steps, token_size))


q_t_traj = tok_0 @ W_q.T
q_t_mf_lsq = tok_0 @ W_q.T
Wk_Wq_x_t_traj = W_k.T @ q_t
Wk_Wq_x_t_mf = W_k.T @ q_t
for t in range(num_running_steps):
    # For k
    # Prev step from average stats
    muWk_t_traj = forward_mf(tok_emb_Wk_mu, tok_emb_Wk_cov, q_t_traj, beta)
    # Prev step from mf and least squares prediction
    muWk_t_mf_lsq = forward_mf(tok_emb_Wk_mu, tok_emb_Wk_cov, q_t_mf_lsq, beta)
    # For x
    # Get x_{t-1} from the average statistics of the real experiment
    mu_t_traj = forward_mf(tok_emb_mu, tok_emb_cov, Wk_Wq_x_t_traj, beta)
      # Get x_{t-1} from the aproximated average
    mu_t_mf = forward_mf(tok_emb_mu, tok_emb_cov, Wk_Wq_x_t_mf, beta)

    # Save data
    tok_stats_mf_Wk_traj[t, :] = muWk_t_traj
    tok_stats_mf_Wk_mf_lsq[t, :] = muWk_t_mf_lsq
    tok_stats_mf_x_mf_lsq[t, :] = muWk_t_mf_lsq @ W_lsq
    tok_stats_mf_traj[t, :] = mu_t_traj
    tok_stats_mf_mf[t, :] = mu_t_mf

    # Get input for the next step from the average of outputs from the previous one
    q_t_traj = tok_stats_avg[t] @ W_q.T
    q_t_mf_lsq = muWk_t_mf_lsq @ W_lsq @ W_q.T
    Wk_Wq_x_t_traj = W_k.T @ q_t
    Wk_Wq_x_t_mf = W_k.T @ (mu_t_mf @ W_q.T)



####################################
# Statistics collection and plotting
####################################

num_feats_to_plot = min(150, token_size) # Plot only the first 150 features for better visualization
plotting_steps = [0,2, 4]
# Plot x
for t in plotting_steps:
  other_stats = [tok_stats_mf_mf[t], tok_stats_mf_x_mf_lsq[t]]
  label = ["MFx", "MFkLSQ"]
  line = ["-.", "--"]

  plot_agg_t_step(tok_stats_avg[t].T, other_stats, num_feats_to_plot, label, line, beta, t, "x")

fig, ax = plt.subplots(2, 2)
ax = ax.ravel()
feats_to_show = [3,5,103,108]
for i in range(0, len(feats_to_show)):
    ax[i].hist((tok_emb @ W_k.T)[:, feats_to_show[i]])
    ax[i].set_title(f"Feat {feats_to_show[i]}")
fig.show()
plt.close()