import matplotlib.pyplot as plt
import torch
from scipy.stats import norm


def learn_v_k_transform(num_tokens, repeat, max_pos, vocab_size, model, Wk_h, Wv_h, layer_i=0, head=0):

    # tok_idx = torch.randint(vocab_size, (num_tokens,))
    # tok_idx = torch.repeat_interleave(tok_idx, repeat)
    tok_idx = torch.arange(vocab_size).repeat(repeat)
    tok_pos = torch.randint(max_pos, (vocab_size * repeat,))
    tok_emb = model.transformer.wte(tok_idx)  # token embeddings of shape (b, t, n_embd)
    pos_emb = model.transformer.wpe(tok_pos)  # position embeddings of shape (t, n_embd)
    tokens = tok_emb + pos_emb
    tokens_ln = model.transformer.h[layer_i].ln_1(tokens)

    v_h =  tokens_ln @ Wv_h[head].T
    k_h =  tokens_ln @ Wk_h[head].T

    W_lsq, residuals, rank, s = torch.linalg.lstsq(k_h, v_h, rcond=None, driver="gels")

    return W_lsq


def compute_v_k_prediction(Wk_h, Wv_h, layer_i, head, model, W_slq_layer0=None):

    # Compute pseudo inverse for Wk
    Wk_h_pinv = torch.linalg.pinv(Wk_h[head])


    # Retrieve desired v and k
    b = 0 # Batch, there's only 1
    v_h = model.transformer.h[layer_i].attn.v[b, head, :, :]
    k_h = model.transformer.h[layer_i].attn.k[b, head, :, :]
    # Try to compute v with pinv
    v_hat_pinv_h = k_h @ Wk_h_pinv.T @ Wv_h[head].T

    # Compute transformation matrix using Least Squares
    W_lsq, residuals, rank, s = torch.linalg.lstsq(k_h, v_h, rcond=None, driver="gels")

    # Same but with partial info
    seq_len = model.transformer.h[layer_i].attn.k.size(2)
    W_lsq_incomp, residuals_half, rank_half, s_half = torch.linalg.lstsq(k_h[int(seq_len / 2):], v_h[int(seq_len / 2):],
                                                                         rcond=None, driver="gels")

    W_lsq_interleave, residuals_half, rank_half, s_half = torch.linalg.lstsq(k_h[::2], v_h[::2],
                                                                         rcond=None, driver="gels")

    print("W_lsq", W_lsq)
    print("W_lsq_incomp", W_lsq_incomp)
    print("Diff", W_lsq - W_lsq_incomp)

    # v predictions
    v_hat_lsq_h = k_h @ W_lsq
    v_hat_lsq_h_half = k_h @ W_lsq_incomp  # Compute all v's with the W learnt from only some of the
    v_hat_lsq_h_interleave = k_h @ W_lsq_interleave  # Compute all v's with the W learnt from every 2 tokens

    if W_slq_layer0 is not None:
        v_hat_lsq_layer_0 = k_h @ W_slq_layer0 #

    fig, ax = plt.subplots(4, 1)
    idxs = [0, int(seq_len / 3), 2 * int(seq_len / 3), -1]
    for i in range(0, 4):
        ax[i].plot(v_h[idxs[i]], label='orig')
        ax[i].plot(v_hat_lsq_h[idxs[i]], "--", label="pred lsq")
        ax[i].plot(v_hat_lsq_h_half[idxs[i]], ":", label="pred lsq_half")
        ax[i].plot(v_hat_lsq_h_interleave[idxs[i]], ":", label="pred lsq_interleave")
        if W_slq_layer0 is not None:
            ax[i].plot(v_hat_lsq_layer_0[idxs[i]], "-.", label="pred lsq_sample")
        ax[i].plot(v_hat_pinv_h[idxs[i]], "--", alpha=0.7, label="pred pinv")
    ax[-1].set_xlabel("layer features")
    plt.legend()
    plt.suptitle(f"Layer {layer_i}")
    plt.show()
    plt.close()

    # Compute rmse
    rmse_lsq = torch.sqrt(torch.mean((v_h - v_hat_lsq_h)**2, dim=1))
    rmse_lsq_half = torch.norm(v_h - v_hat_lsq_h_half, p=2, dim=1)
    rmse_lsq_interleave = torch.norm(v_h - v_hat_lsq_h_interleave, p=2, dim=1)
    rmse_lsq_layer0 = torch.norm(v_h - v_hat_lsq_layer_0, p=2, dim=1)
    rmse_pinv = torch.norm(v_h - v_hat_pinv_h, p=2, dim=1)

    fig = plt.figure()
    plt.plot(rmse_lsq_half, "--",label="LSQ_half")
    plt.plot(rmse_lsq_interleave, ":", label="LSQ_interleave")
    plt.plot(rmse_lsq_layer0, "-.", label="LSQ_sample")
    plt.plot(rmse_pinv, label="PINV")
    plt.plot(rmse_lsq, label="LSQ")
    plt.xlabel("t")
    plt.suptitle("RMSE")
    plt.legend()
    plt.show()
    plt.close()


    # Explore attention projection
    token_idx = -1  # Choose the temporal instant for which to compute the attention score y
    y_k_hat_h = model.transformer.h[layer_i].attn.y_k_hat_h[head, token_idx, :]
    Wk_h_pinv = torch.linalg.pinv(Wk_h[head])
    y_hat_h_pinv = y_k_hat_h @ Wk_h_pinv.T @ Wv_h[head].T
    y_hat_h_lsq = y_k_hat_h @ W_lsq

    # Explore proyection with the multivariate normal
    # Only computed for head 0
    if head == 0:
        y_k_hat_normal = model.transformer.h[layer_i].attn.y_k_hat_normal
        y_hat_h_normal_lsq = y_k_hat_normal @ W_lsq

    y_h = model.transformer.h[layer_i].attn.y_h[head, token_idx, :]

    W_v_hat = W_lsq @ Wk_h[head]

    print("diff W", W_v_hat - Wv_h[head])
    print("diff att", model.transformer.h[layer_i].attn.y_h - y_hat_h_lsq)

    plt.figure()
    plt.plot(y_h, label='original')
    plt.plot(y_hat_h_lsq, ":", label="pred lsq")
    plt.plot(y_hat_h_pinv, label="pred pinv")
    if head == 0:
        plt.plot(y_hat_h_normal_lsq, label="pred gaussian")
    plt.legend()
    plt.show()
    plt.close()

    print(y_h)
    print(y_hat_h_lsq)
    print(y_hat_h_pinv)

def compute_means(ax, model, W, x_tensor, weight_name):

    # Examine feat feat_a from head h_i for different samples
    h_i = torch.randint(model.config.n_head, (1,))[0]
    feat_a = torch.randint(model.config.n_embd // model.config.n_head, (1,))[0]
    print(h_i, feat_a)

    m_alpha_h_i_a = torch.matmul(W[h_i, feat_a], x_tensor.T)

    # remove most probable occurrence
    uniques, counts = torch.unique(m_alpha_h_i_a, return_counts=True)
    mode_idx = torch.argmax(counts)
    mode_elem = uniques[mode_idx]
    m_alpha_h_i_a_no_mode = m_alpha_h_i_a[m_alpha_h_i_a != mode_elem]

    # Get mean and std of the dist without mode
    mean_no_mode = m_alpha_h_i_a_no_mode.mean()
    std_no_mode = m_alpha_h_i_a_no_mode.std()

    # Create gaussian
    x_values_gauss = torch.linspace(m_alpha_h_i_a_no_mode.min(), m_alpha_h_i_a_no_mode.max(), steps=1000)
    gaussian_no_mode = norm.pdf(x_values_gauss, mean_no_mode, std_no_mode)

    # Pre-compute hist
    density = True
    num_bins = 50
    hist, bin_edges = torch.histogram(m_alpha_h_i_a, bins=num_bins, density=density)
    max_bin_count_idx = torch.argmax(hist)
    max_hist = hist[max_bin_count_idx].item()
    hist[:] = 0  # Only plot the peak in dist
    hist[max_bin_count_idx] = max_hist

    # Pre-compute hist no mode
    hist_no_mode, bin_edges_no_mode = torch.histogram(m_alpha_h_i_a_no_mode, bins=num_bins, density=True)

    print("Plot", weight_name, h_i.item(), feat_a.item(), "Unique", len(torch.unique(m_alpha_h_i_a)))

    ax.stairs(hist, bin_edges, fill=True, label="MostProb")
    ax.stairs(hist_no_mode, bin_edges_no_mode, fill=True, color="red", alpha=0.5, label="Others Hist")
    ax.plot(x_values_gauss, gaussian_no_mode, color="darkred")
    ax.axvline(m_alpha_h_i_a.mean(), color='k', linestyle='dashed', alpha=0.7, label="Total Mean")
    ax.axvline(m_alpha_h_i_a_no_mode.mean(), color='darkred', linestyle='dashed', alpha=0.7, label="Others Mean")
    ax.set_title(rf"$m^{weight_name}_{{h_{{{h_i}}},a_{{{feat_a}}}}}$")

def hist_2D(Wa, Wb, W_namea, W_nameb, model, x_tensor, ax):

    # Examine feat feat_a from head h_i for different samples
    h_i = torch.randint(model.config.n_head, (1,))[0]
    feat_a = torch.randint(model.config.n_embd // model.config.n_head, (1,))[0]
    feat_b = torch.randint(model.config.n_embd // model.config.n_head, (1,))[0]
    m_h_i_a = torch.matmul(Wa[h_i, feat_a], x_tensor.T).cpu()
    m_h_i_b = torch.matmul(Wb[h_i, feat_b], x_tensor.T).cpu()

    # Plot 2D hist
    ax.hist2d(m_h_i_a, m_h_i_b, bins=50)
    ax.set_title(
        rf"$m^{W_namea}_{{h_{{{h_i}}},a_{{{feat_a}}}}}$ vs $m^{W_nameb}_{{h_{{{h_i}}},b_{{{feat_b}}}}}$")

