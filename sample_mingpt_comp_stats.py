"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model_tests import GPTConfig, GPT
import matplotlib.pyplot as plt
from explore_statistics import compute_v_k_prediction, compute_means, hist_2D


# -----------------------------------------------------------------------------
init_from = 'gpt2' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-shakespeare-gpu' # ignored if init_from is not 'resume'
# start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# start = "\nMy lord, cheerfully made" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# start = "\nWARWICK:\nWhat, wilt thou, wilt thou not, for thy head?\nQUEEN MARGARET:\nHow now, madam?"
# start = "FILE:text_sample/sample_long.txt"
# start = "FILE:text_sample/sample_long_gpt2_2.txt"
# start = "\nThe Lion King was conceived during conversations among various Disney executives, to whom several writers submitted early treatments. Original director George Scribner had envisioned"
start = "\nNeuroscience is the scientific study of the nervous system (the brain, spinal cord, and peripheral nervous system), its functions, and its disorders. It is a multidisciplinary science that combines"
num_samples = 100 # number of samples to draw
max_new_tokens = 2 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 2000 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf, compute_statistics=True)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

x_tensor_ini = torch.zeros((num_samples, model.config.n_embd))
x_tensor_end = torch.zeros((num_samples, model.config.n_embd))
q_tensor = torch.zeros((num_samples, max_new_tokens, model.config.n_head, model.config.n_embd // model.config.n_head))

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            # y, x_array, q_matrix, x_history,  probs_0, y_k_hat_h, y_h = model.generate_comp_stats(x, max_new_tokens, temperature=temperature, top_k=top_k)

            y, probs_0 = model.generate_comp_stats(x, max_new_tokens, temperature=temperature, top_k=top_k)


            # Retrieve statistics from the first and last layers
            layer_i = 0
            x_history_ini = model.transformer.h[layer_i].attn.x  # Shapes B, T, C
            x_tensor_ini[k] = x_history_ini[:, -1, :]  # To compute mfs, save just last sampled token
            layer_i = -1
            x_history_end = model.transformer.h[layer_i].attn.x
            x_tensor_end[k] = x_history_end[:, -1, :]

            if k % 100 == 0:
                print("Computing sample ", k)

            plot_samples = False
            if plot_samples:
                print("Sample len", len(y[0].tolist()))
                print('---------------')
                print(decode(y[0].tolist()))
                print('---------------')


    # Once all samples have been computed, compute statistics

    for layer_i in [0, -1]:

        if layer_i == 0:
            x_tensor = x_tensor_ini

        elif layer_i == -1:
            x_tensor = x_tensor_end

        else:
            raise Exception("Not implemented")

        # Get separate weights
        Wq, Wk, Wv = model.transformer.h[layer_i].attn.c_attn.weight.split(model.config.n_embd)

        # Divide weight in n_head heads
        Wq_h = Wq.view(model.config.n_head, model.config.n_embd // model.config.n_head, model.config.n_embd)
        Wk_h = Wk.view(model.config.n_head, model.config.n_embd // model.config.n_head, model.config.n_embd)
        Wv_h = Wv.view(model.config.n_head, model.config.n_embd // model.config.n_head, model.config.n_embd)


        ###########################
        # Try to retrieve v from k:
        ###########################

        head = 0
        compute_v_k_prediction(Wk_h, Wv_h, layer_i, head, model)

        ####
        # Explore mean fields
        ###

        # To get the projection for all the heads with some x:
        sample_id = 1
        mq_h = torch.matmul( Wq_h, x_tensor[sample_id])


        fig, ax = plt.subplots(3, 3)
        ax_ravel = ax.ravel()
        print("Plot q")
        for i in range(0, 3):
            # Examine feat feat_a from head h_i for different samples
            compute_means(ax_ravel[i], model, Wq_h, x_tensor, "q")

        print("Plot k")
        for i in range(3, 6):
            # Examine feat feat_a from head h_i for different samples
            compute_means(ax_ravel[i], model, Wk_h, x_tensor, "k")

        print("Plot v")
        for i in range(6 , 9):
            compute_means(ax_ravel[i], model, Wq_h, x_tensor, "v")

        ax_ravel[0].legend(fontsize='xx-small')

        fig.show()
        plt.close(fig)


        #################
        # 2D Hist
        ###################

        #  Compute 2D hists
        W_2D_list = [[Wk_h, Wk_h], [Wq_h, Wq_h], [Wv_h, Wv_h], [Wk_h, Wq_h], [Wq_h, Wv_h], [Wv_h, Wk_h]]
        W_2D_names_list = [["k", "k"], ["q", "q"], ["v", "v"], ["k", "q"], ["q", "v"], ["v", "k"]]

        num_cols = 3
        num_rows = 6
        fig, ax = plt.subplots(num_rows, num_cols)
        ax_ravel = ax.ravel()
        for i in range(0, num_rows * num_cols):
            # Get W
            W_pair = W_2D_list[i // num_cols]
            Wa = W_pair[0]
            Wb = W_pair[1]

            W_names = W_2D_names_list[i // num_cols]
            W_namea = W_names[0]
            W_nameb = W_names[1]

            hist_2D(Wa, Wb, W_namea, W_nameb, model, x_tensor, ax)

        ax_ravel[0].legend(fontsize='xx-small')

        # fig.tight_layout()
        fig.show()


        ########################
        # Explore QK interaction
        ########################

        # # Compute first mean and variances of x
        # # Do this in torch.no_grad or memory requirements will scale
        # x_mean = torch.mean(x_tensor, dim=0)
        # x_sq_mean = torch.mean(x_tensor**2, dim=0)
        # x_var = x_mean**2 - x_sq_mean
        #
        # feat_mean = torch.matmul(x_mean, model.lm_head.weight.T) / gptconf.n_embd
        #
        # num_cov_samples = 5000
        # feat_cov_t_ab_tensor = torch.zeros((num_cov_samples, max_new_tokens))
        # for r in range(num_cov_samples):
        #     # Compute covariances for features a and b
        #     a, b = torch.randint(gptconf.vocab_size, (2,))
        #
        #     feat_cov_t_ab_tensor[r] = torch.einsum('i,i,ti->t', model.lm_head.weight[a], model.lm_head.weight[b], x_var) / gptconf.n_embd ** 2
        #
        #
        #
        # d_list = [0]
        # for d in d_list:
        #     fig, ax = plt.subplots(1, 3)
        #     print("Check step", d)
        #     ax[0].hist(original_m[d], bins=100)
        #     ax[0].set_title(rf"original $m^{{out}}_{{a,t={d}}}$")
        #     ax[1].hist(feat_mean[d], bins=100)
        #     ax[1].set_title(rf"$m^{{out}}_{{a,t={d}}}$")
        #     ax[2].hist(feat_cov_t_ab_tensor[:, d], bins=100)
        #     ax[2].set_title(rf"sample $\Sigma^{{out}}_{{a,b,t={d}}}$")
        #     fig.suptitle("t = " + str(d))
        #     print(feat_mean)
        #     print()
        #
        #     fig.show()
        #     plt.close(fig)
        # import pdb; pdb.set_trace()
        #
