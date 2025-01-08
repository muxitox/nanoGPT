# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-gpu'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 50 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare'
gradient_accumulation_steps = 5
batch_size = 12
block_size = 256 # context of up to 64 previous characters

# baby GPT model :)
n_layer = 2
n_head = 6
n_embd = 384
dropout = 0.0 # No dropout

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 10000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on Colab
device = 'cuda'  # run on cpu only
compile = True # do not torch compile the model