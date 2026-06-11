"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
import json
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from manifold_muon import manifold_muon, init_on_manifold, msign

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 1000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 1 # used to simulate larger batch sizes
batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 50000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 50000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster

# manifold constraint
manifold = 'stiefel' # 'adamw', 'none', 'stiefel', 'dgram', 'oblique', 'hetero', 'hetero-inv'
# muon optimizer
muon_lr = 0.02
muon_momentum = 0.95
muon_dual_ascent_steps = 20
muon_dual_ascent_alpha = 0.1

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# logging 
# ------------------------------------------------------------------------------
# Create experiment output directory
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_name = f"{manifold}_alpha{muon_dual_ascent_alpha}_steps{muon_dual_ascent_steps}_{run_timestamp}"
exp_dir = os.path.join(out_dir, exp_name)
os.makedirs(os.path.join(exp_dir, "spectral"), exist_ok=True)

# Save config
with open(os.path.join(exp_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
# ------------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for 2D weight matrices.
    Unconstrained: momentum -> msign -> update
    Manifold: momentum -> manifold_muon dual ascent -> update
    """
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0,
                 dual_ascent_steps=50, dual_ascent_alpha=0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        dual_ascent_steps=dual_ascent_steps,
                        dual_ascent_alpha=dual_ascent_alpha)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.float()

                state = self.state[p]
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                # Nesterov
                nesterov_g = g + momentum * buf

                mtype = getattr(p, 'manifold_type', 'none')

                if mtype == 'none':
                    # Unconstrained Muon
                    update = msign(nesterov_g)
                    if group['weight_decay'] != 0:
                        p.data.mul_(1 - lr * group['weight_decay'])
                    p.data.add_(update.to(p.dtype), alpha=-lr)
                else:
                    # Manifold Muon
                    W = p.data.float()
                    new_W = manifold_muon(
                        W, nesterov_g,
                        eta=lr,
                        alpha=group['dual_ascent_alpha'],
                        steps=group['dual_ascent_steps'],
                        manifold=mtype,
                    )
                    p.data.copy_(new_W.to(p.dtype))

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer: Muon for 2D weight matrices, AdamW for everything else
muon_params = []
adam_params_decay = []
adam_params_nodecay = []

# Determine manifold type per layer
if manifold == 'hetero':
    attn_manifold = 'stiefel'
    mlp_manifold = 'dgram'
elif manifold == 'hetero-inv':
    attn_manifold = 'dgram'
    mlp_manifold = 'stiefel'
else:
    attn_manifold = manifold
    mlp_manifold = manifold

for pn, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if manifold == 'adamw':
        # AdamW-only: all 2D params get weight decay, 1D params don't
        if p.dim() >= 2:
            adam_params_decay.append(p)
        else:
            adam_params_nodecay.append(p)
    else:
        # 2D weight matrices in attn/mlp -> Muon
        if p.dim() == 2 and ('attn.c_attn' in pn or 'attn.c_proj' in pn):
            p.manifold_type = attn_manifold if manifold != 'none' else 'none'
            muon_params.append(p)
        elif p.dim() == 2 and ('mlp.c_fc' in pn or 'mlp.c_proj' in pn):
            p.manifold_type = mlp_manifold if manifold != 'none' else 'none'
            muon_params.append(p)
        # Everything else -> AdamW
        elif p.dim() >= 2:
            adam_params_decay.append(p)
        else:
            adam_params_nodecay.append(p)

optimizer_muon = None
if muon_params:
    optimizer_muon = Muon(
        [{'params': muon_params}],
        lr=muon_lr,
        momentum=muon_momentum,
        weight_decay=weight_decay,
        dual_ascent_steps=muon_dual_ascent_steps,
        dual_ascent_alpha=muon_dual_ascent_alpha,
    )
optimizer_adam = torch.optim.AdamW([
    {'params': adam_params_decay, 'weight_decay': weight_decay},
    {'params': adam_params_nodecay, 'weight_decay': 0.0},
], lr=learning_rate, betas=(beta1, beta2))
optimizer = optimizer_adam  # for lr scheduling compatibility

# Initialize weights on manifold
if manifold not in ('none', 'adamw'):
    with torch.no_grad():
        for p in muon_params:
            p.data.copy_(init_on_manifold(p.data.float(), p.manifold_type).to(p.dtype))

if init_from == 'resume':
    # optimizer.load_state_dict(checkpoint['optimizer'])
    pass # TODO: restore both optimizers if neededs
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Open metrics log files (placed here to avoid stale file handles on re-run)
metrics_file = open(os.path.join(exp_dir, "metrics.jsonl"), "w")
eval_metrics_file = open(os.path.join(exp_dir, "eval_metrics.jsonl"), "w")
# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer_adam.param_groups:
        param_group['lr'] = lr
    if optimizer_muon is not None:
        for param_group in optimizer_muon.param_groups:
            param_group['lr'] = muon_lr * (lr / learning_rate) # scale proportionally
    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # Save eval metrics
        eval_entry = {
            "step": iter_num,
            "train_loss": losses['train'].item(),
            "val_loss": losses['val'].item(),
            "lr": lr,
        }
        eval_metrics_file.write(json.dumps(eval_entry) + "\n")
        eval_metrics_file.flush()

        # Save spectral info for all Muon params
        spectral_data = {}
        with torch.no_grad():
            for pn, p in raw_model.named_parameters():
                if pn in [n for n, _ in raw_model.named_parameters() if hasattr(_, 'manifold_type')]:
                    if not hasattr(p, 'manifold_type'):
                        continue
                    W = p.data.float()
                    if W.shape[0] < W.shape[1]:
                        W = W.T
                    svs = torch.linalg.svdvals(W)
                    gram = W.T @ W
                    gram_diag = torch.diag(gram)
                    gram_off_diag_norm = (gram - torch.diag(gram_diag)).norm().item()
                    spectral_data[pn] = {
                        "singular_values": svs.cpu(),
                        "gram_diag": gram_diag.cpu(),
                        "gram_off_diag_norm": gram_off_diag_norm,
                        "fro_norm": torch.norm(p.data).item(),
                    }
                    print(f"  {pn}: sv_max={svs[0]:.4f} sv_min={svs[-1]:.4f} gram_offdiag={gram_off_diag_norm:.4f}")
        torch.save(spectral_data, os.path.join(exp_dir, "spectral", f"step_{iter_num:06d}.pt"))

        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer_adam)
        if optimizer_muon is not None:
            scaler.unscale_(optimizer_muon)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer_adam)
    scaler.update()
    if optimizer_muon is not None:
        optimizer_muon.step()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer_adam.zero_grad(set_to_none=True)
    if optimizer_muon is not None:
        optimizer_muon.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        # Save per-step metrics
        step_entry = {
            "step": iter_num,
            "train_loss": lossf,
            "lr": lr,
            "dt_ms": dt * 1000,
            "mfu": running_mfu * 100,
        }
        metrics_file.write(json.dumps(step_entry) + "\n")
        if iter_num % 100 == 0:
            metrics_file.flush()
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

# Save final model and close log files
if master_process:
    torch.save(raw_model.state_dict(), os.path.join(exp_dir, "final_model.pt"))
    metrics_file.close()
    eval_metrics_file.close()
    print(f"Experiment saved to {exp_dir}")

if ddp:
    destroy_process_group()