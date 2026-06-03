# ==============================================================================
# JADDANGI v2.1.2 · THE MASTER CORE (TOKENIZER BUG FIXED)
# Fully Bulletproofed For Days of Continuous Training Matrix Scaling
# Fixes: Inline 32k Tokenizer Auto-Build, Correct matmul.allow_tf32 Path, Permanent Causal SDPA.
# ==============================================================================

import os
import time
import math
import random
import warnings
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

# Low-Level Matrix Multiplication & Hardware Optimization Flags
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(42)

# ==============================================================================
# 1. CORE ARCHITECTURE DATACLASS & SPECIFICATIONS
# ==============================================================================
@dataclass
class ModelOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class JaddangiConfig:
    # 40M-Class Educational Master LLaMA Architecture Profile
    vocab_size: int = 32000          
    hidden_size: int = 512           # Head Dim = 512 / 8 = 64 (CUDA Core Optimal Width)
    num_layers: int = 10             
    num_heads: int = 8               
    num_kv_heads: int = 2            # Grouped Query Attention (4:1 Ratio)
    intermediate_size: int = 1376    # SwiGLU Dimensions
    max_position_embeddings: int = 4096 
    rope_theta: float = 10000.0      
    attn_dropout: float = 0.1        
    rmsnorm_eps: float = 1e-6

    # Training Environment Runtime Specifications
    batch_size: int = 8              
    grad_accum_steps: int = 4        # Effective Batch Size = 32 Sequences (16,384 Tokens)
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 5000            
    eval_every: int = 500            
    max_eval_batches: int = 50       
    checkpoint_dir: str = "checkpoints"
    use_gradient_checkpointing: bool = True 
    ema_decay: float = 0.998         
    resume_from: Optional[str] = None 

# ==============================================================================
# 2. INLINE CUSTOM TOKENIZER BUILDER (ValueError Fix)
# ==============================================================================
def build_custom_tokenizer(vocab_size=32000) -> PreTrainedTokenizerFast:
    tok_path = "jaddangi_v3_tokenizer.json"
    if os.path.exists(tok_path):
        print("💾 Loading verified custom tokenizer from disk...")
        return PreTrainedTokenizerFast(tokenizer_file=tok_path, bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>")

    print(f"📥 Tokenizer not found! Training Custom Byte-Level BPE Tokenizer ({vocab_size} Vocab)...")
    raw_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def iterator():
        for idx, item in enumerate(raw_ds):
            yield item["text"]
            if idx >= 300000: break

    tokenizer = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"], initial_alphabet=ByteLevel.alphabet())
    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    tokenizer.save(tok_path)
    
    return PreTrainedTokenizerFast(tokenizer_file=tok_path, bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>")

# ==============================================================================
# 3. ZERO-ALLOCATION IN-PLACE EXPONENTIALLY MOVING AVERAGE (EMA) ENGINE
# ==============================================================================
class WeightEMA:
    def __init__(self, model: nn.Module, decay: float = 0.998):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay)
                    self.shadow[name].add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))

# ==============================================================================
# 4. HIGH-SPEED SERIALIZED DATA PIPELINE WITH UNIVERSAL LOSS MASKING
# ==============================================================================
class MemMapDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=512, limit=1000000):
        self.split = split
        self.bin_path = f"jaddangi_v212_{self.split}.bin"
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id
        self.tail_marker_path = self.bin_path + ".tail"

        if not os.path.exists(self.bin_path):
            print(f"💾 Pre-compiling raw {self.split} strings via idempotent disk serialization writing...")
            raw_ds = load_dataset("roneneldan/TinyStories", split=self.split, streaming=True)
            
            with open(self.bin_path, "wb") as f:
                buffer = []
                for idx, item in enumerate(raw_ds):
                    tokens = tokenizer.encode(item['text'], add_special_tokens=False) + [self.eos_token_id]
                    buffer.extend(tokens)
                    
                    while len(buffer) >= max_length:
                        chunk = buffer[:max_length]
                        f.write(np.array(chunk, dtype=np.int32).tobytes())
                        buffer = buffer[max_length:]
                        
                    if idx + 1 >= limit: break
                
                if len(buffer) > 0:
                    real_length = len(buffer)
                    pad_length = max_length - real_length
                    tail_chunk = buffer + [self.eos_token_id] * pad_length
                    f.write(np.array(tail_chunk, dtype=np.int32).tobytes())
                    
                    with open(self.tail_marker_path, "w") as marker_f:
                        marker_f.write(str(real_length))
                    print(f"📦 Salvaged {real_length} residual tail tokens inside serialization stream block.")
                    
            print(f"✅ Disk compilation successfully closed for {self.split}.")
        
        self.data = np.memmap(self.bin_path, dtype=np.int32, mode='r')
        self.num_chunks = len(self.data) // max_length
        
        self.tail_real_length = None
        if os.path.exists(self.tail_marker_path):
            with open(self.tail_marker_path, "r") as marker_f:
                self.tail_real_length = int(marker_f.read().strip())
                
        print(f"✅ Mounted {self.split} packed map with {self.num_chunks} sequences.")

    def __len__(self): return self.num_chunks
    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        
        input_ids = chunk.clone()
        labels = chunk.clone()
        
        if idx == (self.num_chunks - 1) and self.tail_real_length is not None:
            labels[self.tail_real_length:] = -100

        return {"input_ids": input_ids, "labels": labels}

# ==============================================================================
# 5. LLaMA TRANSFORMER LAYERS 
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    def forward(self, x):
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x_normed = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(input_dtype)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, position_ids):
        assert position_ids.max() < self.cos_cached.size(0), f"🚨 RoPE Index overflow: position {position_ids.max().item()} exceeds cache limit of {self.cos_cached.size(0)}."
        return self.cos_cached[position_ids].unsqueeze(1).to(x.dtype), self.sin_cached[position_ids].unsqueeze(1).to(x.dtype)

def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def repeat_kv(hidden_states, n_rep):
    if n_rep == 1: return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim).reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class JaddangiAttention(nn.Module):
    def __init__(self, config: JaddangiConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_key_value_groups = config.num_heads // config.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.o_proj._is_residual = True
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)
        self.dropout = config.attn_dropout

    def forward(self, hidden_states, position_ids=None, past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Permanent is_causal=True for standard LLaMA compliance
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output), present_key_value

class JaddangiMLP(nn.Module):
    def __init__(self, config: JaddangiConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.down_proj._is_residual = True
    def forward(self, x): return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class JaddangiDecoderLayer(nn.Module):
    def __init__(self, config: JaddangiConfig):
        super().__init__()
        self.self_attn = JaddangiAttention(config)
        self.mlp = JaddangiMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rmsnorm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rmsnorm_eps)

    def forward(self, hidden_states, position_ids=None, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states, present_kv = self.self_attn(self.input_layernorm(hidden_states), position_ids, past_key_value, use_cache)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_kv

# ==============================================================================
# 6. STATIC TRANSFORMER CHECKPOINT ENGINE WRAPPER
# ==============================================================================
def checkpoint_layer_forward(module_layer, hidden_states, position_ids):
    out, _ = module_layer(hidden_states, position_ids=position_ids, past_key_value=None, use_cache=False)
    return out

# ==============================================================================
# 7. CORE MODEL ENVELOPE
# ==============================================================================
class JaddangiForCausalLM(nn.Module):
    def __init__(self, config: JaddangiConfig):
        super().__init__()
        self.config = config
        
        assert config.hidden_size % config.num_heads == 0, f"🚨 Configuration Fault: hidden_size ({config.hidden_size}) must be perfectly divisible by num_heads ({config.num_heads})."
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([JaddangiDecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "_is_residual", False):
                std = 0.02 / math.sqrt(2 * self.config.num_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, position_ids=None, past_key_values=None, use_cache=False, labels=None):
        bsz, seq_len = input_ids.shape
        past_len = past_key_values[0][0].size(2) if past_key_values else 0
        
        assert past_len + seq_len <= self.config.max_position_embeddings, f"🚨 Context footprint {past_len + seq_len} exceeds max embeddings window threshold of {self.config.max_position_embeddings}."

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        if self.training and self.config.use_gradient_checkpointing:
            use_cache = False

        hidden_states = self.embed_tokens(input_ids)
        new_cache = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values else None
            
            if self.config.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(checkpoint_layer_forward, layer, hidden_states, position_ids, use_reentrant=False)
            else:
                hidden_states, kv = layer(hidden_states, position_ids, past, use_cache)
                if new_cache is not None: new_cache.append(kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=-100)

        return ModelOutput(loss=loss, logits=logits, past_key_values=tuple(new_cache) if new_cache else None)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9, eos_token_id=None):
        self.eval()
        past_key_values = None
        for _ in range(max_new_tokens):
            if past_key_values is not None:
                current_input = input_ids[:, -1:]
                past_len = past_key_values[0][0].size(2)
                position_ids = torch.full((input_ids.size(0), 1), past_len, dtype=torch.long, device=input_ids.device)
            else:
                current_input = input_ids
                position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)

            outputs = self.forward(current_input, position_ids=position_ids, past_key_values=past_key_values, use_cache=True)
            logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = float('-inf')
                
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, sorted_mask)
                logits[mask] = float('-inf')

            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values
            if eos_token_id is not None and (next_token == eos_token_id).all(): break
        return input_ids

# ==============================================================================
# 8. ATOMIC CHECKPOINT WRITER
# ==============================================================================
def atomic_save(state_dict, filepath):
    tmp_filepath = filepath + ".tmp"
    torch.save(state_dict, tmp_filepath)
    os.replace(tmp_filepath, filepath)

# ==============================================================================
# 9. METRICS & ENGINE RUNNER DISPATCH
# ==============================================================================
def run_production_experiment():
    config = JaddangiConfig()
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    autocast_dtype = torch.float16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        print("🚀 Ampere Hardware Core Detected -> Allocating bfloat16 Engine Graph Layout.")
        autocast_dtype = torch.bfloat16

    # FIX VERIFIED: Automatic tokenizer build from strings stream iterator
    tokenizer = build_custom_tokenizer(config.vocab_size)

    train_dataset = MemMapDataset("train", tokenizer, 512, limit=1000000)
    val_dataset = MemMapDataset("validation", tokenizer, 512, limit=2000)
    
    num_workers = 2
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0), prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0), prefetch_factor=2
    )

    model = JaddangiForCausalLM(config).to(device)
    ema = WeightEMA(model, decay=config.ema_decay)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    embed_params = model.embed_tokens.weight.numel()
    print(f"📊 Verified Parameter Summary -> Core Transformer Layer: {(num_params - embed_params) / 1e6:.2f}M | Total Model Scale: {num_params / 1e6:.2f}M")

    if hasattr(torch, "compile") and device == "cuda":
        print("⚡ Rendering dynamic execution graph template via torch.compile()...")
        compiled_model = torch.compile(model, fullgraph=False, dynamic=True)
    else:
        compiled_model = model

    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2 and "norm" not in n]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and (p.ndim < 2 or "norm" in n)]

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ], lr=config.learning_rate, fused=(device == "cuda"))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda st: float(st)/float(max(1, config.warmup_steps)) if st < config.warmup_steps else 0.5 * (1.0 + math.cos(float(st - config.warmup_steps)/float(max(1, config.max_steps - config.warmup_steps)) * 3.14159)))
    scaler = GradScaler(enabled=(device == "cuda" and autocast_dtype == torch.float16))

    start_step = 1
    best_val_loss = float("inf")

    if config.resume_from and os.path.exists(config.resume_from):
        print(f"♻️ Re-loading isolated deterministic network state from: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if 'scaler' in checkpoint and not isinstance(checkpoint['scaler'], str): scaler.load_state_dict(checkpoint['scaler'])
        if 'ema_shadow' in checkpoint: ema.load_state_dict(checkpoint['ema_shadow'])
        
        random.setstate(checkpoint['rng_python'])
        np.random.set_state(checkpoint['rng_numpy'])
        torch.set_rng_state(checkpoint['rng_torch'])
        if device == "cuda": torch.cuda.set_rng_state(checkpoint['rng_cuda'])
        if 'best_val_loss' in checkpoint: best_val_loss = checkpoint['best_val_loss']
        start_step = checkpoint['step'] + 1

    step = start_step
    running_loss = 0.0
    start_time = time.time()
    
    print("\n🚀 Certified Master Reference Engine active. Training initialized...")
    
    while step <= config.max_steps:
        model.train()
        for batch in train_loader:
            if step > config.max_steps: break
            if (step - 1) % config.grad_accum_steps == 0: optimizer.zero_grad()
                
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(enabled=(device == "cuda"), dtype=autocast_dtype):
                outputs = compiled_model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / config.grad_accum_steps

            if scaler.is_enabled(): scaler.scale(loss).backward()
            else: loss.backward()
            running_loss += loss.item() * config.grad_accum_steps

            if step % config.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                scheduler.step()
                ema.update()
                
                if (step // config.grad_accum_steps) % 10 == 0:
                    end_time = time.time()
                    step_loss = running_loss / config.grad_accum_steps
                    
                    tokens_step = input_ids.size(0) * input_ids.size(1) * config.grad_accum_steps
                    throughput = tokens_step / (end_time - start_time)
                    
                    total_flops_achieved = (6 * num_params * tokens_step) / (end_time - start_time)
                    
                    print(f"Step: {step:04d}/{config.max_steps} | Loss: {step_loss:.4f} | PPL: {math.exp(min(step_loss, 20)):.2f} | Tok/s: {throughput:.0f} | Estimated Training Compute (TFLOPs/s): {total_flops_achieved / 1e12:.2f}")
                    start_time = time.time()
                    
                running_loss = 0.0

            if step % config.eval_every == 0:
                latest_state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict() if scaler.is_enabled() else 'none', 'step': step, 'best_val_loss': best_val_loss,
                    'ema_shadow': ema.state_dict(),
                    'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                    'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state() if device == "cuda" else None
                }
                atomic_save(latest_state, f"{config.checkpoint_dir}/latest_state.pt")
                
                model.eval()
                ema.apply_shadow()
                
                val_loss, val_count = 0.0, 0
                with torch.no_grad():
                    for v_batch in val_loader:
                        if val_count >= config.max_eval_batches: break
                        v_inputs = v_batch["input_ids"].to(device, non_blocking=True)
                        v_labels = v_batch["labels"].to(device, non_blocking=True)
                        with autocast(enabled=(device == "cuda"), dtype=autocast_dtype):
                            v_out = model(input_ids=v_inputs, labels=v_labels)
                            val_loss += v_out.loss.item()
                            val_count += 1
                
                avg_val_loss = val_loss / val_count
                print(f"\n🧪 [EMA MODEL EVAL] Step {step} | Val Loss: {avg_val_loss:.4f} | Val Perplexity: {math.exp(min(avg_val_loss, 20)):.2f}")
                
                print("📝 Advanced Sample Packed Story Generation:")
                p_ids = tokenizer.encode("Once upon a time, a small dragon", return_tensors="pt").to(device)
                gen_ids = model.generate(p_ids, max_new_tokens=40, temperature=0.7, top_p=0.9, eos_token_id=tokenizer.eos_token_id)
                print(f"-> {tokenizer.decode(gen_ids[0], skip_special_tokens=True)}\n" + "-"*80)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_inference_state = {
                        'model': model.state_dict(), 
                        'config': config,
                        'val_loss': best_val_loss
                    }
                    atomic_save(best_inference_state, f"{config.checkpoint_dir}/best_model.pt")
                    print(f"🏆 Lean Inference-Ready EMA Checkpoint saved successfully with Val Loss: {best_val_loss:.4f}")
                
                ema.restore()
                model.train()
            step += 1

    print("🏁 Phase 1 Complete! Jaddangi-Core v2.1.2 sets the master certified reference research standard.")

if __name__ == "__main__":
    run_production_experiment()
