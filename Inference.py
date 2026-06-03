# ==============================================================================
# JADDANGI 44M · HUGGING FACE INFERENCE TEST (ARCHITECTURE FIXED)
# ==============================================================================

!pip install huggingface_hub transformers -q

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download
import __main__ 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Running on: {DEVICE.upper()}\n")

# ==============================================================================
# 1. 44M Architecture Config (Fixed num_kv_heads to 2)
# ==============================================================================
@dataclass
class ModelOutput:
    logits: torch.FloatTensor = None

@dataclass
class Jaddangi44MConfig:
    vocab_size: int = 32000          
    hidden_size: int = 512           
    num_layers: int = 10             
    num_heads: int = 8               
    num_kv_heads: int = 2            # 👈 4 నుండి 2 కి మార్చబడింది
    intermediate_size: int = 1376    
    max_position_embeddings: int = 512 
    rope_theta: float = 10000.0      
    rmsnorm_eps: float = 1e-6

__main__.JaddangiConfig = Jaddangi44MConfig 

# ==============================================================================
# 2. MODEL CLASSES (The Core Architecture)
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (self.weight * (x.to(torch.float32) * torch.rsqrt(variance + self.eps))).to(x.dtype)

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
    def forward(self, x, seq_len):
        return self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(1).to(x.dtype), self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(1).to(x.dtype)

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def repeat_kv(hidden_states, n_rep):
    if n_rep == 1: return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim).reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class JaddangiAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads, self.num_kv_heads = config.num_heads, config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        # 👈 Bias=True గా మార్చబడింది (To match your saved checkpoint)
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(self, hidden_states):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(value_states, q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states, value_states = repeat_kv(key_states, self.num_key_value_groups), repeat_kv(value_states, self.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)

class JaddangiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    def forward(self, x): return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class JaddangiDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = JaddangiAttention(config)
        self.mlp = JaddangiMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rmsnorm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rmsnorm_eps)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(self.input_layernorm(hidden_states))
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

class JaddangiForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([JaddangiDecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers: hidden_states = layer(hidden_states)
        return ModelOutput(logits=self.lm_head(self.norm(hidden_states)))

# ==============================================================================
# 3. Download from Hugging Face & Load Safely
# ==============================================================================
repo_id = "VenkataRamanaKurumallajaddangi/Jaddangi-44M-TinyStories"

print("📥 Downloading Tokenizer...")
tokenizer_path = hf_hub_download(repo_id=repo_id, filename="jaddangi_v3_tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

print("📥 Downloading Model Weights...")
model_path = hf_hub_download(repo_id=repo_id, filename="best_model.pt")

print("⚙️ Initializing Model & Loading Weights...")
config = Jaddangi44MConfig()
model = JaddangiForCausalLM(config).to(DEVICE)

checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

if isinstance(checkpoint, dict):
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

model.eval()
print("✅ Model Successfully Loaded!\n")

# ==============================================================================
# 4. Simple Generation Function
# ==============================================================================
@torch.no_grad()
def generate_story(prompt, max_new_tokens=100, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    for _ in range(max_new_tokens):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(input_ids[0], skip_special_tokens=True).replace('Ġ', ' ').replace('Ċ', '\n').strip()

# ==============================================================================
# 5. Test the Model!
# ==============================================================================
test_prompt = "One day, a clever little fox found a magical"
print("🎯 Generating Story...")
print("-" * 60)
print(test_prompt + " " + generate_story(test_prompt))
print("-" * 60)
