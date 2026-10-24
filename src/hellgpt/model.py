import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import save_file
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.float().pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps).type_as(x)
        return x_normed * self.weight


def precompute_rope_cache(
    seq_len: int, head_dim: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    seq_idx = torch.arange(seq_len).float()
    freqs = torch.outer(seq_idx, theta)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim)
    T = x.size(2)
    cos = cos[:T].unsqueeze(0).unsqueeze(0).type_as(x)  # (1, 1, T, head_dim // 2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0).type_as(x)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
    ) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)

        q = self.q_norm(q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2))
        k = self.k_norm(k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2))
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        # 8/3 * d_model matches the parameter count of a standard 4 * d_model MLP
        hidden_dim = int(8 * d_model / 3)
        hidden_dim = (hidden_dim + 63) // 64 * 64  # Align to 64 for GPU Tensor Cores
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, dropout)

    def forward(
        self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniDiscordGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        max_seq_len: int = 160,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.12,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.label_smoothing = 0.02

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        cos, sin = precompute_rope_cache(max_seq_len, d_model // n_heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        # Depth-scaled initialization for residual projections (GPT-2 / Llama style)
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            torch.nn.init.normal_(block.mlp.w2.weight, mean=0.0, std=residual_std)

        self.tok_emb.weight = self.head.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        y_full: torch.Tensor = None,
        y_take_only: torch.Tensor = None,
        label_smoothing: float = None,
    ):
        B, T = idx.size()
        if (
                T > self.rope_cos.size(0)
                or self.rope_cos.device != idx.device
                or torch.isnan(self.rope_cos).any()
                or torch.isinf(self.rope_cos).any()
        ):
            cos, sin = precompute_rope_cache(max(T, self.max_seq_len), self.d_model // self.n_heads)
            self.rope_cos = cos.to(idx.device)
            self.rope_sin = sin.to(idx.device)

        x = self.drop(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        smoothing = self.label_smoothing if label_smoothing is None else label_smoothing

        if y_take_only is not None and y_full is not None:
            loss_take = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_take_only.view(-1),
                ignore_index=-100,
                label_smoothing=smoothing,
            )
            loss_lm = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_full.view(-1),
                ignore_index=-100,
                label_smoothing=smoothing,
            )
            loss = 0.85 * loss_take + 0.15 * loss_lm
        elif y_take_only is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_take_only.view(-1),
                ignore_index=-100,
            )
        return logits, loss


# ============================================================================
# HUGGING FACE TRANSFORMERS COMPATIBLE WRAPPER
# ============================================================================


class MiniDiscordGPTConfig(PretrainedConfig):
    model_type = "mini_discord_gpt"

    def __init__(
        self,
        vocab_size: int = 4096,
        max_seq_len: int = 160,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.12,
        bos_token_id: int = 1,
        eos_token_id: int = 3,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class HFMiniDiscordGPTForCausalLM(PreTrainedModel):
    config_class = MiniDiscordGPTConfig
    _tied_weights_keys = {"model.head.weight": "model.tok_emb.weight"}

    def __init__(self, config: MiniDiscordGPTConfig):
        super().__init__(config)
        self.model = MiniDiscordGPT(
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )
        self.post_init()
        cos, sin = precompute_rope_cache(config.max_seq_len, config.d_model // config.n_heads)
        self.model.rope_cos = cos
        self.model.rope_sin = sin

    def get_input_embeddings(self):
        return self.model.tok_emb

    def set_input_embeddings(self, value):
        self.model.tok_emb = value

    def get_output_embeddings(self):
        return self.model.head

    def set_output_embeddings(self, new_embeddings):
        self.model.head = new_embeddings

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, **kwargs):
        logits, loss = self.model(input_ids, y_take_only=labels)
        return CausalLMOutput(loss=loss, logits=logits)


def export_raw_safetensors(model: MiniDiscordGPT, output_path: str | Path) -> Path:
    """Exports flat safetensors directly compatible with your Rust Candle MiniDiscordGPT struct."""
    output_path = Path(output_path)
    if output_path.is_dir():
        output_path = output_path / "discord_mini_gpt.safetensors"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    state_dict = model.state_dict()
    clean_state_dict = {
        k: v.contiguous().cpu() for k, v in state_dict.items() if k != "head.weight"
    }
    save_file(clean_state_dict, str(output_path))
    print(f"Exported flat Safetensors to: {output_path}")
    return output_path


def export_huggingface_format(model: MiniDiscordGPT, tokenizer, output_dir: str | Path) -> Path:
    """Exports full Hugging Face Transformers folder (config.json, model.safetensors, tokenizer files)."""
    from hellgpt.tokenizer import save_hf_tokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    MiniDiscordGPTConfig.register_for_auto_class()
    HFMiniDiscordGPTForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    config = MiniDiscordGPTConfig(
        vocab_size=model.vocab_size,
        max_seq_len=model.max_seq_len,
        d_model=model.d_model,
        n_layers=model.n_layers,
        n_heads=model.n_heads,
        dropout=model.dropout,
    )
    hf_model = HFMiniDiscordGPTForCausalLM(config)
    hf_model.model.load_state_dict(model.state_dict())
    hf_model.eval()

    hf_model.save_pretrained(str(output_dir), safe_serialization=True)
    save_hf_tokenizer(tokenizer, output_dir)
    print(f"Exported Hugging Face Transformers format to directory: {output_dir}")
    return output_dir