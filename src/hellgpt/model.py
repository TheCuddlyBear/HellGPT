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


class DifferentialSelfAttention(nn.Module):
    """
    (#1) Differential Self-Attention + (#3) Value Residual Connections (v0).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, layer_idx: int = 0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.sub_dim = self.head_dim // 2
        self.layer_idx = layer_idx
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for DiffAttn + RoPE"

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.sub_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

        # (#3) Value Residual mixing parameter for layers > 0
        if layer_idx > 0:
            self.v_mix = nn.Parameter(torch.zeros(1, n_heads, 1, 1))

        # (#1) Differential Attention noise-canceling lambda parameters
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.n_heads, self.sub_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.n_heads, self.sub_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.n_heads, self.sub_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.n_heads, self.sub_dim))

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        v0: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        q, k, v = self.qkv(x).split(C, dim=2)

        q = self.q_norm(q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2))
        k = self.k_norm(k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2))
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # (#3) Value Residual: anchor Layer 0 values and blend into deeper layers
        if self.layer_idx == 0 or v0 is None:
            v0 = v
        else:
            alpha = torch.sigmoid(self.v_mix).type_as(v)
            v = alpha * v + (1.0 - alpha) * v0

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        # (#1) Split Q and K into differential half-heads: (B, n_heads, T, head_dim // 2)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        drop_p = self.dropout if self.training else 0.0
        attn1 = F.scaled_dot_product_attention(q1, k1, v, dropout_p=drop_p, is_causal=True)
        attn2 = F.scaled_dot_product_attention(q2, k2, v, dropout_p=drop_p, is_causal=True)

        # Compute per-head learnable noise-canceling scalar lambda: (1, n_heads, 1, 1)
        lam1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(x)
        lam2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(x)
        lam = (lam1 - lam2 + self.lambda_init).view(1, self.n_heads, 1, 1)

        out = self.sub_norm(attn1 - lam * attn2) * (1.0 - self.lambda_init)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), v0


class BilingualSwiGLUMLP(nn.Module):
    """
    (#2) Shared Core + Soft-Routed Bilingual (NL/EN) Dual-Expert SwiGLU.
    """

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        shared_dim = (int(2 * d_model) + 63) // 64 * 64
        expert_dim = (int(2 * d_model / 3) + 63) // 64 * 64

        # Shared core trunk
        self.w1_s = nn.Linear(d_model, shared_dim, bias=False)
        self.w3_s = nn.Linear(d_model, shared_dim, bias=False)
        self.w2_s = nn.Linear(shared_dim, d_model, bias=False)

        # Soft 2-expert router (Expert 0 & Expert 1 specialize in NL vs EN phrasing/slang)
        self.router = nn.Linear(d_model, 2, bias=False)
        self.w1_exp0 = nn.Linear(d_model, expert_dim, bias=False)
        self.w3_exp0 = nn.Linear(d_model, expert_dim, bias=False)
        self.w2_exp0 = nn.Linear(expert_dim, d_model, bias=False)

        self.w1_exp1 = nn.Linear(d_model, expert_dim, bias=False)
        self.w3_exp1 = nn.Linear(d_model, expert_dim, bias=False)
        self.w2_exp1 = nn.Linear(expert_dim, d_model, bias=False)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_out = self.w2_s(F.silu(self.w1_s(x)) * self.w3_s(x))

        gates = F.softmax(self.router(x), dim=-1)
        exp0_out = self.w2_exp0(F.silu(self.w1_exp0(x)) * self.w3_exp0(x))
        exp1_out = self.w2_exp1(F.silu(self.w1_exp1(x)) * self.w3_exp1(x))

        expert_out = gates[..., 0:1] * exp0_out + gates[..., 1:2] * exp1_out
        return self.drop(shared_out + expert_out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, layer_idx: int = 0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = DifferentialSelfAttention(d_model, n_heads, dropout, layer_idx=layer_idx)
        self.ln2 = RMSNorm(d_model)
        self.mlp = BilingualSwiGLUMLP(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        v0: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, v0 = self.attn(self.ln1(x), rope_cos, rope_sin, v0=v0)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, v0


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
            [
                TransformerBlock(d_model, n_heads, dropout, layer_idx=i)
                for i in range(n_layers)
            ]
        )
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # (#5) Multi-Token Prediction (t+2 lookahead) adapter — only active during training
        self.mtp_norm = RMSNorm(d_model)
        self.mtp_proj = nn.Linear(d_model, d_model, bias=False)

        self.apply(self._init_weights)

        # Depth-scaled initialization for all residual output projections
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for block in self.blocks:
            torch.nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            torch.nn.init.normal_(block.mlp.w2_s.weight, mean=0.0, std=residual_std)
            torch.nn.init.normal_(block.mlp.w2_exp0.weight, mean=0.0, std=residual_std)
            torch.nn.init.normal_(block.mlp.w2_exp1.weight, mean=0.0, std=residual_std)

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
            cos, sin = precompute_rope_cache(
                max(T, self.max_seq_len), self.d_model // self.n_heads
            )
            self.rope_cos = cos.to(idx.device)
            self.rope_sin = sin.to(idx.device)

        x = self.drop(self.tok_emb(idx))
        v0 = None
        for block in self.blocks:
            x, v0 = block(x, self.rope_cos, self.rope_sin, v0=v0)

        h = self.ln_f(x)
        logits = self.head(h)

        loss = None
        smoothing = (
            self.label_smoothing if label_smoothing is None else label_smoothing
        )

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
            loss_t1 = 0.85 * loss_take + 0.15 * loss_lm

            # (#5) Multi-Token Prediction (t+2 lookahead loss)
            if T > 1:
                mtp_hidden = self.mtp_norm(x[:, :-1, :] + F.silu(self.mtp_proj(h[:, :-1, :])))
                mtp_logits = self.head(mtp_hidden)
                y_take_t2 = y_take_only[:, 1:].contiguous()

                if (y_take_t2 != -100).any():
                    loss_t2 = F.cross_entropy(
                        mtp_logits.view(-1, mtp_logits.size(-1)),
                        y_take_t2.view(-1),
                        ignore_index=-100,
                        label_smoothing=smoothing,
                    )
                    loss = loss_t1 + 0.25 * loss_t2
                else:
                    loss = loss_t1
            else:
                loss = loss_t1

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
        cos, sin = precompute_rope_cache(
            config.max_seq_len, config.d_model // config.n_heads
        )
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
        k: v.contiguous().cpu()
        for k, v in state_dict.items()
        if k != "head.weight" and not k.startswith("mtp_")
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