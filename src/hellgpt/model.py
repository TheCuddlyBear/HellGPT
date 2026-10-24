from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors.torch import save_file
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniDiscordGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2048,
        max_seq_len: int = 96,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.22,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)
        self.tok_emb.weight = self.head.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, y_full: torch.Tensor = None, y_take_only: torch.Tensor = None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if y_take_only is not None and y_full is not None:
            loss_take = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_take_only.view(-1),
                ignore_index=-100,
                label_smoothing=0.1,
            )
            loss_lm = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_full.view(-1),
                ignore_index=-100,
                label_smoothing=0.1,
            )
            loss = 0.7 * loss_take + 0.3 * loss_lm
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
        vocab_size: int = 2048,
        max_seq_len: int = 96,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.22,
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
    clean_state_dict = {k: v.contiguous().cpu() for k, v in state_dict.items() if k != "head.weight"}
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