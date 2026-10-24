from pathlib import Path
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|sep|>", "<|eos|>", "<|unk|>", "<|nl|>", "<|en|>"]


def train_server_tokenizer(
    discord_pairs: list[tuple[str, str, str]],
    external_pairs: list[tuple[str, str, str]],
    vocab_size: int = 2048,
    save_path: str | Path = "discord_tokenizer.json",
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )

    sample_ext = external_pairs[: len(discord_pairs) * 3] if external_pairs else []
    corpus = [f"{p[0]} {p[1]}" for p in (discord_pairs * 3 + sample_ext)]

    tokenizer.train_from_iterator(corpus, trainer=trainer)
    tokenizer.save(str(save_path))
    return tokenizer


def save_hf_tokenizer(tokenizer: Tokenizer, output_dir: str | Path) -> None:
    """Saves the custom BPE tokenizer in Hugging Face PreTrainedTokenizerFast format."""
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        sep_token="<|sep|>",
        additional_special_tokens=["<|nl|>", "<|en|>"],
    )
    fast_tok.save_pretrained(str(output_dir))


class DiscordTakeDataset(Dataset):
    def __init__(self, tagged_pairs: list[tuple[str, str, str]], tokenizer: Tokenizer, max_len: int = 96):
        self.max_len = max_len
        self.pad_id = tokenizer.token_to_id("<|pad|>")
        self.bos_id = tokenizer.token_to_id("<|bos|>")
        self.sep_id = tokenizer.token_to_id("<|sep|>")
        self.eos_id = tokenizer.token_to_id("<|eos|>")
        self.nl_id = tokenizer.token_to_id("<|nl|>")
        self.en_id = tokenizer.token_to_id("<|en|>")

        self.samples = []
        for trigger, take, lang_tag in tagged_pairs:
            trigger_ids = tokenizer.encode(trigger).ids
            take_ids = tokenizer.encode(take).ids
            lang_id = self.nl_id if lang_tag == "<|nl|>" else self.en_id

            if len(trigger_ids) > max_len - 13 or len(take_ids) == 0:
                continue

            full_ids = [self.bos_id] + trigger_ids + [self.sep_id, lang_id] + take_ids + [self.eos_id]
            if len(full_ids) > max_len:
                full_ids = full_ids[: max_len - 1] + [self.eos_id]

            take_start_in_y = len(trigger_ids) + 2
            self.samples.append((full_ids, take_start_in_y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        full_ids, take_start_in_y = self.samples[idx]
        pad_len = self.max_len - len(full_ids)
        padded = full_ids + [self.pad_id] * pad_len

        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)

        y_take_only = y.clone()
        y_take_only[:take_start_in_y] = -100
        y_take_only[y_take_only == self.pad_id] = -100

        y_full = y.clone()
        y_full[y_full == self.pad_id] = -100

        return x, y_full, y_take_only