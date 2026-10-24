import torch
from torch.nn import functional as F
from tokenizers import Tokenizer
from hellgpt.data import DiscordDataProcessor, detect_lang
from hellgpt.model import MiniDiscordGPT


@torch.no_grad()
def generate_take(
    model: MiniDiscordGPT,
    tokenizer: Tokenizer,
    user_message: str,
    force_lang: str = None,
    max_new_tokens: int = 36,
    temperature: float = 0.85,
    min_p: float = 0.06,
    repetition_penalty: float = 1.25,
    num_candidates: int = 5,
    show_candidates: bool = True,
) -> str:
    device = next(model.parameters()).device
    model.eval()

    processor = DiscordDataProcessor()
    clean_msg = processor.clean_text(user_message)
    lang_tag = force_lang if force_lang in ("<|nl|>", "<|en|>") else detect_lang(clean_msg)

    pad_id = tokenizer.token_to_id("<|pad|>")
    bos_id = tokenizer.token_to_id("<|bos|>")
    sep_id = tokenizer.token_to_id("<|sep|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    unk_id = tokenizer.token_to_id("<|unk|>")
    nl_id = tokenizer.token_to_id("<|nl|>")
    en_id = tokenizer.token_to_id("<|en|>")

    lang_id = nl_id if lang_tag == "<|nl|>" else en_id
    prompt_ids = [bos_id] + tokenizer.encode(clean_msg).ids + [sep_id, lang_id]

    idx = torch.tensor([prompt_ids] * num_candidates, dtype=torch.long, device=device)
    prompt_len = idx.size(1)

    finished = torch.zeros(num_candidates, dtype=torch.bool, device=device)
    log_probs_sum = torch.zeros(num_candidates, device=device)
    lengths = torch.zeros(num_candidates, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        for b in range(num_candidates):
            generated_so_far = set(idx[b, prompt_len:].tolist())
            for token_id in generated_so_far:
                if logits[b, token_id] < 0:
                    logits[b, token_id] *= repetition_penalty
                else:
                    logits[b, token_id] /= repetition_penalty

        for special_id in (pad_id, bos_id, sep_id, nl_id, en_id, unk_id):
            if special_id is not None:
                logits[:, special_id] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        max_probs, _ = probs.max(dim=-1, keepdim=True)
        probs[probs < (min_p * max_probs)] = 0.0
        probs = probs / probs.sum(dim=-1, keepdim=True)

        next_token = torch.multinomial(probs, num_samples=1)
        token_log_prob = torch.log(probs.gather(1, next_token) + 1e-10).squeeze(-1)

        active = ~finished
        log_probs_sum[active] += token_log_prob[active]
        lengths[active] += 1

        idx = torch.cat((idx, next_token), dim=1)
        finished = finished | (next_token.squeeze(-1) == eos_id)
        if finished.all():
            break

    candidates = []
    for b in range(num_candidates):
        gen_ids = idx[b, prompt_len:].tolist()
        if eos_id in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(eos_id)]

        text = tokenizer.decode(gen_ids).strip()
        word_count = len(text.split())

        score = (log_probs_sum[b] / (lengths[b].clamp(min=1) ** 0.8)).item()
        if word_count < 3:
            score -= 10.0
        if detect_lang(text) != lang_tag:
            score -= 3.5

        candidates.append((score, text))

    candidates.sort(key=lambda x: x[0], reverse=True)

    if show_candidates:
        print(f'Prompt ({lang_tag}): "{user_message}"')
        for rank, (score, text) in enumerate(candidates, 1):
            marker = "👑 [WINNER]" if rank == 1 else f"   [#{rank}]    "
            print(f"  {marker} (score: {score:.2f}) -> {text}")
        print("-" * 60)

    return candidates[0][1]