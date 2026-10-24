import json
import re
import random
from pathlib import Path
from datetime import datetime
import os
from huggingface_hub import hf_hub_download

# Replace with your actual Hugging Face username/dataset name
HF_DATASET_REPO = "TheCuddlyBear/Hell-Dataset"

DISCORD_JSON_FILES = [
    "bingi.json",
    "bingi-snol.json",
    "bingi-vc.json",
    "bingi-gaming.json",
    "bingi-nasi.json",
]

NL_WORDS = {
    "de", "het", "een", "ik", "je", "jij", "wij", "we", "ze", "hij", "zij", "niet", "wel",
    "maar", "want", "omdat", "ook", "nog", "al", "naar", "voor", "met", "van", "op", "aan",
    "uit", "over", "bij", "als", "dan", "toch", "echt", "gewoon", "heel", "veel", "meer",
    "geen", "niks", "iets", "iemand", "niemand", "alles", "hoe", "wat", "waar", "waarom",
    "wanneer", "wie", "welke", "deze", "dit", "die", "dat", "mijn", "jouw", "zijn", "haar",
    "onze", "jullie", "hun", "ben", "bent", "heb", "hebt", "heeft", "hebben", "had", "was",
    "waren", "wordt", "worden", "kan", "kunt", "kunnen", "moet", "moeten", "mag", "wil",
    "willen", "ga", "gaat", "gaan", "kom", "komt", "komen", "doe", "doet", "doen", "zie",
    "ziet", "zien", "zeg", "zegt", "denken", "weet", "weten", "vind", "vinden", "lekker",
    "leuk", "mooi", "goed", "slecht", "dik", "lelijk", "gekke", "facking", "ff", "wrm",
    "gwn", "sws", "eigenlijk", "misschien", "straks", "vanavond", "morgen", "gisteren"
}

EN_WORDS = {
    "the", "and", "you", "that", "was", "for", "are", "with", "his", "they", "this",
    "have", "from", "one", "had", "word", "but", "not", "what", "all", "were", "when",
    "your", "can", "said", "there", "use", "each", "which", "she", "how", "their", "will",
    "other", "about", "out", "many", "then", "them", "these", "some", "her", "would",
    "make", "like", "him", "into", "time", "has", "look", "two", "more", "write", "see",
    "number", "way", "could", "people", "than", "first", "water", "been", "call", "who",
    "its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may",
    "part", "actually", "really", "think", "know", "good", "bad", "yeah", "nah", "bro",
    "dude", "literally", "always", "never", "someone", "anyone", "tonight", "game"
}


def detect_lang(text: str) -> str:
    """Returns '<|nl|>' or '<|en|>'. Defaults to '<|nl|>' on ties."""
    words = re.findall(r"[a-zA-Z']+", text.lower())
    nl_score = sum(1 for w in words if w in NL_WORDS)
    en_score = sum(1 for w in words if w in EN_WORDS)
    return "<|en|>" if en_score > nl_score else "<|nl|>"

def download_default_exports(
    data_dir: Path, 
    repo_id: str = HF_DATASET_REPO, 
    token: str | None = None
) -> list[Path]:
    """
    Downloads raw Discord JSON exports from a private Hugging Face dataset repo.
    Automatically uses the HF_TOKEN environment variable or cached CLI login if token is None.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    
    # Fallback to HF_TOKEN environment variable if not explicitly passed
    auth_token = token or os.environ.get("HF_TOKEN")

    for filename in DISCORD_JSON_FILES:
        dest = data_dir / filename
        if not dest.exists():
            print(f"Downloading {filename} from HF dataset ({repo_id})...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",  # Crucial: tells HF this is a dataset, not a model
                local_dir=data_dir,
                token=auth_token,
            )
            paths.append(Path(downloaded_path))
        else:
            paths.append(dest)

    return paths


class DiscordDataProcessor:
    def __init__(self, merge_window_sec: int = 60, reply_window_sec: int = 180, min_len: int = 3):
        self.merge_window_sec = merge_window_sec
        self.reply_window_sec = reply_window_sec
        self.min_len = min_len

    def clean_text(self, text: str) -> str:
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"<a?(:[a-zA-Z0-9_]+:)\d+>", r"\1", text)
        text = re.sub(r"<[@#!&]\d+>", "@user", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def is_valid_message(self, msg: dict, text: str) -> bool:
        if msg.get("author", {}).get("isBot", False):
            return False
        if msg.get("type") not in ("Default", "Reply"):
            return False
        if text.startswith(("!", "/", "$", ".", "-", "?", "p!", "m!", ">>")):
            return False
        if len(text) < self.min_len:
            return False
        return True

    def parse_timestamp(self, ts_str: str) -> float:
        return datetime.fromisoformat(ts_str).timestamp()

    def process_files(self, json_paths: list[str | Path]) -> list[tuple[str, str, str]]:
        pairs = []
        for path in json_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            raw_messages = data.get("messages", [])
            cleaned_msgs = []
            msg_by_id = {}

            for m in raw_messages:
                clean_content = self.clean_text(m.get("content", ""))
                if not self.is_valid_message(m, clean_content):
                    continue

                entry = {
                    "id": m["id"],
                    "author_id": m["author"]["id"],
                    "content": clean_content,
                    "timestamp": self.parse_timestamp(m["timestamp"]),
                    "ref_id": m.get("reference", {}).get("messageId") if m.get("reference") else None,
                }
                cleaned_msgs.append(entry)
                msg_by_id[entry["id"]] = entry

            # 1. Explicit Discord Replies
            for m in cleaned_msgs:
                ref_id = m["ref_id"]
                if ref_id and ref_id in msg_by_id:
                    parent = msg_by_id[ref_id]
                    if parent["author_id"] != m["author_id"]:
                        pairs.append((parent["content"], m["content"]))

            # 2. Consecutive message merging & adjacent pairing
            merged = []
            for m in cleaned_msgs:
                if (
                    merged
                    and merged[-1]["author_id"] == m["author_id"]
                    and (m["timestamp"] - merged[-1]["timestamp"]) <= self.merge_window_sec
                    and m["ref_id"] is None
                ):
                    merged[-1]["content"] += " " + m["content"]
                    merged[-1]["timestamp"] = m["timestamp"]
                else:
                    merged.append(m.copy())

            for i in range(len(merged) - 1):
                curr_m, next_m = merged[i], merged[i + 1]
                if (
                    curr_m["author_id"] != next_m["author_id"]
                    and (next_m["timestamp"] - curr_m["timestamp"]) <= self.reply_window_sec
                ):
                    pairs.append((curr_m["content"], next_m["content"]))

        unique_pairs = list(dict.fromkeys(pairs))
        tagged_pairs = []
        for trig, take in unique_pairs:
            if len(trig.split()) >= 2 and len(take.split()) >= 3 and len(take) >= 15:
                trig_lang = detect_lang(trig)
                take_lang = detect_lang(take)
                lang_tag = take_lang if trig_lang == take_lang else trig_lang
                tagged_pairs.append((trig, take, lang_tag))

        nl_cnt = sum(1 for _, _, l in tagged_pairs if l == "<|nl|>")
        en_cnt = sum(1 for _, _, l in tagged_pairs if l == "<|en|>")
        print(f"Filtered Discord pairs: {len(tagged_pairs)} (NL: {nl_cnt}, EN: {en_cnt})")
        return tagged_pairs


# In data.py: Remove `import time` and `import requests`, and replace
# `scrape_subreddit_pairs` + `load_external_bilingual_conversations` with:

def _extract_short_turn(text: str, processor: DiscordDataProcessor, max_words: int = 32) -> str:
    """Extracts the first 1-2 sentences and caps word length rather than discarding the turn."""
    if not text:
        return ""
    cleaned = processor.clean_text(text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if s.strip()]
    if not sentences:
        return ""
    out = sentences[0]
    if len(out.split()) < 8 and len(sentences) > 1:
        out = f"{out} {sentences[1]}"
    words = out.split()
    if len(words) > max_words:
        out = " ".join(words[:max_words])
    return out


def load_external_bilingual_conversations(
    processor: DiscordDataProcessor, target_per_lang: int = 40000
) -> list[tuple[str, str, str]]:
    from datasets import load_dataset

    nl_pairs = []
    en_pairs = []

    def is_good_pair(trig: str, take: str) -> bool:
        return (
            2 <= len(trig.split()) <= 32
            and 3 <= len(take.split()) <= 36
            and len(take) >= 12
        )

    print("Loading bilingual dialogue & knowledge pairs from Hugging Face (No OAuth)...")

    # 1. OpenAssistant 2 (Cap EN at 50% of target so SmolTalk also gets loaded)
    try:
        oasst = load_dataset("OpenAssistant/oasst2", split="train")
        msg_map, children_map = {}, {}
        for row in oasst:
            if row["lang"] in ("nl", "en") and not row.get("deleted", False):
                mid, pid = row["message_id"], row["parent_id"]
                msg_map[mid] = (row["text"], "<|nl|>" if row["lang"] == "nl" else "<|en|>")
                if pid:
                    children_map.setdefault(pid, []).append(mid)

        en_cap = target_per_lang // 2
        for pid, c_ids in children_map.items():
            if pid in msg_map:
                p_text, p_lang = msg_map[pid]
                p_short = _extract_short_turn(p_text, processor)
                for cid in c_ids:
                    c_text, c_lang = msg_map[cid]
                    c_short = _extract_short_turn(c_text, processor)
                    if p_lang == c_lang and is_good_pair(p_short, c_short):
                        if p_lang == "<|nl|>":
                            nl_pairs.append((p_short, c_short, "<|nl|>"))
                        elif len(en_pairs) < en_cap:
                            en_pairs.append((p_short, c_short, "<|en|>"))
        print(f"  + OpenAssistant2: {len(nl_pairs)} NL, {len(en_pairs)} EN")
    except Exception as e:
        print(f"  ! OASST2 skip: {e}")

    # 2. Dutch: UltraFeedback Dutch (Uses split='train_prefs' and 'chosen' message list)
    try:
        nl_uf = load_dataset("BramVanroy/ultra_feedback_dutch", split="train_prefs", streaming=True)
        before = len(nl_pairs)
        for row in nl_uf:
            if len(nl_pairs) >= int(target_per_lang * 0.55):
                break
            msgs = row.get("chosen") or row.get("messages") or []
            for i in range(len(msgs) - 1):
                if isinstance(msgs[i], dict) and isinstance(msgs[i + 1], dict):
                    trig = _extract_short_turn(msgs[i].get("content", ""), processor)
                    take = _extract_short_turn(msgs[i + 1].get("content", ""), processor)
                    if is_good_pair(trig, take):
                        nl_pairs.append((trig, take, "<|nl|>"))
        print(f"  + UltraFeedback-NL: +{len(nl_pairs) - before} NL")
    except Exception as e:
        print(f"  ! UltraFeedback-NL skip: {e}")

    # 3. Dutch: Dolly-15k & Alpaca Dutch (Both use split='train_sft' and 'messages' list)
    for ds_name in ["BramVanroy/dolly-15k-dutch", "BramVanroy/alpaca-cleaned-dutch"]:
        if len(nl_pairs) >= target_per_lang:
            break
        try:
            ds = load_dataset(ds_name, split="train_sft", streaming=True)
            before = len(nl_pairs)
            for row in ds:
                if len(nl_pairs) >= target_per_lang:
                    break
                msgs = row.get("messages", [])
                if len(msgs) >= 2:
                    for i in range(len(msgs) - 1):
                        trig = _extract_short_turn(msgs[i].get("content", ""), processor)
                        take = _extract_short_turn(msgs[i + 1].get("content", ""), processor)
                        if is_good_pair(trig, take):
                            nl_pairs.append((trig, take, "<|nl|>"))
                elif "instruction" in row or "prompt" in row:
                    prompt = row.get("instruction") or row.get("prompt") or ""
                    if row.get("input"):
                        prompt = f"{prompt} {row['input']}"
                    trig = _extract_short_turn(prompt, processor)
                    take = _extract_short_turn(row.get("output") or row.get("response") or "", processor)
                    if is_good_pair(trig, take):
                        nl_pairs.append((trig, take, "<|nl|>"))
            print(f"  + {ds_name}: +{len(nl_pairs) - before} NL")
        except Exception as e:
            print(f"  ! {ds_name} skip: {e}")

    # 4. English: SmolTalk Everyday Conversations + Smol-Magpie-Ultra
    for subset in ["everyday-conversations", "smol-magpie-ultra"]:
        if len(en_pairs) >= target_per_lang:
            break
        try:
            smol = load_dataset("HuggingFaceTB/smoltalk", subset, split="train", streaming=True)
            before = len(en_pairs)
            for row in smol:
                if len(en_pairs) >= target_per_lang:
                    break
                msgs = row.get("messages", [])
                for i in range(len(msgs) - 1):
                    trig = _extract_short_turn(msgs[i].get("content", ""), processor)
                    take = _extract_short_turn(msgs[i + 1].get("content", ""), processor)
                    if is_good_pair(trig, take):
                        en_pairs.append((trig, take, "<|en|>"))
            print(f"  + SmolTalk ({subset}): +{len(en_pairs) - before} EN")
        except Exception as e:
            print(f"  ! SmolTalk ({subset}) skip: {e}")

    external = nl_pairs[:target_per_lang] + en_pairs[:target_per_lang]
    random.shuffle(external)
    print(
        f"Total external pairs loaded: {len(external)} "
        f"(NL: {len(nl_pairs[:target_per_lang])}, EN: {len(en_pairs[:target_per_lang])})"
    )
    return external