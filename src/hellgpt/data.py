import json
import re
import time
import random
from pathlib import Path
from datetime import datetime
import requests
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


def scrape_subreddit_pairs(
    subreddit: str, lang_tag: str, processor: DiscordDataProcessor, max_posts: int = 20
) -> list[tuple[str, str, str]]:
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) BingiGPTTrainer/1.0"}
    extracted = []

    for sort in ["hot", "top"]:
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json?limit={max_posts}&t=month"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                continue
            posts = resp.json().get("data", {}).get("children", [])
        except Exception:
            continue

        for post in posts[:20]:
            permalink = post.get("data", {}).get("permalink")
            post_title = processor.clean_text(post.get("data", {}).get("title", ""))
            if not permalink:
                continue
            try:
                c_resp = requests.get(
                    f"https://www.reddit.com{permalink}.json?limit=50", headers=headers, timeout=10
                )
                if c_resp.status_code != 200:
                    continue
                c_data = c_resp.json()
                if len(c_data) < 2:
                    continue
                comments = c_data[1].get("data", {}).get("children", [])

                for c in comments:
                    c_body = c.get("data", {}).get("body", "")
                    if not c_body or c_body in ("[deleted]", "[removed]"):
                        continue
                    clean_parent = processor.clean_text(c_body)

                    if 2 <= len(post_title.split()) <= 22 and 3 <= len(clean_parent.split()) <= 28:
                        extracted.append((post_title, clean_parent, lang_tag))

                    replies = c.get("data", {}).get("replies")
                    if isinstance(replies, dict):
                        for r in replies.get("data", {}).get("children", []):
                            r_body = r.get("data", {}).get("body", "")
                            if not r_body or r_body in ("[deleted]", "[removed]"):
                                continue
                            clean_reply = processor.clean_text(r_body)
                            if 2 <= len(clean_parent.split()) <= 25 and 3 <= len(clean_reply.split()) <= 28:
                                extracted.append((clean_parent, clean_reply, lang_tag))
                time.sleep(0.3)
            except Exception:
                continue
    return extracted


def load_external_bilingual_conversations(
    processor: DiscordDataProcessor, target_per_lang: int = 18000
) -> list[tuple[str, str, str]]:
    external = []
    nl_subs = ["ik_ihe", "thenetherlands", "Nederland", "cirkeltrek"]
    en_subs = ["CasualConversation", "gaming", "memes", "AskReddit"]

    print("Scraping Dutch & English Reddit threads...")
    for sub in nl_subs:
        pairs_sub = scrape_subreddit_pairs(sub, "<|nl|>", processor, max_posts=15)
        external.extend(pairs_sub)
        print(f"  r/{sub}: +{len(pairs_sub)} pairs")

    for sub in en_subs:
        pairs_sub = scrape_subreddit_pairs(sub, "<|en|>", processor, max_posts=15)
        external.extend(pairs_sub)
        print(f"  r/{sub}: +{len(pairs_sub)} pairs")

    from datasets import load_dataset

    print("Loading supplementary Dutch & English dialogue pairs from Hugging Face...")
    try:
        oasst = load_dataset("OpenAssistant/oasst1", split="train")
        msg_map, children_map = {}, {}
        for row in oasst:
            if row["lang"] in ("nl", "en") and not row.get("deleted", False):
                mid, pid = row["message_id"], row["parent_id"]
                msg_map[mid] = (row["text"], "<|nl|>" if row["lang"] == "nl" else "<|en|>")
                if pid:
                    children_map.setdefault(pid, []).append(mid)

        for pid, c_ids in children_map.items():
            if pid in msg_map:
                p_text, p_lang = msg_map[pid]
                p_short = processor.clean_text(re.split(r"[.!?\n]+", p_text)[0])
                for cid in c_ids:
                    c_text, c_lang = msg_map[cid]
                    c_short = processor.clean_text(re.split(r"[.!?\n]+", c_text)[0])
                    if p_lang == c_lang and 2 <= len(p_short.split()) <= 25 and 3 <= len(c_short.split()) <= 28:
                        external.append((p_short, c_short, p_lang))
    except Exception as e:
        print(f"OASST skip note: {e}")

    try:
        nl_ds = load_dataset("BramVanroy/ultra_feedback_dutch", split="train_sft", streaming=True)
        nl_count = 0
        for row in nl_ds:
            msgs = row.get("messages", [])
            if len(msgs) >= 2:
                trig = processor.clean_text(re.split(r"[.!?\n]+", msgs[0].get("content", ""))[0])
                take = processor.clean_text(re.split(r"[.!?\n]+", msgs[1].get("content", ""))[0])
                if 2 <= len(trig.split()) <= 24 and 3 <= len(take.split()) <= 26 and len(take) >= 15:
                    external.append((trig, take, "<|nl|>"))
                    nl_count += 1
                    if nl_count >= target_per_lang:
                        break
    except Exception as e:
        print(f"Dutch HF note: {e}")

    try:
        en_ds = load_dataset("roskoN/dailydialog", split="train")
        en_count = 0
        for row in en_ds:
            utterances = row.get("utterances", [])
            for i in range(len(utterances) - 1):
                trig = processor.clean_text(utterances[i])
                take = processor.clean_text(utterances[i + 1])
                if 2 <= len(trig.split()) <= 24 and 3 <= len(take.split()) <= 26 and len(take) >= 15:
                    external.append((trig, take, "<|en|>"))
                    en_count += 1
                    if en_count >= target_per_lang:
                        break
            if en_count >= target_per_lang:
                break
    except Exception as e:
        print(f"English HF note: {e}")

    random.shuffle(external)
    nl_tot = sum(1 for _, _, l in external if l == "<|nl|>")
    en_tot = sum(1 for _, _, l in external if l == "<|en|>")
    print(f"Total external pairs loaded: {len(external)} (NL: {nl_tot}, EN: {en_tot})")
    return external