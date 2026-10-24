import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer
from safetensors.torch import load_file

from hellgpt.data import (
    DiscordDataProcessor,
    download_default_exports,
    load_external_bilingual_conversations,
)
from hellgpt.tokenizer import train_server_tokenizer, DiscordTakeDataset
from hellgpt.model import (
    MiniDiscordGPT,
    HFMiniDiscordGPTForCausalLM,
    export_raw_safetensors,
    export_huggingface_format,
)
from hellgpt.train import run_two_stage_training
from hellgpt.generate import generate_take


def cmd_train(args: argparse.Namespace) -> None:
    processor = DiscordDataProcessor()

    if args.download_defaults:
        json_files = download_default_exports(Path(args.data_dir))
    else:
        json_files = [Path(p) for p in args.files]

    if not json_files:
        raise ValueError("Provide Discord JSON files via --files or pass --download-defaults.")

    discord_pairs = processor.process_files(json_files)

    external_pairs = []
    if not args.skip_stage1:
        external_pairs = load_external_bilingual_conversations(
            processor, target_per_lang=args.external_limit
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "discord_tokenizer.json"

    tokenizer = train_server_tokenizer(
        discord_pairs,
        external_pairs,
        vocab_size=args.vocab_size,
        save_path=tok_path,
    )

    discord_ds = DiscordTakeDataset(discord_pairs, tokenizer, max_len=args.max_seq_len)
    external_ds = (
        DiscordTakeDataset(external_pairs, tokenizer, max_len=args.max_seq_len)
        if external_pairs
        else None
    )

    model = MiniDiscordGPT(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
    )

    ckpt_path = out_dir / "discord_mini_gpt.pt"
    model = run_two_stage_training(
        model=model,
        discord_dataset=discord_ds,
        external_dataset=external_ds,
        stage1_epochs=0 if args.skip_stage1 else args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        stage1_lr=args.stage1_lr,
        stage2_lr=args.stage2_lr,
        patience=args.patience,
        checkpoint_path=ckpt_path,
    )

    # Export in the requested format(s)
    if args.export_format in ("safetensors", "both"):
        export_raw_safetensors(model, out_dir / "discord_mini_gpt.safetensors")

    if args.export_format in ("hf", "both"):
        hf_dir = out_dir / "hf_model" if args.export_format == "both" else out_dir
        export_huggingface_format(model, tokenizer, hf_dir)


def cmd_generate(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model_path)

    # Check if loading from a Hugging Face directory or raw safetensors
    if model_path.is_dir() and (model_path / "config.json").exists():
        hf_model = HFMiniDiscordGPTForCausalLM.from_pretrained(str(model_path)).to(device)
        model = hf_model.model
        tok_file = model_path / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tok_file))
    else:
        tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
        model = MiniDiscordGPT(
            vocab_size=tokenizer.get_vocab_size(),
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        ).to(device)
        state_dict = load_file(str(model_path))
        state_dict["head.weight"] = state_dict["tok_emb.weight"]
        model.load_state_dict(state_dict)

    for prompt in args.prompts:
        generate_take(
            model=model,
            tokenizer=tokenizer,
            user_message=prompt,
            force_lang=args.force_lang,
            temperature=args.temperature,
            min_p=args.min_p,
            num_candidates=args.candidates,
            show_candidates=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="bingi-gpt", description="Bingi Discord Mini-GPT CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- TRAIN SUBCOMMAND ---
    p_train = subparsers.add_parser("train", help="Train the model and export weights")
    p_train.add_argument("--files", nargs="*", default=[], help="Paths to DiscordChatExporter JSON files")
    p_train.add_argument("--download-defaults", action="store_true", help="Download the 5 default Bingi JSON files")
    p_train.add_argument("--data-dir", default="./data", help="Folder to store downloaded JSON files")
    p_train.add_argument("--output-dir", default="./artifacts", help="Output folder for weights and tokenizer")
    p_train.add_argument(
        "--export-format",
        choices=["safetensors", "hf", "both"],
        default="both",
        help="Output format: 'safetensors' (flat for Rust), 'hf' (HuggingFace Transformers), or 'both'",
    )
    p_train.add_argument("--skip-stage1", action="store_true", help="Skip Stage 1 external Reddit/HF pre-training")
    p_train.add_argument("--external-limit", type=int, default=40000, help="Max external pairs per language")
    p_train.add_argument("--vocab-size", type=int, default=4096)
    p_train.add_argument("--max-seq-len", type=int, default=96)
    p_train.add_argument("--d-model", type=int, default=512)
    p_train.add_argument("--n-layers", type=int, default=6)
    p_train.add_argument("--n-heads", type=int, default=8)
    p_train.add_argument("--dropout", type=float, default=0.10)
    p_train.add_argument("--stage1-epochs", type=int, default=15)
    p_train.add_argument("--stage2-epochs", type=int, default=25)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--stage1-lr", type=float, default=1e-3)
    p_train.add_argument("--stage2-lr", type=float, default=3e-4)
    p_train.add_argument("--patience", type=int, default=6)
    p_train.set_defaults(func=cmd_train)

    # --- GENERATE SUBCOMMAND ---
    p_gen = subparsers.add_parser("generate", help="Generate takes from a trained model")
    p_gen.add_argument("prompts", nargs="+", help="One or more input messages to react to")
    p_gen.add_argument("--model-path", default="./artifacts/hf_model", help="Path to HF model dir or .safetensors file")
    p_gen.add_argument("--tokenizer-path", default="./artifacts/discord_tokenizer.json", help="Path to tokenizer.json (if using raw safetensors)")
    p_gen.add_argument("--force-lang", choices=["<|nl|>", "<|en|>"], default=None)
    p_gen.add_argument("--temperature", type=float, default=0.85)
    p_gen.add_argument("--min-p", type=float, default=0.06)
    p_gen.add_argument("--candidates", type=int, default=5)
    p_gen.add_argument("--max-seq-len", type=int, default=96)
    p_gen.add_argument("--d-model", type=int, default=512)
    p_gen.add_argument("--n-layers", type=int, default=6)
    p_gen.add_argument("--n-heads", type=int, default=8)
    p_gen.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()