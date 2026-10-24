from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from hellgpt.model import MiniDiscordGPT


def evaluate_val_loss(model: MiniDiscordGPT, val_loader: DataLoader, device: str, amp_dtype) -> float:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _, y_take in val_loader:
            x, y_take = x.to(device), y_take.to(device)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(x, y_take_only=y_take)
            val_loss += loss.item()
    return val_loss / max(1, len(val_loader))


def run_two_stage_training(
    model: MiniDiscordGPT,
    discord_dataset,
    external_dataset=None,
    stage1_epochs: int = 15,
    stage2_epochs: int = 25,
    batch_size: int = 64,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 3e-4,
    patience: int = 6,
    checkpoint_path: str | Path = "discord_mini_gpt.pt",
    device: str = None,
) -> MiniDiscordGPT:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    amp_dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    model = model.to(device)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | Device: {device}")

    val_size = max(1, int(0.1 * len(discord_dataset)))
    train_size = len(discord_dataset) - val_size
    discord_train_ds, discord_val_ds = random_split(
        discord_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    stage2_loader = DataLoader(discord_train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"))
    val_loader = DataLoader(discord_val_ds, batch_size=batch_size, shuffle=False, pin_memory=(device == "cuda"))

    decay_params = [p for _, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for _, p in model.named_parameters() if p.dim() < 2]

    # ========================================================================
    # STAGE 1: Base Bilingual Grammar Pre-Training
    # ========================================================================
    if external_dataset is not None and len(external_dataset) > 0 and stage1_epochs > 0:
        stage1_combined_ds = ConcatDataset([external_dataset, discord_train_ds])
        stage1_loader = DataLoader(stage1_combined_ds, batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"))

        print(f"\n=== STAGE 1: Pre-training on External + Discord ({len(stage1_combined_ds)} samples) ===")
        s1_groups = [
            {"params": decay_params, "weight_decay": 0.05, "lr": stage1_lr},
            {"params": nodecay_params, "weight_decay": 0.0, "lr": stage1_lr},
        ]
        optimizer = torch.optim.AdamW(s1_groups, lr=stage1_lr, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=stage1_lr, epochs=stage1_epochs, steps_per_epoch=len(stage1_loader), pct_start=0.1
        )

        for epoch in range(stage1_epochs):
            model.train()
            train_loss = 0.0
            for x, y_full, y_take in stage1_loader:
                x, y_full, y_take = x.to(device), y_full.to(device), y_take.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                    _, loss = model(x, y_full=y_full, y_take_only=y_take)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            avg_val = evaluate_val_loss(model, val_loader, device, amp_dtype)
            print(
                f"Stage 1 Epoch {epoch+1:02d}/{stage1_epochs:02d} | "
                f"Train (Smoothed): {train_loss/len(stage1_loader):.4f} | Discord Val Loss: {avg_val:.4f}"
            )

    # ========================================================================
    # STAGE 2: Discord Server Fine-Tuning
    # ========================================================================
    print(f"\n=== STAGE 2: Fine-tuning on Discord + 15% Replay Buffer ===")
    if external_dataset is not None and len(external_dataset) > 0:
        replay_size = min(len(external_dataset), int(0.18 * len(discord_train_ds)))
        replay_subset, _ = random_split(
            external_dataset,
            [replay_size, len(external_dataset) - replay_size],
            generator=torch.Generator().manual_seed(42),
        )
        stage2_ds = ConcatDataset([discord_train_ds, replay_subset])
    else:
        stage2_ds = discord_train_ds

    stage2_loader = DataLoader(stage2_ds, batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"))
    s2_groups = [
        {"params": decay_params, "weight_decay": 0.1, "lr": stage2_lr},
        {"params": nodecay_params, "weight_decay": 0.0, "lr": stage2_lr},
    ]
    optimizer = torch.optim.AdamW(s2_groups, lr=stage2_lr, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage2_epochs * len(stage2_loader), eta_min=1e-5
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(stage2_epochs):
        model.train()
        train_loss = 0.0
        for x, y_full, y_take in stage2_loader:
            x, y_full, y_take = x.to(device), y_full.to(device), y_take.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(x, y_full=y_full, y_take_only=y_take)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(stage2_loader)
        avg_val_loss = evaluate_val_loss(model, val_loader, device, amp_dtype)

        saved_marker = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            saved_marker = " * [Saved Best]"
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Stage 2 Epoch {epoch+1:02d}/{stage2_epochs:02d} | "
            f"Train (Smoothed): {avg_train_loss:.4f} | Discord Val Loss: {avg_val_loss:.4f}{saved_marker}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at Stage 2 Epoch {epoch+1}! Best Val Loss: {best_val_loss:.4f}")
            break

    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model