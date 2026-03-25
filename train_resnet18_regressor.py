#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_TURN_THRESHOLD = 0.1


class DrivingRegressionDataset(Dataset):
    def __init__(self, dataset_dir: Path, transform: transforms.Compose) -> None:
        self.dataset_dir = dataset_dir
        self.images_dir = dataset_dir / "images"
        self.labels_path = dataset_dir / "labels.csv"
        self.transform = transform
        self.samples: list[tuple[Path, tuple[float, float]]] = []

        if not self.labels_path.exists():
            raise FileNotFoundError(f"labels.csv not found: {self.labels_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"images directory not found: {self.images_dir}")

        with self.labels_path.open(newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                image_num = row["image_num"].strip()
                image_path = self.images_dir / f"{image_num}.jpg"
                if not image_path.exists():
                    continue
                linear = float(row["linear"])
                angular = float(row["angular"])
                self.samples.append((image_path, (linear, angular)))

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {dataset_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, target_values = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        target = torch.tensor(target_values, dtype=torch.float32)
        return image, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a pretrained ResNet18 regressor that predicts (linear, angular) from train/images and train/labels.csv.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("train"), help="Dataset directory with images/ and labels.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints_resnet18"), help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of data used for validation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help='Device: "auto", "cpu", or "cuda"')
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights")
    parser.add_argument(
        "--turn-threshold",
        type=float,
        default=DEFAULT_TURN_THRESHOLD,
        help="Threshold used to classify straight vs left/right for the weighted sampler.",
    )
    parser.add_argument(
        "--disable-balanced-sampler",
        action="store_true",
        help="Disable weighted oversampling of straight/left/right classes in the training loader.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def describe_device(device: torch.device) -> None:
    cuda_available = torch.cuda.is_available()
    print(f"Requested device mode: {device}")
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if device.type == "cuda":
        device_index = device.index if device.index is not None else 0
        gpu_name = torch.cuda.get_device_name(device_index)
        print(f"Using GPU: cuda:{device_index} ({gpu_name})")
    else:
        print("Using CPU for training")


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def split_dataset(dataset: Dataset, val_ratio: float, seed: int):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation ratio")

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def categorize_angular(angular: float, turn_threshold: float) -> str:
    if angular >= turn_threshold:
        return "left"
    if angular <= -turn_threshold:
        return "right"
    return "straight"


def build_balanced_train_sampler(train_subset, turn_threshold: float) -> tuple[WeightedRandomSampler, dict[str, int]]:
    counts = {"straight": 0, "left": 0, "right": 0}
    categories: list[str] = []

    for sample_index in train_subset.indices:
        _, (_, angular) = train_subset.dataset.samples[sample_index]
        category = categorize_angular(angular, turn_threshold)
        categories.append(category)
        counts[category] += 1

    weights = []
    for category in categories:
        class_count = counts[category]
        if class_count == 0:
            weights.append(0.0)
        else:
            weights.append(1.0 / class_count)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    return sampler, counts


def create_model(use_pretrained: bool) -> nn.Module:
    try:
        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        model = resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load pretrained ResNet18 weights. If this is an offline environment, rerun with --no-pretrained."
        ) from exc

    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def print_batch_progress(
    split_name: str,
    epoch: int,
    total_epochs: int,
    batch_idx: int,
    total_batches: int,
    current_loss: float,
    start_time: float,
) -> None:
    elapsed = time.perf_counter() - start_time
    progress_pct = batch_idx / total_batches * 100
    message = (
        f"\r[{split_name}] epoch {epoch}/{total_epochs} | "
        f"batch {batch_idx}/{total_batches} ({progress_pct:5.1f}%) | "
        f"current_loss={current_loss:.6f} | elapsed={elapsed:7.1f}s"
    )
    sys.stdout.write(message)
    sys.stdout.flush()


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    train_mode: bool,
    epoch: int,
    total_epochs: int,
):
    if train_mode:
        model.train()
        split_name = "train"
    else:
        model.eval()
        split_name = "val"

    total_loss = 0.0
    total_mae = torch.zeros(2, dtype=torch.float64)
    total_count = 0
    epoch_start = time.perf_counter()
    total_batches = len(loader)

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch_idx, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            if train_mode:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_mae += (outputs - targets).abs().sum(dim=0).detach().cpu().to(torch.float64)
            total_count += batch_size

            print_batch_progress(split_name, epoch, total_epochs, batch_idx, total_batches, loss.item(), epoch_start)

    print()
    avg_loss = total_loss / total_count
    avg_mae = (total_mae / total_count).tolist()
    epoch_elapsed = time.perf_counter() - epoch_start
    return avg_loss, avg_mae, epoch_elapsed


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform()
    full_dataset = DrivingRegressionDataset(args.data_dir, transform)
    train_dataset, val_dataset = split_dataset(full_dataset, args.val_ratio, args.seed)

    pin_memory = device.type == "cuda"
    train_sampler = None
    train_class_counts = None
    if not args.disable_balanced_sampler:
        train_sampler, train_class_counts = build_balanced_train_sampler(train_dataset, args.turn_threshold)

    train_loader = build_dataloader(
        train_dataset,
        args.batch_size,
        args.num_workers,
        shuffle=True,
        pin_memory=pin_memory,
        sampler=train_sampler,
    )
    val_loader = build_dataloader(val_dataset, args.batch_size, args.num_workers, False, pin_memory)

    model = create_model(use_pretrained=not args.no_pretrained).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    total_start = time.perf_counter()

    describe_device(device)
    print(f"Dataset: {args.data_dir.resolve()}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Train batches/epoch: {len(train_loader)}")
    print(f"Val batches/epoch: {len(val_loader)}")
    print("Image preprocessing: uint8 -> float32, 0~1 scaling, ImageNet normalization")
    print(f"Turn threshold for sampler: {args.turn_threshold}")
    if train_sampler is None:
        print("Train sampler: standard shuffled batches")
    else:
        print("Train sampler: WeightedRandomSampler enabled")
        print(
            "Train class counts: "
            f"straight={train_class_counts['straight']}, "
            f"left={train_class_counts['left']}, "
            f"right={train_class_counts['right']}"
        )

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss, train_mae, train_elapsed = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train_mode=True,
            epoch=epoch,
            total_epochs=args.epochs,
        )
        val_loss, val_mae, val_elapsed = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train_mode=False,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mae_linear": train_mae[0],
            "train_mae_angular": train_mae[1],
            "val_mae_linear": val_mae[0],
            "val_mae_angular": val_mae[1],
            "train_elapsed_sec": train_elapsed,
            "val_elapsed_sec": val_elapsed,
        }
        history.append(epoch_result)
        save_history(history, output_dir)

        latest_ckpt = output_dir / "latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "args": vars(args),
            },
            latest_ckpt,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = output_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "args": vars(args),
                },
                best_ckpt,
            )

        total_elapsed = time.perf_counter() - total_start
        print(
            f"Epoch {epoch:02d}/{args.epochs} done | "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} | "
            f"train_mae(linear={train_mae[0]:.6f}, angular={train_mae[1]:.6f}) | "
            f"val_mae(linear={val_mae[0]:.6f}, angular={val_mae[1]:.6f}) | "
            f"train_time={train_elapsed:.1f}s val_time={val_elapsed:.1f}s total_elapsed={total_elapsed:.1f}s"
        )

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
