"""CIFAR-10 classification with pcn-torch.

Run:
    python examples/cifar10.py

Trains a 3-layer Predictive Coding Network on CIFAR-10 using local
Hebbian-like update rules (no backpropagation).

Architecture and hyperparameters match arXiv:2506.06332v1 Section 5.

Requirements (for this example only):
    pip install torchvision
"""

import time

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from pcn_torch import (
    PredictiveCodingNetwork,
    RichCallback,
    TrainConfig,
    test_pcn,
    train_pcn,
)

# ---------------------------------------------------------------------------
# Configuration -- matches arXiv:2506.06332v1 Section 5
# ---------------------------------------------------------------------------
BATCH_SIZE = 500
NUM_EPOCHS = 3
T_INFER = 50  # Training inference steps (supervised, with labels clamped)
T_INFER_TEST = 500  # Test inference steps (unsupervised, needs more steps)
LR_INFER = 0.05
LR_LEARN = 0.05
# Manual LR schedule: (epoch, batch, lr) breakpoints
# Start aggressive at 0.05, drop to 0.005 at batch 20 of epoch 0
LR_LEARN_STEPS: list[tuple[int, int, float]] = [
    (0, 0, 0.05),
    (0, 20, 0.005),
]
DATA_ROOT = "./data"


def load_cifar10() -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 and return train and test DataLoaders."""
    # Paper normalizes to [0,1] only (ToTensor does this).
    # No per-channel mean/std normalization.
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    try:
        train_set = torchvision.datasets.CIFAR10(
            root=DATA_ROOT,
            train=True,
            download=True,
            transform=transform,
        )
        test_set = torchvision.datasets.CIFAR10(
            root=DATA_ROOT,
            train=False,
            download=True,
            transform=transform,
        )
    except Exception as exc:
        raise SystemExit(
            f"Failed to load CIFAR-10 dataset: {exc}\n"
            "Check your internet connection or download manually."
        ) from exc

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def build_model() -> PredictiveCodingNetwork:
    """Build the CIFAR-10 PCN model.

    Architecture from arXiv:2506.06332v1:
        input(3072) -> hidden(1000) -> hidden(500) -> top(10) -> readout(10)

    The top latent is 10-dimensional (same as number of classes).
    The readout is a 10x10 matrix mapping top latent to output.
    """
    return PredictiveCodingNetwork(
        dims=[3072, 1000, 500, 10],
        activation="relu",
        output_dim=10,
        mode="classification",
    )


def main() -> None:
    """Train a PCN on CIFAR-10 and print a summary."""
    print("Loading CIFAR-10...")  # noqa: T201
    train_loader, test_loader = load_cifar10()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building model on {device}: dims=[3072, 1000, 500, 10], output_dim=10")  # noqa: T201
    model = build_model().to(device)

    config = TrainConfig(
        task="classification",
        T_infer=T_INFER,
        T_infer_test=T_INFER_TEST,
        lr_infer=LR_INFER,
        lr_learn=LR_LEARN,
        num_epochs=NUM_EPOCHS,
        lr_learn_steps=LR_LEARN_STEPS,
        callback=RichCallback(device=str(device)),
    )

    # Train
    start = time.time()
    history = train_pcn(model, train_loader, config)
    elapsed = time.time() - start

    # Evaluate on test set
    results = test_pcn(model, test_loader, config)

    # Print summary
    print()  # noqa: T201
    print("=" * 52)  # noqa: T201
    print("  CIFAR-10 Training Summary")  # noqa: T201
    print("=" * 52)  # noqa: T201
    accuracy = results["accuracy"]
    print(f"  Final test accuracy:  {accuracy:.1%}")  # noqa: T201
    minutes = elapsed / 60
    print(f"  Total training time: {minutes:.1f} minutes")  # noqa: T201
    if history.energy.per_epoch:
        e_first = history.energy.per_epoch[0]
        e_last = history.energy.per_epoch[-1]
        print(f"  Energy (epoch 1):    {e_first:.6f}")  # noqa: T201
        print(f"  Energy (final):      {e_last:.6f}")  # noqa: T201
        convergence = e_first - e_last
        print(f"  Energy reduction:    {convergence:.6f}")  # noqa: T201
    if history.lr_learn_per_epoch:
        lr_start = history.lr_learn_per_epoch[0]
        lr_final = history.lr_learn_per_epoch[-1]
        if lr_final < lr_start:
            print(f"  LR schedule:         {lr_start:.6f} -> {lr_final:.6f}")  # noqa: T201
        else:
            print(f"  LR (unchanged):      {lr_start:.6f}")  # noqa: T201
    print("=" * 52)  # noqa: T201


if __name__ == "__main__":
    main()
