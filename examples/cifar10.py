"""CIFAR-10 classification with pcn-torch.

Run:
    python examples/cifar10.py

Trains a 3-hidden-layer Predictive Coding Network on CIFAR-10 using local
Hebbian-like update rules (no backpropagation). Achieves approximately
45-55% test accuracy in 5-15 minutes on CPU.

Requirements (for this example only):
    pip install torchvision
"""

import time

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
# Configuration (fixed -- no CLI arguments by design)
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 10
T_INFER = 20  # 20 steps (paper uses 50; 20 saves ~60% CPU time)
LR_INFER = 0.05
LR_LEARN = 0.001
DATA_ROOT = "./data"

# CIFAR-10 channel statistics (per-channel mean and std)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def load_cifar10() -> tuple[DataLoader, DataLoader]:
    """Download CIFAR-10 and return train and test DataLoaders."""
    # NOTE: No Flatten() transform -- trainer.py calls view(B, -1).
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
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

    Architecture:
        input(3072) -> hidden(1024) -> hidden(1024)
        -> top(512) -> readout(10)

    dims[0] = input dimension (3 x 32 x 32 = 3072)
    dims[-1] = top latent dimension (512)
    output_dim = number of classes (10)
    """
    return PredictiveCodingNetwork(
        dims=[3072, 1024, 1024, 512],
        activation="relu",
        output_dim=10,
        mode="classification",
    )


def main() -> None:
    """Train a PCN on CIFAR-10 and print a summary."""
    print("Loading CIFAR-10...")  # noqa: T201
    train_loader, test_loader = load_cifar10()

    msg = "Building model: dims=[3072, 1024, 1024, 512], output_dim=10"
    print(msg)  # noqa: T201
    model = build_model()

    config = TrainConfig(
        task="classification",
        T_infer=T_INFER,
        lr_infer=LR_INFER,
        lr_learn=LR_LEARN,
        num_epochs=NUM_EPOCHS,
        early_stop_threshold=1e-4,
        callback=RichCallback(),
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
    print("=" * 52)  # noqa: T201


if __name__ == "__main__":
    main()
