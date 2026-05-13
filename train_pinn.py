"""Train a PINN checkpoint from the project's CFD dataset pipeline.

This script bootstraps the existing development pipeline:
1. Create a synthetic UCM artifact.
2. Generate a CFD dataset through ``CFDRunner.generate_training_dataset``.
3. Train ``PINNModel`` on that dataset.
4. Save a versioned checkpoint into ``checkpoints/`` (or a custom location).

The resulting ``.pt`` file can be picked up automatically by the Governance API.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cfd.runner import CFDRunner
from src.pinn.model import PINNModel
from src.provenance.store import ProvenanceStore
from src.shared.config import CFDConfig, Config, PINNConfig
from src.shared.types import UCMArtifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save a PINN checkpoint.")
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of CFD samples to generate for training (default: 200).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of PINN training epochs (default: 300).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="PINN learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory to write the trained checkpoint into (default: config checkpoint_dir).",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Optional explicit checkpoint filename. Defaults to the model's versioned naming.",
    )
    parser.add_argument(
        "--encoder-layers",
        type=int,
        default=8,
        help="PINN encoder layer count (default: 8).",
    )
    parser.add_argument(
        "--encoder-width",
        type=int,
        default=512,
        help="PINN encoder hidden width (default: 512).",
    )
    parser.add_argument(
        "--mass-conservation-threshold",
        type=float,
        default=3.5,
        help=(
            "Validation threshold for synthetic CFD samples. "
            "The repo's production default is 1e-4, but the development generator "
            "needs a looser threshold to retain usable samples (default: 3.5)."
        ),
    )
    return parser.parse_args()


def build_synthetic_ucm() -> UCMArtifact:
    return UCMArtifact(
        udt_version_id="synthetic-udt-dev",
        block_count=20,
        canyon_class_distribution={
            "deep_canyon": 5,
            "regular_canyon": 5,
            "shallow_canyon": 5,
            "transitional_zone": 5,
        },
    )


def resolve_checkpoint_dir(config: Config, requested_dir: Path | None) -> Path:
    if requested_dir is not None:
        return requested_dir.resolve()
    return (Path(__file__).resolve().parent / config.checkpoint_dir).resolve()


def main() -> int:
    args = parse_args()
    config = Config.from_env()
    provenance = ProvenanceStore()

    checkpoint_dir = resolve_checkpoint_dir(config, args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cfd_runner = CFDRunner(
        config=CFDConfig(
            n_samples=args.samples,
            mass_conservation_threshold=args.mass_conservation_threshold,
        ),
        provenance_store=provenance,
    )
    pinn_model = PINNModel(
        input_dim=10,
        config=PINNConfig(
            encoder_layers=args.encoder_layers,
            encoder_width=args.encoder_width,
            learning_rate=args.learning_rate,
            max_epochs=args.epochs,
        ),
        provenance_store=provenance,
    )

    print(f"Generating CFD dataset with {args.samples} samples...")
    ucm_artifact = build_synthetic_ucm()
    dataset = cfd_runner.generate_training_dataset(
        ucm_artifact=ucm_artifact,
        blocks=None,
        n_samples=args.samples,
    )
    print(
        "CFD dataset ready:",
        f"version={dataset.version_id}",
        f"valid={dataset.valid_sample_count}",
        f"failed={dataset.failed_sample_count}",
    )
    if dataset.valid_sample_count == 0:
        raise RuntimeError(
            "No valid CFD samples were retained. "
            "Increase --mass-conservation-threshold for the synthetic generator."
        )

    print(f"Training PINN for {args.epochs} epochs...")
    losses = pinn_model.train_on_cfd_data(
        cfd_dataset=dataset,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    print("Training complete.")
    print(
        "Final losses:",
        ", ".join(f"{name}={value:.6f}" for name, value in sorted(losses.items())),
    )

    checkpoint_path = checkpoint_dir / args.checkpoint_name if args.checkpoint_name else checkpoint_dir / "pinn_latest.pt"
    artifact = pinn_model.save_checkpoint(checkpoint_path)

    print("Checkpoint saved.")
    print(f"Path: {artifact.file_path}")
    print(f"Version: {artifact.version_id}")
    print("")
    print("Next steps:")
    print("1. Start the API again so it reloads the checkpoint.")
    print(f"2. Keep the file in {checkpoint_dir} or set PINN_CHECKPOINT_PATH explicitly.")
    print("3. Call /v1/pinn/predict and check model_source=trained_checkpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
