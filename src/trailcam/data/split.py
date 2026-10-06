"""Image-level train/validation/test splits for cropped animal images."""


def split_dataset(
    interim_dir: str,
    processed_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Split crops by original image so near-duplicates stay in one fold.

    Not implemented yet.
    """
    raise NotImplementedError("Dataset splitting is not implemented yet.")
