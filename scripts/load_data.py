from __future__ import annotations

import shutil
import kagglehub

from pathlib import Path


DATASET = "karnikakapoor/digits"


def find_digits_directory(dataset_path: Path) -> Path:
    """Finds the 'digits updated' directory inside the downloaded dataset."""

    directories = [
        path
        for path in dataset_path.rglob("digits updated/digits updated")
        if  path.is_dir()
    ]

    if not directories:
        raise FileNotFoundError(
            "Could not find the 'digits updated' directory in the downloaded dataset."
        )

    return directories[0]


def download_data() -> Path:
    """
    Downloads the MNIST digits dataset and copies classes 0-9 to ./digits.

    The destination is relative to the directory from which this script is executed.
    """

    root_directory   = Path.cwd()
    output_directory = root_directory / "digits"

    print("Downloading dataset...")

    dataset_path     = Path(
        kagglehub.dataset_download(DATASET)
    )
    source_directory = find_digits_directory(dataset_path)

    print(f"Dataset downloaded to: {dataset_path    }")
    print(f"Source directory     : {source_directory}")

    output_directory.mkdir(parents=True, exist_ok=True)

    for digit in range(10):
        source      = source_directory / str(digit)
        destination = output_directory / str(digit)

        if not source.is_dir():
            raise FileNotFoundError(
                f"Digit directory not found: {source}"
            )

        shutil.copytree(
            source     ,
            destination,
            dirs_exist_ok=True,
        )

        print(f"Copied digit {digit}.")

    shutil.rmtree(dataset_path)

    print(f"Dataset available at: {output_directory}")

    return output_directory


if __name__ == "__main__":
    download_data()
