"""Download the TurnBench dev set into data/ (git-ignored).

    uv run hf auth login        # once; the dataset is gated, accept its terms on Hugging Face first
    uv run python scripts/download_data.py
"""

from turn_detector.data import download_dev_set

if __name__ == "__main__":
    print(f"downloaded to {download_dev_set()}")
