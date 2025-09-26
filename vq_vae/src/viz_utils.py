from matplotlib import pyplot as plt
import numpy as np

from src.utils import preprocess_weights

def plot_n_pairs(x, recon, n):
    # Plot n pairs in rows: left = original, right = reconstruction
    fig, axes = plt.subplots(n, 2, figsize=(6, 3*n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i in range(n):
        orig = x[i].permute(1,2,0).clamp(0,1).numpy()
        rec  = recon[i].permute(1,2,0).clamp(0,1).numpy()

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title("original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(rec)
        axes[i, 1].set_title("reconstruction")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

def plot_n(x, n):
    """Simple grid of up to n RGB torch images (B,3,H,W). Max 5 per row."""
    # This code is written by ChatGPT-5
    per_row = 5
    rows = (n + per_row - 1) // per_row
    cols = min(per_row, n)

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    # flatten axes to 1D array for uniform indexing
    axes = np.atleast_1d(axes).flatten()

    for i in range(rows * cols):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue
        img = x[i].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()  # HWC
        ax.imshow(img)

    plt.tight_layout()
    plt.show()

def visualize_epochs(losses: list, epochs: list, out_path: str | None = None, show: bool = True):
    plt.figure(figsize=(6,4))
    plt.plot(epochs, losses, marker="o", linestyle="-")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.grid(True)
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def visualize_distribution(data_loader):
    # collect (B,2) arrays and concatenate
    cats_list = []
    for x, c in data_loader:
        c2 = preprocess_weights(c, idxs=[1, 3])  # -> tensor [B,2]
        cats_list.append(c2.cpu().numpy())

    if len(cats_list) == 0:
        print("No data found in data_loader")
        return

    cats = np.concatenate(cats_list, axis=0)   # shape [N,2]
    unique_rows, counts = np.unique(cats, axis=0, return_counts=True)

    # make readable labels like "(2,9)"
    labels = [f"({int(r[0])},{int(r[1])})" for r in unique_rows]

    # sort by counts descending
    order = np.argsort(-counts)
    labels = [labels[i] for i in order]
    counts = counts[order]

    x = np.arange(len(labels))
    plt.figure(figsize=(max(8, len(labels)*0.25), 4))
    plt.bar(x, counts)
    plt.xticks(x, labels, rotation=90)
    plt.xlabel("Scores for (Moira, Fernando)")
    plt.ylabel("Number of occurrences")
    plt.title("Distribution of scores for (Moira, Fernando)")
    plt.tight_layout()
    plt.show()
