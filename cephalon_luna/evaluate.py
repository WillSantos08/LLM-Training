import os
import sys
import json
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

from config.settings import load_config

if "--save" in sys.argv:
    matplotlib.use("Agg")


# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────

def load_history(logs_dir: str) -> dict:
    path = os.path.join(logs_dir, "history.json")
    if not os.path.exists(path):
        print(f"  ❌ Histórico não encontrado : {path}")
        print("  👉 Execute : python cephalon_luna/train.py")
        sys.exit(1)
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────
#  Gráficos
# ─────────────────────────────────────────

def plot_loss_curves(history: dict, ax: plt.Axes):
    epochs     = [h["epoch"]      for h in history["epochs"]]
    train_loss = [h["train_loss"] for h in history["epochs"]]
    val_loss   = [h["val_loss"]   for h in history["epochs"]]

    ax.plot(epochs, train_loss, "o-",
            color="#4C9BE8", linewidth=2, markersize=4,
            label="Train Loss")
    ax.plot(epochs, val_loss, "s--",
            color="#E87B4C", linewidth=2, markersize=4,
            label="Val Loss")

    best_idx = val_loss.index(min(val_loss))
    ax.scatter(
        epochs[best_idx], val_loss[best_idx],
        s=120, zorder=5, color="#E84C4C",
        label=f"Melhor Val ({val_loss[best_idx]:.4f})",
    )

    ax.set_title("Loss por Epoch", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Cross-Entropy)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_perplexity(history: dict, ax: plt.Axes):
    epochs = [h["epoch"]              for h in history["epochs"]]
    ppl    = [math.exp(h["val_loss"]) for h in history["epochs"]]

    ax.fill_between(epochs, ppl, alpha=0.15, color="#9B59B6")
    ax.plot(epochs, ppl, "D-",
            color="#9B59B6", linewidth=2, markersize=4,
            label="Perplexidade (Val)")

    ax.set_title("Perplexidade por Epoch", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Perplexidade")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"  {ppl[-1]:.1f}",
        xy=(epochs[-1], ppl[-1]),
        fontsize=9, color="#9B59B6",
    )


def plot_delta(history: dict, ax: plt.Axes):
    epochs   = [h["epoch"]    for h in history["epochs"]]
    val_loss = [h["val_loss"] for h in history["epochs"]]

    deltas = [0.0] + [
        val_loss[i] - val_loss[i - 1]
        for i in range(1, len(val_loss))
    ]
    colors = ["#2ECC71" if d <= 0 else "#E74C3C" for d in deltas]

    ax.bar(epochs, deltas, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Δ Val Loss (verde = melhorou)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Δ Loss")
    ax.grid(True, alpha=0.3, axis="y")


def plot_lr(history: dict, ax: plt.Axes):
    if "lr" not in history["epochs"][0]:
        ax.text(0.5, 0.5, "LR não registrado",
                ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title("Learning Rate", fontsize=13, fontweight="bold")
        return

    epochs = [h["epoch"] for h in history["epochs"]]
    lrs    = [h["lr"]    for h in history["epochs"]]

    ax.plot(epochs, lrs, "^-", color="#1ABC9C",
            linewidth=2, markersize=4)
    ax.set_title("Learning Rate por Epoch", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, alpha=0.3)


def plot_summary_table(history: dict, ax: plt.Axes):
    epochs = history["epochs"]
    best   = min(epochs, key=lambda h: h["val_loss"])
    last   = epochs[-1]

    rows = [
        ["Total de Epochs",  str(len(epochs))],
        ["Melhor Epoch",     str(best["epoch"])],
        ["Melhor Val Loss",  f"{best['val_loss']:.4f}"],
        ["Melhor PPL",       f"{math.exp(best['val_loss']):.2f}"],
        ["Train Loss Final", f"{last['train_loss']:.4f}"],
        ["Val Loss Final",   f"{last['val_loss']:.4f}"],
        ["PPL Final",        f"{math.exp(last['val_loss']):.2f}"],
    ]

    ax.axis("off")
    table = ax.table(
        cellText  = rows,
        colLabels = ["Métrica", "Valor"],
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for col in range(2):
        table[0, col].set_facecolor("#2C3E50")
        table[0, col].set_text_props(color="white", fontweight="bold")

    best_row = next(
        i + 1 for i, h in enumerate(epochs)
        if h["epoch"] == best["epoch"]
    )
    for col in range(2):
        if best_row < len(rows) + 1:
            table[best_row, col].set_facecolor("#D5F5E3")

    ax.set_title("Resumo do Treinamento",
                 fontsize=13, fontweight="bold", pad=20)


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────

def main():
    print("""
╔══════════════════════════════════════════╗
       🌙  Cephalon Luna — Avaliação       
╚══════════════════════════════════════════╝
    """)

    cfg     = load_config("config.yaml")
    history = load_history(cfg.paths.logs)

    print(f"  📊 Epochs registradas : {len(history['epochs'])}")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "🌙 Cephalon Luna — Evolução do Treinamento",
        fontsize=16, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    plot_loss_curves  (history, fig.add_subplot(gs[0, 0]))
    plot_perplexity   (history, fig.add_subplot(gs[0, 1]))
    plot_delta        (history, fig.add_subplot(gs[0, 2]))
    plot_lr           (history, fig.add_subplot(gs[1, 0]))
    plot_summary_table(history, fig.add_subplot(gs[1, 1:]))

    if "--save" in sys.argv:
        save_path = os.path.join(cfg.paths.logs, "evaluation.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  💾 Gráfico salvo : {save_path}")
    else:
        print("\n  📈 Exibindo gráficos...")
        plt.show()


if __name__ == "__main__":
    main()