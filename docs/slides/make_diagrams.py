"""Architecture diagrams for the deck's section slides, drawn in code so every label is exact.

    python3 docs/slides/make_diagrams.py     writes docs/slides/arch-*.png (3200 x 880 px); the deck uses copies named with their hash

Each diagram is one or two rows of boxes joined by arrows: what runs on the laptop, how the
request travels, and what serves it. Labels come from backends.py, stage/up.sh and the 2026-10-12
rehearsal (results/rehearsal-20261012T1429Z).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
INK, NAVY, ORANGE, TEAL, BG = "#1E1E1E", "#232F3E", "#FF9900", "#1A9C8C", "#EEEEEE"
FILL = {"laptop": "#FFFFFF", "aws": "#FFF1DB", "path": "#E3F4F1", "chip": "#232F3E"}
W, H = 32, 8.8  # data units; a 16 x 4.4 in figure at 200 dpi -> 3200 x 880 px
FIG, DPI = (16, 4.4), 200
plt.rcParams["font.family"] = ["Roboto", "DejaVu Sans"]


def box(ax, x, y, w, h, title, sub, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.35",
                                fc=FILL[kind], ec=NAVY if kind != "path" else TEAL, lw=3))
    color = "white" if kind == "chip" else INK
    ax.text(x + w / 2, y + h - 0.95, title, ha="center", va="center", fontsize=21, weight="bold", color=color)
    ax.text(x + w / 2, y + h - 1.55, sub, ha="center", va="top", fontsize=16, color=color, linespacing=1.25)


def arrow(ax, x0, x1, y, label=""):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=34, lw=3.5, color=ORANGE))
    if label:
        ax.text((x0 + x1) / 2, y + 0.35, label, ha="center", va="bottom", fontsize=15, color=INK)


def row(ax, y, h, items):
    """items: [(title, sub, kind, width, arrow_label_to_next)] laid out left to right."""
    gap = 2.4 if len(items) > 3 else 3.2
    total = sum(i[3] for i in items)
    x = (W - total - gap * (len(items) - 1)) / 2
    for i, (title, sub, kind, w, label) in enumerate(items):
        box(ax, x, y, w, h, title, sub, kind)
        if i < len(items) - 1:
            arrow(ax, x + w + 0.1, x + w + gap - 0.1, y + h / 2, label)
        x += w + gap


def canvas():
    fig = plt.figure(figsize=FIG, dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")
    return fig, ax


def figure(name, rows):
    fig, ax = canvas()
    for y, h, items in rows:
        row(ax, y, h, items)
    path = os.path.join(OUT, f"arch-{name}.png")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(path)


def main():
    # Slide 11: the two managed paths
    figure("managed", [
        (4.8, 3.2, [("Strands agent", "BedrockModel", "laptop", 5.6, "Converse API"),
                    ("Amazon Bedrock", "us-east-1, per token", "aws", 7.4, ""),
                    ("Nova Micro", "managed model", "chip", 5.6, "")]),
        (0.8, 3.2, [("Strands agent", "SageMakerAIModel", "laptop", 5.6, "InvokeEndpoint"),
                    ("SageMaker endpoint", "ml.g6.xlarge, us-east-2\nAWS vLLM 0.31 container", "aws", 7.4, ""),
                    ("NVIDIA L4", "Gemma 4 E2B repack", "chip", 5.6, "")]),
    ])
    # Slide 14: vLLM on EC2 GPUs, reached through SSM
    figure("gpus", [
        (4.8, 3.2, [("Strands agent", "OpenAIModel", "laptop", 5.4, "localhost"),
                    ("SSM port forward", "no inbound ports", "path", 5.8, ""),
                    ("g6.xlarge", "x86_64 host\nvLLM 0.30", "aws", 6.4, ""),
                    ("NVIDIA L4", "24 GB\nE2B repack", "chip", 5.4, "")]),
        (0.8, 3.2, [("Strands agent", "OpenAIModel", "laptop", 5.4, "localhost"),
                    ("SSM port forward", "no inbound ports", "path", 5.8, ""),
                    ("g5g.2xlarge", "Graviton2 Arm host\nvLLM 0.27.2rc0, sm_75", "aws", 6.4, ""),
                    ("NVIDIA T4G", "16 GB\nE2B fp16", "chip", 5.4, "")]),
    ])
    # Slide 17: the hand-ported Neuron server
    figure("neuron", [
        (2.7, 3.4, [("Strands agent", "plain HTTP tool\n(demo 2 only)", "laptop", 5.4, "localhost"),
                    ("SSM port forward", "no inbound ports", "path", 5.8, ""),
                    ("inf2 or trn1", "gemma4-optb:slim\nhand-ported Gemma 4", "aws", 6.4, ""),
                    ("2 NeuronCores", "NeuronCore-v2\nsame image, both chips", "chip", 5.4, "")]),
    ])
    # Slide 21: one agent, six backends, one row each
    fig, ax = canvas()
    box(ax, 0.8, 2.9, 7.0, 3.0, "One Strands agent", "a model object\nper backend", "laptop")
    backends = [("Bedrock", "BedrockModel", "aws"), ("SageMaker", "SageMakerAIModel", "aws"),
                ("EC2 g6, NVIDIA L4", "OpenAIModel", "aws"), ("EC2 g5g, NVIDIA T4G", "OpenAIModel", "aws"),
                ("Inferentia2", "plain HTTP tool", "chip"), ("Trainium", "plain HTTP tool", "chip")]
    h, step = 1.18, 1.4
    for i, (name, cls, kind) in enumerate(backends):
        y = H - 0.4 - h - i * step
        ax.add_patch(FancyBboxPatch((14.5, y), 16.5, h, boxstyle="round,pad=0.02,rounding_size=0.3",
                                    fc=FILL[kind], ec=NAVY, lw=3))
        color = "white" if kind == "chip" else INK
        ax.text(15.1, y + h / 2, name, ha="left", va="center", fontsize=24, weight="bold", color=color)
        ax.text(30.4, y + h / 2, cls, ha="right", va="center", fontsize=19, color=color)
        ax.add_patch(FancyArrowPatch((7.9, 4.4), (14.4, y + h / 2), arrowstyle="-|>", mutation_scale=28,
                                     lw=3, color=ORANGE))
    path = os.path.join(OUT, "arch-overview.png")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    print(path)

if __name__ == "__main__":
    main()
