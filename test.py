import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =========================
# 1. Data
# =========================
models = ["Baseline", "RLVR", "CMAO"]
categories = ["Enumeration", "Backsolve", "Equation", "Tool", "Other"]

counts = {
    "Baseline": [438, 110, 40, 29, 0],
    "RLVR":     [457, 124, 54, 57, 1],
    "CMAO":     [424, 150, 58, 66, 0],
}

# =========================
# 2. Convert to proportions
# =========================
totals = {m: sum(counts[m]) for m in models}
props = {
    m: np.array(counts[m]) / totals[m] * 100
    for m in models
}

# =========================
# 3. Style
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

# Elegant muted palette
colors = {
    "Enumeration": "#4C78A8",  # muted blue
    "Backsolve":   "#F58518",  # orange
    "Equation":    "#54A24B",  # green
    "Tool":        "#B279A2",  # purple
    "Other":       "#9D9D9D",  # gray
}

# =========================
# 4. Plot
# =========================
fig, ax = plt.subplots(figsize=(12, 5.8))
y_positions = np.arange(len(models))

bar_height = 0.58

for i, model in enumerate(models):
    left = 0
    for j, cat in enumerate(categories):
        value = props[model][j]
        if value > 0:
            ax.barh(
                y=i,
                width=value,
                left=left,
                height=bar_height,
                color=colors[cat],
                edgecolor="white",
                linewidth=1.5
            )

            # Label inside segment if segment is large enough
            if value >= 6:
                ax.text(
                    left + value / 2,
                    i,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=10
                )
        left += value

# =========================
# 5. Annotate changes vs Baseline
# =========================
baseline = props["Baseline"]

for i, model in enumerate(models):
    if model == "Baseline":
        continue
    
    # show changes for selected important categories
    important_cats = ["Enumeration", "Backsolve", "Tool"]
    x_text = 103.5
    y0 = i - 0.24
    
    lines = []
    for cat in important_cats:
        idx = categories.index(cat)
        delta = props[model][idx] - baseline[idx]
        sign = "+" if delta >= 0 else ""
        lines.append(f"{cat}: {sign}{delta:.1f} pp")
    
    ax.text(
        x_text,
        y0,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=10.5,
        color="#333333"
    )

# =========================
# 6. Axes and layout
# =========================
ax.set_xlim(0, 118)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_xticklabels([f"{x}%" for x in range(0, 101, 20)])
ax.set_yticks(y_positions)
ax.set_yticklabels(models)
ax.invert_yaxis()  # Baseline on top
ax.set_xlabel("Reasoning Mode Distribution")
ax.set_title("Shift in Reasoning Mode Distribution After Training", pad=16)

# Grid
ax.xaxis.grid(True, linestyle="--", alpha=0.25)
ax.set_axisbelow(True)

# Remove top/right/left spines for cleaner look
for spine in ["top", "right", "left"]:
    ax.spines[spine].set_visible(False)



# Legend
legend_elements = [
    Patch(facecolor=colors[cat], edgecolor="none", label=cat)
    for cat in categories
]
ax.legend(
    handles=legend_elements,
    ncol=5,
    bbox_to_anchor=(0.5, -0.15),
    loc="upper center",
    frameon=False
)

# Subtitle-like note
fig.text(
    0.125, 0.02,
    "Percentages are normalized within each model. Right-side annotations show percentage-point shift relative to Baseline.",
    fontsize=10,
    color="#555555"
)

plt.tight_layout()
plt.savefig("reasoning_mode_distribution.png", dpi=300)
