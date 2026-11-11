import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

os.makedirs(os.path.join("docs", "figures"), exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

# 1. Parse splits
data_dir = os.path.join("data", "quality")
splits = ["train", "dev", "test"]

records = []
for split in splits:
    file_path = os.path.join(data_dir, f"QuALITY.v1.0.1.htmlstripped.{split}")
    if not os.path.exists(file_path):
        continue
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            art_words = len(d.get("article", "").split())
            qs = d.get("questions", [])
            for q in qs:
                records.append(
                    {
                        "split": split.upper(),
                        "article_id": d.get("article_id"),
                        "article_words": art_words,
                        "question_id": q.get("question_unique_id"),
                        "gold_label": q.get("gold_label"),
                        "difficult": bool(q.get("difficult", 0)),
                    }
                )

df = pd.DataFrame(records)

# Plot 1: Article Word Count Distribution (Long-Context nature)
plt.figure(figsize=(9, 5))
article_df = df.drop_duplicates(subset=["article_id"])
sns.histplot(article_df["article_words"], kde=True, bins=30, color="#1f77b4")
plt.axvline(
    article_df["article_words"].mean(),
    color="red",
    linestyle="--",
    label=f"Mean Length ({int(article_df['article_words'].mean())} words)",
)
plt.title(
    "QuALITY Dataset: Article Word Count Distribution (Long Context)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Words per Article", fontsize=12)
plt.ylabel("Number of Articles", fontsize=12)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("docs/figures/article_length_distribution.png", dpi=300)
plt.close()

# Plot 2: Split Overview (Articles and Questions count)
plt.figure(figsize=(9, 5))
split_stats = (
    df.groupby("split")
    .agg(Articles=("article_id", "nunique"), Questions=("question_id", "count"))
    .reset_index()
)

tidy_split = pd.melt(
    split_stats,
    id_vars=["split"],
    value_vars=["Articles", "Questions"],
    var_name="Type",
    value_name="Count",
)

ax = sns.barplot(data=tidy_split, x="split", y="Count", hue="Type", palette=["#2ca02c", "#ff7f0e"])
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(
            f"{int(h)}",
            (p.get_x() + p.get_width() / 2.0, h / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11,
        )

plt.title("QuALITY Dataset: Train / Validation / Test Breakdown", fontsize=14, fontweight="bold")
plt.xlabel("Dataset Split", fontsize=12)
plt.ylabel("Total Count", fontsize=12)
plt.tight_layout()
plt.savefig("docs/figures/dataset_splits_distribution.png", dpi=300)
plt.close()

# Plot 3: Difficulty & Gold Label Uniformity
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Subplot 3a: Difficulty Ratio
diff_counts = df["difficult"].value_counts(normalize=True) * 100
axes[0].pie(
    diff_counts,
    labels=["Standard (Skimmable)", "Difficult (Deep Reasoning)"],
    autopct="%1.1f%%",
    colors=["#aec7e8", "#d62728"],
    startangle=140,
    wedgeprops=dict(width=0.6, edgecolor="white"),
)
axes[0].set_title(
    "Question Difficulty Distribution\n(Untimed vs 45s Speed-Skimming)", fontweight="bold"
)

# Subplot 3b: Option Label Distribution
sns.countplot(data=df, x="gold_label", ax=axes[1], palette="crest")
axes[1].set_title("Gold Label Option Balance (1 to 4)", fontweight="bold")
axes[1].set_xlabel("Correct Option Index", fontsize=11)
axes[1].set_ylabel("Question Count", fontsize=11)

plt.tight_layout()
plt.savefig("docs/figures/difficulty_and_labels.png", dpi=300)
plt.close()

print("Plots successfully generated in 'figures/' directory.")
