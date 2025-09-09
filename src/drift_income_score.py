
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# "Income" feature: prod shifted higher mean
train_income = np.random.normal(50000, 8000, 1000)
prod_income = np.random.normal(55000, 9000, 1000)

# "Score": prod distribution shifted lower quality
train_score = np.random.beta(5, 3, 1000)  # mean ~0.62
prod_score = np.random.beta(4, 4, 1000)   # mean ~0.50

# Two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

# Income drift
axes[0].hist(train_income, bins=30, alpha=0.6, label="Train", density=True)
axes[0].hist(prod_income, bins=30, alpha=0.6, label="Prod", density=True)
axes[0].set_title("Data Drift: Income")
axes[0].set_xlabel("Income")
axes[0].set_ylabel("Density")
axes[0].legend()

# Score drift
axes[1].hist(train_score, bins=20, alpha=0.6, label="Train", density=True)
axes[1].hist(prod_score, bins=20, alpha=0.6, label="Prod", density=True)
axes[1].set_title("Score Drift: Model Scores")
axes[1].set_xlabel("Score")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()

out_path = "/workspaces/devfest_bishkek_2025/docs/income_score_drift.png"
plt.savefig(out_path, bbox_inches="tight")
out_path
