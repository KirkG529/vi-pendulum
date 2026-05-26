import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Candidate file paths for the 2x2 montage (expected output from experiments)
files = [
    "./results/experiments/mild/cost_min_time_base.png",
    "./results/experiments/mild/policy_min_time_base.png",
    "./results/experiments/mild/cost_quad_base.png",
    "./results/experiments/mild/policy_quad_base.png",
]

# only keep files that actually exist; fall back to alternative names if needed
files = [f for f in files if os.path.exists(f)]
if len(files) < 4:
    # try alternative/more generic names used by earlier scripts
    alt = [
        "./results/experiments/mild/cost_min_time.png",
        "./results/experiments/mild/policy_min_time.png",
        "./results/experiments/mild/cost_quadratic.png",
        "./results/experiments/mild/policy_quadratic.png",
    ]
    for f in alt:
        if os.path.exists(f):
            files.append(f)
files = files[:4]

# load images and create a 2x2 figure with titles and axis labels
imgs = [mpimg.imread(f) for f in files]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
titles = [
    "(a) Min-Time Cost-to-Go",
    "(b) Min-Time Policy",
    "(c) Quadratic Cost-to-Go",
    "(d) Quadratic Policy",
]
for ax, img, title in zip(axes.flatten(), imgs, titles):
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    # label axes consistently so readers can map pixels -> (q, qdot)
    ax.set_xlabel("q", fontsize=10)
    ax.set_ylabel("qdot", fontsize=10)
    ax.tick_params(labelsize=8)

plt.tight_layout()
os.makedirs("./assets/figures", exist_ok=True)
out_path = './assets/figures/combined_2x2.png'
plt.savefig(out_path, dpi=150)
print(f'Saved {out_path}')
