import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

for ax in [ax1, ax2]:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e')

epochs = np.array([1])
ax1.plot(epochs, [1.5], 'go-', linewidth=2.5, markersize=8, label='Fine-tuned')
ax1.plot(epochs, [2.5], 'bo-', linewidth=2.5, markersize=8, label='Baseline')
ax1.set_title('Training Loss', color='#e6edf3', fontsize=13)
ax1.set_ylabel('Loss', color='#8b949e')
ax1.legend(facecolor='#21262d', labelcolor='#e6edf3')
ax1.grid(True, alpha=0.2)

ax2.plot(epochs, [0.68], 'go-', linewidth=2.5, markersize=8, label='Fine-tuned')
ax2.plot(epochs, [0.45], 'bo-', linewidth=2.5, markersize=8, label='Baseline')
ax2.set_title('Decision Accuracy', color='#e6edf3', fontsize=13)
ax2.set_ylabel('Accuracy', color='#8b949e')
ax2.legend(facecolor='#21262d', labelcolor='#e6edf3')
ax2.grid(True, alpha=0.2)

plt.suptitle('CDN Cache Optimizer: Fine-tuning Results', color='#e6edf3', fontsize=14)
plt.tight_layout()
plt.savefig('training_results_finetuned.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Chart saved!")