import matplotlib.pyplot as plt
import numpy as np

def plot_histograms_styled(orig, wm):
    """
    Plot side-by-side histograms of original and watermarked images.
    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    ax[0].hist(orig.flatten(), bins=256, range=(0, 255), color='dodgerblue', alpha=0.8)
    ax[0].set_title('Original Histogram', fontsize=10)
    ax[1].hist(wm.flatten(), bins=256, range=(0, 255), color='orangered', alpha=0.8)
    ax[1].set_title('Watermarked Histogram', fontsize=10)
    plt.tight_layout()
    return fig

def plot_difference_styled(orig, wm):
    """
    Plot absolute difference heatmap between original and watermarked images.
    """
    diff = np.abs(orig.astype(int) - wm.astype(int))
    fig, ax = plt.subplots(figsize=(3.5, 3))
    im = ax.imshow(diff, cmap='viridis')
    fig.colorbar(im, ax=ax)
    ax.set_title('Absolute Difference', fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_comparative_lines(x, y1, y2, y3=None, metric_name="Metric", method1_name="PEE", method2_name="HS", method3_name="ML-Assisted", param_name="Parameter"):
    """
    Plot comparative line graphs for up to three methods.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(x, y1, label=method1_name, color='deepskyblue', marker='o', linestyle='-')
    ax.plot(x, y2, label=method2_name, color='salmon', marker='x', linestyle='--')
    if y3 is not None:
        ax.plot(x, y3, label=method3_name, color='mediumseagreen', marker='^', linestyle=':')
    ax.set_xlabel(param_name, fontsize=9)
    ax.set_ylabel(metric_name, fontsize=9)
    ax.set_title(f'{metric_name} vs. {param_name}', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    return fig

def plot_comparative_bar(methods_list, values_list, metric_name="Metric", title_extra=""):
    """
    Plot a comparative bar chart for multiple methods.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = ['deepskyblue', 'salmon', 'mediumseagreen']  # Supports up to 3 methods
    bars = ax.bar(methods_list, values_list, color=colors[:len(methods_list)], width=0.5, alpha=0.8)
    ax.set_ylabel(metric_name, fontsize=9)
    ax.set_title(f'{metric_name} Comparison {title_extra}', fontsize=11)
    ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
    ax.set_ylim(bottom=0, top=max(values_list) * 1.15 if values_list else 10)
    ax.grid(True, linestyle=':', axis='y', alpha=0.7)
    plt.tight_layout()
    return fig
