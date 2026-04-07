import matplotlib.pyplot as plt
import numpy as np

"""
Tekijä: Akseli Leino
"""

def plot_dvhs(ax_to_plot, dvhs):
    """Plots DVH to given axis.

    Args: ax_to_plot (plt.ax

    Returns: None
    """
    for i, (organ, dvh) in enumerate(dvhs.items()):
        ax_to_plot.plot(np.arange(0, int(len(dvh)/10) + 0.1, 0.1), dvh, label=organ)
        ax_to_plot.set_xlabel('Dose [Gy]')
        ax_to_plot.set_ylabel('Volume [%]')
        ax_to_plot.set_xlim(0, 0.1*len(dvh))
        ax_to_plot.set_ylim(0, 100)

        
    