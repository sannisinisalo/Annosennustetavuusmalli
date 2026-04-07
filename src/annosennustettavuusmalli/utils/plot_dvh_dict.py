import numpy as np

"""
Tekijä: Akseli Leino
Muokkaaja: Sanni Sinisalo
"""

def plot_dvhs(ax_to_plot, dvhs, linestyle='-'):
    """Plots DVH to given axis.

    Args: ax_to_plot (plt.ax

    Returns: None
    """
    # Määritellään värit elinten mukaan
    colors = {
        "PTV": "red",
        "Heart": "blue",
        "Ipsilateral lung": "green",
        "Contralateral lung": "orange",
        "Contralateral breast": "purple",
    }

    for organ, dvh in dvhs.items():
        color = colors.get(organ, 'black')  # default black if organ not in colors
        ax_to_plot.plot(np.arange(0, int(len(dvh)/10) + 0.1, 0.1), dvh, 
                        label=organ, color=color, linestyle=linestyle)

    ax_to_plot.set_xlabel('Dose [Gy]')
    ax_to_plot.set_ylabel('Volume [%]')
    ax_to_plot.set_xlim(0, 0.1*len(next(iter(dvhs.values()))))  # oletetaan kaikki dvh:t yhtä pitkiä
    ax_to_plot.set_ylim(0, 100)
