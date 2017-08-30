#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from distutils.version import LooseVersion

def set_defaults():
    """
    Set default plotting behavior.
    """

    start_colors = ["#D11700", "#8EBA42", "#348ABD", "#988ED5", "#777777",
                    "#FBC15E", "#FFB5B8"]
    if LooseVersion(mpl.__version__) < LooseVersion("1.5.1"):
        plt.rcParams["axes.color_cycle"] = start_colors
    else:
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", start_colors)

    plt.rcParams["axes.facecolor"]  = "w"
    plt.rcParams["axes.grid"]       = True
    plt.rcParams["axes.axisbelow"]  = True
    plt.rcParams["axes.linewidth"]  = 0
    plt.rcParams["axes.labelcolor"] = "k"

    plt.rcParams["figure.facecolor"]      = "w"
    plt.rcParams["figure.subplot.bottom"] = 0.125
    plt.rcParams["figure.subplot.left"]   = 0.1

    plt.rcParams["lines.linewidth"] = 2

    plt.rcParams["grid.color"]     = "#444444"
    plt.rcParams["grid.linewidth"] = 1.5
    plt.rcParams["grid.linestyle"] = ":"

    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.color"]     = "k"
    plt.rcParams["ytick.color"]     = "k"

    plt.rcParams["text.color"] = "k"

    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["image.cmap"] = "gist_gray"

    return


# -- utility for plotting
def plot_rgb(hcube, lam=[610., 540., 475.], scl=2.5):
    """
    Plot a false color version of the data cube.
    """
    ind = [np.argmin(np.abs(hcube.waves - clr)) for clr in lam]
    rgb = hcube.data[ind].copy()
    wgt = rgb.mean(-1).mean(-1)
    scl = scl * wgt[0] / wgt * 2.**8 / 2.**12
    rgb = (rgb * scl).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)

    fig, ax = plt.subplots(figsize=[6.5, 6.5/2])
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)
    ax.axis("off")
    im = ax.imshow(rgb,aspect=0.45)
    fig.canvas.draw()

    return

