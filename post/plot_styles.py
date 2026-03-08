#!/usr/bin/env python3

import matplotlib as mpl
mpl.rcParams.update({
    'text.usetex': True,
    # Font family must be serif for lmodern
    'font.family': 'serif',

    # Base font sizes (LaTeX-controlled)
    'font.size': 11,          # match your document's \normalsize
    'axes.labelsize': 11,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,

    # LaTeX preamble — MATCHES YOUR DOCUMENT
    'text.latex.preamble': r'''
        \usepackage[T1]{fontenc}
        \usepackage{lmodern}
        \usepackage{textcomp}
        \usepackage{microtype}
        \usepackage{bm}
    '''
    })