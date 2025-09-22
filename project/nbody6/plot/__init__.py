import os

import matplotlib.pyplot as plt

style_path = os.path.join(os.path.dirname(__file__), "style.mplstyle")
plt.style.use(style_path)

PLOT_STYLE_DICT = {
    "single": {"marker": "o", "color": "tab:green"},
    "resolved": {"marker": "^", "color": "orange"},
    "unresolved": {"marker": "s", "color": "tab:red"},
}
