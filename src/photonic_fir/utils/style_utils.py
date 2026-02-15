from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_calibration_style(dark: bool = False) -> None:
    """Register and apply the calibration matplotlib stylesheet.

    Call this once before any plotting. Safe to call multiple times
    (duplicate paths are ignored).

    Parameters
    ----------
    dark : bool
        If True, apply the dark mode overlay on top of the base style.
    """
    styles_dir = Path(__file__).parent
    if str(styles_dir) not in mpl.style.core.USER_LIBRARY_PATHS:
        mpl.style.core.USER_LIBRARY_PATHS.append(str(styles_dir))
        mpl.style.core.reload_library()

    styles = ["calibration", "calibration_dark"] if dark else ["calibration"]
    plt.style.use(styles)
