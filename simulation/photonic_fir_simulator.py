"""
Photonic FIR Filter Simulator
==============================
Simulates an ideal photonic FIR filter with optional tap-dependent loss.
Frequency axis is normalised: f_hat = f * T in [-0.5, 0.5].

Usage:
    python photonic_fir.py
    python photonic_fir.py --taps "1" "0.5+0.5j" "-1+0j" "0.5-0.5j" "0.25"
    python photonic_fir.py --loss 1.5 --loss-model propagation
    python photonic_fir.py --taps "1" "1" "1" "1" "1" --loss 2.0 --loss-model coupler --npoints 4096
"""

import argparse
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---------------------------------------------------------------------------
# Complex coefficient parser
# ---------------------------------------------------------------------------


def parse_complex(s: str) -> complex:
    """Parse a string like '0.5+0.3j', '-1j', '0.5' into a Python complex."""
    s = s.strip().replace(" ", "")
    try:
        return complex(s)
    except ValueError:
        pass
    s = re.sub(r"(?<=[0-9])\+j$", "+1j", s)
    s = re.sub(r"(?<=[0-9])-j$", "-1j", s)
    s = re.sub(r"^\+j$", "1j", s)
    s = re.sub(r"^-j$", "-1j", s)
    try:
        return complex(s)
    except ValueError:
        raise ValueError(f"Cannot parse '{s}' as a complex number.")


# ---------------------------------------------------------------------------
# Loss model
# ---------------------------------------------------------------------------


def apply_loss(
    taps: np.ndarray, loss_db: float, model: str = "propagation"
) -> np.ndarray:
    """
    Apply tap-dependent loss to the filter coefficients.

    Parameters
    ----------
    taps     : complex ndarray, shape (N,)
    loss_db  : loss magnitude in dB (>=0). Interpretation depends on model.
    model    : one of
               'propagation' — waveguide loss accumulates with path length:
                               scale_k = exp(-alpha * k)
               'uniform'     — fixed insertion loss, same on every tap:
                               scale_k = exp(-alpha)  for all k
               'coupler'     — excess loss at each beamsplitter stage:
                               scale_k = exp(-alpha * k / 2)

    Returns
    -------
    lossy_taps : complex ndarray, shape (N,)
    """
    if loss_db < 0:
        raise ValueError("loss_db must be >= 0.")
    alpha = loss_db / 8.6859  # dB → nepers (1 Np = 20/ln(10) dB ≈ 8.686 dB)
    k = np.arange(len(taps), dtype=float)

    if model == "propagation":
        envelope = np.exp(-alpha * k)
    elif model == "uniform":
        envelope = np.full(len(taps), np.exp(-alpha))
    elif model == "coupler":
        envelope = np.exp(-alpha * k / 2)
    else:
        raise ValueError(
            f"Unknown loss model '{model}'. "
            "Choose from: propagation, uniform, coupler."
        )
    return taps * envelope


# ---------------------------------------------------------------------------
# Frequency response
# ---------------------------------------------------------------------------


def freq_response(taps: np.ndarray, n_points: int = 2048):
    """
    Compute the frequency response of a photonic FIR filter.

    Transfer function:
        H(f_hat) = sum_k h_k * exp(-j * 2*pi * f_hat * k)

    Parameters
    ----------
    taps     : complex ndarray, shape (N,)
    n_points : number of frequency points over [-0.5, 0.5)

    Returns
    -------
    f_hat : ndarray  — normalised frequency axis
    H     : ndarray  — complex transfer function
    """
    f_hat = np.linspace(-0.5, 0.5, n_points, endpoint=False)
    k = np.arange(len(taps))
    phase = np.exp(-1j * 2 * np.pi * np.outer(f_hat, k))  # (n_points, N)
    H = phase @ taps  # (n_points,)
    return f_hat, H


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_fir(
    taps: np.ndarray,
    lossy_taps: np.ndarray,
    loss_db: float,
    loss_model: str,
    n_points: int = 2048,
):
    """
    Plot tap weights and frequency response for lossless and lossy cases.

    Layout
    ------
    Row 0  : tap magnitudes | tap phases
    Row 1  : magnitude response, linear (lossless vs lossy overlay)
    Row 2  : magnitude response, dB     | phase response
    """
    N = len(taps)
    tap_idx = np.arange(N)

    # --- Tap quantities ---
    lossless_mags = np.abs(taps)
    lossy_mags = np.abs(lossy_taps)
    lossy_phases = np.angle(lossy_taps, deg=True)

    # --- Frequency responses ---
    f_hat, H = freq_response(taps, n_points)
    _, HL = freq_response(lossy_taps, n_points)

    H_mag = np.abs(H)
    HL_mag = np.abs(HL)
    H_mag_dB = 20 * np.log10(H_mag + 1e-12)
    HL_mag_dB = 20 * np.log10(HL_mag + 1e-12)
    HL_phase = np.angle(HL, deg=True)

    has_loss = loss_db > 0

    # --- Figure layout ---
    fig = plt.figure(figsize=(12, 9))
    title = f"Photonic FIR — {N}-tap filter"
    if has_loss:
        title += f"  |  loss: {loss_db:.1f} dB/tap ({loss_model})"
    fig.suptitle(title, fontsize=13, fontweight="normal", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # --- Row 0 left: tap magnitudes ---
    ax0 = fig.add_subplot(gs[0, 0])
    width = 0.35
    ax0.bar(
        tap_idx - width / 2,
        lossless_mags,
        width=width,
        color="#378ADD",
        alpha=0.85,
        label="Lossless",
    )
    if has_loss:
        ax0.bar(
            tap_idx + width / 2,
            lossy_mags,
            width=width,
            color="#D85A30",
            alpha=0.75,
            label="Lossy",
        )
        ax0.legend(fontsize=8)
    ax0.set_xlabel("Tap index $k$", fontsize=10)
    ax0.set_ylabel(r"$|h_k|$", fontsize=10)
    ax0.set_title("Tap magnitudes", fontsize=10)
    ax0.set_xticks(tap_idx)
    ax0.grid(axis="y", linewidth=0.4, alpha=0.5)

    # --- Row 0 right: tap phases (lossy, since loss is real-valued) ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(tap_idx, lossy_phases, color="#EF9F27", alpha=0.8, width=0.6)
    ax1.set_xlabel("Tap index $k$", fontsize=10)
    ax1.set_ylabel(r"$\angle h_k$ (deg)", fontsize=10)
    ax1.set_title("Tap phases" + (" (after loss)" if has_loss else ""), fontsize=10)
    ax1.set_xticks(tap_idx)
    ax1.axhline(0, color="grey", linewidth=0.5)
    ax1.grid(axis="y", linewidth=0.4, alpha=0.5)

    # --- Row 1: magnitude response, linear ---
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(f_hat, H_mag, color="#1D9E75", linewidth=1.4, label="Lossless")
    if has_loss:
        ax2.plot(
            f_hat,
            HL_mag,
            color="#D85A30",
            linewidth=1.2,
            linestyle="--",
            label=f"Lossy ({loss_db} dB/tap, {loss_model})",
        )
        ax2.legend(fontsize=9)
    ax2.set_xlabel(r"Normalised frequency $\hat{f} = fT$", fontsize=10)
    ax2.set_ylabel(r"$|H(\hat{f})|$", fontsize=10)
    ax2.set_title("Magnitude response (linear)", fontsize=10)
    ax2.set_xlim(-0.5, 0.5)
    ax2.grid(linewidth=0.4, alpha=0.5)
    ax2.axhline(0, color="grey", linewidth=0.5)

    # --- Row 2 left: magnitude response, dB ---
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(f_hat, H_mag_dB, color="#534AB7", linewidth=1.4, label="Lossless")
    if has_loss:
        ax3.plot(
            f_hat,
            HL_mag_dB,
            color="#D85A30",
            linewidth=1.2,
            linestyle="--",
            label="Lossy",
        )
        ax3.legend(fontsize=9)
    ax3.set_xlabel(r"Normalised frequency $\hat{f} = fT$", fontsize=10)
    ax3.set_ylabel(r"$|H(\hat{f})|$ (dB)", fontsize=10)
    ax3.set_title("Magnitude response (dB)", fontsize=10)
    ax3.set_xlim(-0.5, 0.5)
    ax3.grid(linewidth=0.4, alpha=0.5)

    # --- Row 2 right: phase response ---
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(f_hat, HL_phase, color="#D85A30", linewidth=1.2)
    ax4.set_xlabel(r"Normalised frequency $\hat{f} = fT$", fontsize=10)
    ax4.set_ylabel(r"$\angle H(\hat{f})$ (deg)", fontsize=10)
    ax4.set_title("Phase response" + (" (lossy)" if has_loss else ""), fontsize=10)
    ax4.set_xlim(-0.5, 0.5)
    ax4.axhline(0, color="grey", linewidth=0.5)
    ax4.grid(linewidth=0.4, alpha=0.5)

    plt.savefig("photonic_fir.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to photonic_fir.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_TAPS = ["1", "0.5+0.5j", "-1+0j", "0.5-0.5j", "0.25"]


def main():
    parser = argparse.ArgumentParser(
        description="Photonic FIR filter simulator — ideal, with optional loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--taps",
        nargs="+",
        default=DEFAULT_TAPS,
        metavar="COEFF",
        help="Complex tap coefficients, e.g. --taps 1 '0.5+0.5j' '-1+0j'",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=2048,
        help="Number of frequency points",
    )
    parser.add_argument(
        "--loss",
        type=float,
        default=0.0,
        metavar="DB",
        help="Loss per tap in dB (0 = lossless)",
    )
    parser.add_argument(
        "--loss-model",
        default="propagation",
        choices=["propagation", "uniform", "coupler"],
        help="Loss model: propagation | uniform | coupler",
    )
    args = parser.parse_args()

    # Parse tap coefficients
    try:
        taps = np.array([parse_complex(s) for s in args.taps], dtype=complex)
    except ValueError as e:
        print(f"Error parsing taps: {e}", file=sys.stderr)
        sys.exit(1)

    # Apply loss
    try:
        lossy_taps = apply_loss(taps, args.loss, args.loss_model)
    except ValueError as e:
        print(f"Error applying loss: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print(
        f"\nFilter: {len(taps)} taps  |  loss: {args.loss:.1f} dB/tap ({args.loss_model})"
    )
    print(
        f"{'k':>4}  {'Re(h_k)':>10}  {'Im(h_k)':>10}  "
        f"{'|h_k| (lossless)':>18}  {'|h_k| (lossy)':>14}  {'arg (deg)':>10}"
    )
    for k, (h, hl) in enumerate(zip(taps, lossy_taps)):
        print(
            f"{k:>4}  {h.real:>+10.4f}  {h.imag:>+10.4f}  "
            f"{abs(h):>18.4f}  {abs(hl):>14.4f}  "
            f"{np.angle(hl, deg=True):>+10.2f}"
        )

    f_hat, H = freq_response(taps, args.npoints)
    _, HL = freq_response(lossy_taps, args.npoints)
    print(
        f"\nLossless  — peak |H|: {np.max(np.abs(H)):.4f} "
        f"at f_hat = {f_hat[np.argmax(np.abs(H))]:+.4f}"
    )
    if args.loss > 0:
        print(
            f"Lossy     — peak |H|: {np.max(np.abs(HL)):.4f} "
            f"at f_hat = {f_hat[np.argmax(np.abs(HL))]:+.4f}"
        )

    plot_fir(taps, lossy_taps, args.loss, args.loss_model, args.npoints)


if __name__ == "__main__":
    main()
