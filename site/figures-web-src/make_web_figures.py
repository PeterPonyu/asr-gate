#!/usr/bin/env python3
"""Web redraws of the science figures from stamped site/_data extracts.

Same frozen arrays and claims as print. New canvas, IBM Plex, SVG + PNG@2x.
Never overwrites print figure PDFs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

SITE = Path(__file__).resolve().parents[1]
DATA = SITE / "_data"
OUT = SITE / "figures-web"
FONTDIR = SITE / "fonts"

C = {
    "score": "#0072B2",
    "oracle": "#009E73",
    "random": "#D55E00",
    "accept": "#0072B2",
    "defer": "#D55E00",
    "target": "#333333",
    "warn": "#D55E00",
    "vacuous": "#8A6D1A",
    "clean": "#0072B2",
    "n25": "#56B4E9",
    "n15": "#E69F00",
    "n5": "#D55E00",
    "whisper": "#CC79A7",
    "ink": "#1A1A1A",
    "ground": "#F7F4EE",
    "box": "#F2F2F2",
    "line": "#404040",
}


def load(name: str):
    with (DATA / name).open() as fh:
        return json.load(fh)


def setup_fonts() -> None:
    regular = FONTDIR / "IBMPlexSans-Regular.ttf"
    medium = FONTDIR / "IBMPlexSans-Medium.ttf"
    family = "DejaVu Sans"
    if regular.exists():
        font_manager.fontManager.addfont(str(regular))
        family = font_manager.FontProperties(fname=str(regular)).get_name()
    if medium.exists():
        font_manager.fontManager.addfont(str(medium))
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": C["ink"],
            "text.color": C["ink"],
            "axes.labelcolor": C["ink"],
            "xtick.color": C["ink"],
            "ytick.color": C["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{stem}.svg"
    png = OUT / f"{stem}-2x.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=180)
    plt.close(fig)
    print("wrote", svg.name, png.name)


def fig0() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, text, fill=C["box"], edge=C["line"], lw=1.4):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12)

    ax.text(0.3, 13.2, "Offline", fontsize=16, fontweight="medium", ha="left")
    ax.text(0.3, 7.35, "Live", fontsize=16, fontweight="medium", ha="left")
    ax.plot([0.2, 21.6], [8.05, 8.05], color="#D9D3C7", lw=1.0)

    box(0.4, 10.2, 5.4, 2.2, "Speaker-disjoint\ncalibration split")
    box(6.4, 10.2, 6.4, 2.2, "LTT λ-grid\nEB p-values, Bonferroni\nfreeze λ★")
    vac = FancyBboxPatch(
        (13.6, 10.35),
        7.6,
        1.9,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor="#F4EED9",
        edgecolor=C["vacuous"],
        linewidth=1.4,
        linestyle=(0, (4, 3)),
    )
    ax.add_patch(vac)
    ax.text(
        17.4,
        11.3,
        "if no λ certifies:\nvacuous-at-target",
        ha="center",
        va="center",
        fontsize=12,
        color=C["vacuous"],
    )
    ax.annotate(
        "",
        xy=(13.55, 11.3),
        xytext=(12.85, 11.3),
        arrowprops=dict(arrowstyle="->", color=C["vacuous"], lw=1.2, linestyle="dashed"),
    )
    ax.annotate("", xy=(6.35, 11.3), xytext=(5.85, 11.3), arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.4))

    box(0.4, 4.6, 4.6, 2.2, "Utterance audio")
    box(5.5, 4.6, 5.2, 2.2, "Frozen ASR decode\nartifacts\n(never the weights)")
    box(11.2, 4.6, 4.6, 2.2, "Confidence score s")
    ax.annotate("", xy=(5.45, 5.7), xytext=(5.05, 5.7), arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.4))
    ax.annotate("", xy=(11.15, 5.7), xytext=(10.75, 5.7), arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.4))

    hinge = FancyBboxPatch(
        (16.4, 4.35),
        5.0,
        2.7,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor="#E8F1F8",
        edgecolor=C["ink"],
        linewidth=2.0,
    )
    ax.add_patch(hinge)
    ax.text(18.9, 5.7, "s ≥ λ★ ?", ha="center", va="center", fontsize=16)

    ax.annotate(
        "",
        xy=(16.35, 6.55),
        xytext=(12.8, 11.15),
        arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.3),
    )
    ax.annotate("", xy=(16.35, 5.7), xytext=(15.85, 5.7), arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.4))

    acc = FancyBboxPatch(
        (1.2, 0.55),
        9.2,
        2.5,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor="#E8F1F8",
        edgecolor=C["accept"],
        linewidth=2.0,
    )
    ax.add_patch(acc)
    ax.text(
        5.8,
        1.8,
        "Accept\nmacro-CER ≤ α with probability ≥ 1−δ",
        ha="center",
        va="center",
        fontsize=13,
        color=C["ink"],
    )
    de = FancyBboxPatch(
        (11.6, 0.55),
        9.2,
        2.5,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor="#F8EDE4",
        edgecolor=C["defer"],
        linewidth=2.0,
        linestyle=(0, (6, 3)),
    )
    ax.add_patch(de)
    ax.text(16.2, 1.8, "Defer\nhuman review", ha="center", va="center", fontsize=13)
    ax.annotate("", xy=(5.8, 3.1), xytext=(17.4, 4.3), arrowprops=dict(arrowstyle="->", color=C["accept"], lw=1.6))
    ax.annotate(
        "",
        xy=(16.2, 3.1),
        xytext=(18.9, 4.3),
        arrowprops=dict(arrowstyle="->", color=C["defer"], lw=1.6, linestyle="dashed"),
    )
    save(fig, "fig0_overview")


def fig1() -> None:
    rc = load("rc.json")["paraformer_clean"]
    fr = load("frontier.json")
    op = fr["operating_point"]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    cov = np.array(rc["coverage"]) * 100
    ax.plot(cov, np.array(rc["risk"]) * 100, color=C["score"], lw=2.4, label="log-posterior gate")
    ax.plot(
        cov,
        np.array(rc["oracle_risk"]) * 100,
        color=C["oracle"],
        lw=1.8,
        ls="--",
        label="oracle (sort by true CER)",
    )
    ax.axhline(rc["random_line"] * 100, color=C["random"], lw=1.6, ls=":", label="analytic random deferral")
    ax.axhline(2.0, color=C["target"], lw=1.1, alpha=0.65)
    ax.text(2.2, 2.18, "α = 2% target", fontsize=12, color=C["ink"])
    ax.plot(op["coverage"] * 100, op["accepted_macro_cer"] * 100, "o", color=C["ink"], ms=9, zorder=5)
    ax.annotate(
        "certified operating point\nreseed 0: 85.7% coverage,\n1.00% accepted-set macro-CER, α = 2%",
        (op["coverage"] * 100, op["accepted_macro_cer"] * 100),
        xytext=(42, 3.15),
        fontsize=11,
        ha="left",
        color=C["ink"],
        arrowprops=dict(arrowstyle="->", lw=0.9, color=C["ink"]),
    )
    ax.set_xlabel("coverage (% auto-accepted)")
    ax.set_ylabel("accepted-set macro-CER (%)")
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig1_teaser_rc")


def fig2() -> None:
    holm = load("holm.json")
    keyorder = [
        ("paraformer", "aishell_clean"),
        ("paraformer", "musan5db"),
        ("paraformer", "musan15db"),
        ("paraformer", "musan25db"),
        ("whisper", "aishell_clean"),
        ("whisper", "thchs30_crosscorpus"),
    ]
    short = {
        "aishell_clean": "clean",
        "musan5db": "+5 dB",
        "musan15db": "+15 dB",
        "musan25db": "+25 dB",
        "thchs30_crosscorpus": "THCHS-30",
    }
    bbname = {"paraformer": "Paraformer", "whisper": "Whisper"}
    by = {(r["backbone"], r["condition"], r["score"]): r for r in holm["rows"]}
    mat = np.zeros((len(keyorder), 2))
    ylabels = []
    for i, (bb, cond) in enumerate(keyorder):
        ylabels.append(f"{bbname[bb]} / {short[cond]}")
        for j, sc in enumerate(["s1", "s2"]):
            mat[i, j] = by[(bb, cond, sc)]["excess_aurc"]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=mat.max())
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["s1 log-posterior", "s2 weak-link"])
    ax.set_yticks(range(len(keyorder)))
    ax.set_yticklabels(ylabels)
    for i, (bb, cond) in enumerate(keyorder):
        for j, sc in enumerate(["s1", "s2"]):
            r = by[(bb, cond, sc)]
            ax.text(j, i, f"{r['excess_aurc']:.3f}*", ha="center", va="center", fontsize=13, color=C["ink"])
    ax.set_title("excess-AURC over random deferral")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=11)
    ax.text(
        0.0,
        -0.18,
        holm["p_holm_note"],
        transform=ax.transAxes,
        fontsize=11,
        ha="left",
        va="top",
        wrap=True,
    )
    save(fig, "fig2_holm_matrix")


def fig3() -> None:
    fr = load("frontier.json")
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    xs, heights, cers, ok = [], [], [], []
    for row in fr["alphas"]:
        xs.append(row["alpha"] * 100)
        heights.append(row["coverage"] * 100)
        cers.append(None if row["accepted_macro_cer"] is None else row["accepted_macro_cer"] * 100)
        ok.append(row["certified"])
    for x, h, cer, certified in zip(xs, heights, cers, ok):
        if certified:
            ax.bar(x, h, width=0.55, color=C["accept"], alpha=0.9, edgecolor=C["accept"])
            ax.plot(x, cer, "o", color=C["ink"], ms=8, zorder=5)
            ax.text(x, h + 3.2, f"CER {cer:.2f}%", ha="center", fontsize=12)
        else:
            ax.bar(
                x,
                0.01,
                width=0.55,
                facecolor="white",
                edgecolor=C["vacuous"],
                linewidth=1.8,
                linestyle="dashed",
                hatch="///",
            )
            ax.text(x, 18, "vacuous\nat target", ha="center", fontsize=12, color=C["vacuous"])
    ax.set_xlabel("certified target α (% macro-CER)")
    ax.set_ylabel("test coverage (% auto-accepted)")
    ax.set_ylim(0, 118)
    ax.set_xlim(0.2, 5.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.text(
        2.6,
        62,
        f"20-reseed attainment at certified α = 2%:\n"
        f"{fr['display']['violations']} violations within δ = 0.1",
        ha="center",
        va="center",
        fontsize=12,
    )
    save(fig, "fig3_certified_frontier")


def fig4() -> None:
    noise = load("noise.json")
    lab = [r["label"] for r in noise["rows"]]
    acccer = np.array([r["accepted_cer_mean"] * 100 for r in noise["rows"]])
    lo = np.array([r["accepted_cer_min"] * 100 for r in noise["rows"]])
    hi = np.array([r["accepted_cer_max"] * 100 for r in noise["rows"]])
    accfrac = np.array([r["coverage_mean"] * 100 for r in noise["rows"]])
    viol = [r["violations"] for r in noise["rows"]]
    x = np.arange(len(lab))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 9.0))
    yerr = [acccer - lo, hi - acccer]
    ax1.errorbar(x, acccer, yerr=yerr, fmt="o-", color=C["accept"], capsize=4, lw=1.8, ms=7)
    ax1.axhline(2.0, color=C["target"], ls="--", lw=1.1)
    ax1.text(0.05, 2.12, "α = 2% target", fontsize=12)
    for xi, vv in zip(x, viol):
        ax1.text(xi, 0.22, f"{vv}/20", ha="center", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lab)
    ax1.set_ylabel("accepted-set macro-CER (%)")
    ax1.set_xlabel("acoustic condition (ESC-50 additive)")
    ax1.set_ylim(0, 3.0)
    ax1.set_title("certificate holds under noise")
    ax1.text(-0.12, 1.04, "(a)", transform=ax1.transAxes, fontweight="medium", fontsize=14)

    ax2.plot(x, accfrac, "s-", color=C["oracle"], lw=1.8, ms=7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(lab)
    ax2.set_ylabel("coverage (% auto-accepted)")
    ax2.set_xlabel("acoustic condition (ESC-50 additive)")
    ax2.set_ylim(70, 100)
    ax2.set_title("coverage cost of noise")
    ax2.text(-0.12, 1.04, "(b)", transform=ax2.transAxes, fontweight="medium", fontsize=14)
    fig.tight_layout()
    save(fig, "fig4_noise_robustness")


def fig5() -> None:
    rc = load("rc.json")
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    series = [
        ("paraformer_clean", "Paraformer / Aishell-1 clean", C["clean"], "-"),
        ("whisper_clean", "Whisper / Aishell-1 clean", C["whisper"], "--"),
        ("whisper_thchs30", "Whisper / THCHS-30", C["n15"], "-."),
    ]
    for key, lab, col, ls in series:
        d = rc[key]
        ax.plot(np.array(d["coverage"]) * 100, np.array(d["risk"]) * 100, color=col, ls=ls, lw=2.0, label=lab)
    ax.axhline(2.0, color=C["target"], lw=1.2, ls=":")
    ax.text(48, 3.2, "α = 2% certifiable ceiling", fontsize=12)
    ax.set_xlabel("coverage (% auto-accepted)")
    ax.set_ylabel("accepted-set macro-CER (%)")
    ax.set_ylim(0, 35)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 1.0))
    ax.set_title("certifiable vs vacuous backbones")
    save(fig, "fig5_vacuity_rc")


def _crossover_xy(crossover, alphas):
    xs, ys, vac = [], [], []
    for a in alphas:
        key = f"{a:.3f}" if a < 0.1 else f"{a:.2f}"
        n = crossover.get(key, crossover.get(f"{a:g}"))
        xs.append(a * 100)
        if n is None:
            ys.append(np.nan)
            vac.append(True)
        else:
            ys.append(n)
            vac.append(False)
    return np.array(xs), np.array(ys, dtype=float), vac


def fig6() -> None:
    cal = load("calsize.json")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 9.2))
    man_alphas = [0.015, 0.019, 0.02, 0.03, 0.05]
    x, y, _ = _crossover_xy(cal["paraformer_aishell_crossover"], man_alphas)
    ax1.plot(x, y, "o-", color=C["clean"], lw=1.8, ms=7, label="Paraformer / Aishell-1")

    bel_alphas = [0.015, 0.02, 0.03, 0.05]
    bx, by, bvac = _crossover_xy(cal["belle_min_n_cal_90pct"], bel_alphas)
    ax1.plot(bx[~np.isnan(by)], by[~np.isnan(by)], "s-", color=C["n15"], lw=1.8, ms=7, label="Belle / Aishell-1 (≥90% of 20 reseeds)")
    pool = cal["belle_pool_size"]
    for xi, yi, vac in zip(bx, by, bvac):
        if vac:
            ax1.plot(xi, pool, "s", ms=8, mfc="white", mec=C["n15"], mew=1.4)
            ax1.text(xi, pool + 550, "never", ha="center", fontsize=12)

    ex, ey, _ = _crossover_xy(cal["english_wav2vec2_large_crossover"], [0.02, 0.03, 0.05])
    ax1.plot(ex[~np.isnan(ey)], ey[~np.isnan(ey)], "^-", color=C["whisper"], lw=1.8, ms=7, label="wav2vec 2.0 large / LibriSpeech")
    ax1.set_xlabel("certified target α (% macro-CER)")
    ax1.set_ylabel("min n_cal to certify")
    ax1.set_xlim(0.8, 5.6)
    ax1.set_ylim(0, 16800)
    ax1.legend(loc="upper right", frameon=False, fontsize=11)
    ax1.set_title("calibration budget vs target")
    ax1.text(-0.12, 1.04, "(a)", transform=ax1.transAxes, fontweight="medium", fontsize=14)

    for a, col, ls, lab in [
        (0.05, C["clean"], "-", "Belle α = 5%"),
        (0.03, C["n15"], "--", "Belle α = 3%"),
        (0.02, C["warn"], ":", "Belle α = 2%"),
    ]:
        rows = [c for c in cal["belle_power_cells"] if abs(c["alpha"] - a) < 1e-9]
        rows = sorted(rows, key=lambda c: c["n_cal"])
        ax2.plot([c["n_cal"] for c in rows], [c["cert_fraction"] * 100 for c in rows], ls, color=col, lw=1.8, label=lab)
    ax2.axhline(90, color=C["target"], ls="--", lw=1.1)
    ax2.text(8200, 93, "90% of reseeds", fontsize=12)
    ax2.set_xlabel("calibration size n_cal")
    ax2.set_ylabel("% of reseeds that certify")
    ax2.set_ylim(-4, 108)
    ax2.set_xlim(0, 15000)
    ax2.legend(loc="center left", frameon=False, fontsize=11)
    ax2.set_title("Belle reseed attainment")
    ax2.text(-0.12, 1.04, "(b)", transform=ax2.transAxes, fontweight="medium", fontsize=14)
    fig.tight_layout()
    save(fig, "fig6_calsize_sweep")


def fig7() -> None:
    land = load("landscape.json")
    bbs = ["paraformer", "belle", "zipformer"]
    corps = ["aishell", "thchs30", "magicdata"]
    by = {(c["backbone"], c["corpus"]): c for c in land["cells"]}
    mat = np.full((3, 3), np.nan)
    labels = [[""] * 3 for _ in range(3)]
    vac = np.zeros((3, 3), dtype=bool)
    for i, bb in enumerate(bbs):
        for j, co in enumerate(corps):
            cell = by[(bb, co)]
            if cell["vacuous"]:
                vac[i, j] = True
                labels[i][j] = "vac."
            else:
                mat[i, j] = cell["tightest_alpha"] * 100
                labels[i][j] = f"{cell['tightest_alpha']*100:.1f}%\n{cell['mean_acceptance']*100:.1f}% acc."
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="Blues_r", aspect="equal", vmin=1.5, vmax=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Aishell-1", "THCHS-30", "MagicData"])
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Paraformer", "Belle", "zipformer"])
    for i in range(3):
        for j in range(3):
            if vac[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#D9D9D9", edgecolor="none"))
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=14, color=C["ink"])
            else:
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=12, color=C["ink"])
    ax.set_title("tightest certified α (0/20 violations)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("α (%)")
    save(fig, "fig7_landscape")


def main() -> None:
    setup_fonts()
    fig0()
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    print("web figures written under", OUT)


if __name__ == "__main__":
    main()
