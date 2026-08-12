#!/usr/bin/env python3
"""Regenerate F1-F7 for the TASLP manuscript from frozen result JSONs.

F1-F5 and the landscape heatmap read ../results/numbers.json. The calibration-size
panels additionally read the frozen calsweep / power-curve result files next to
the science home (no new decodes). No hardcoded science arrays.

Run from the manuscript dir:  python3 figures/make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(HERE, "..", "results")
NUM = json.load(open(os.path.join(RES, "numbers.json")))


def _load_frozen(rel):
    path = os.path.join(REPO, rel)
    with open(path) as fh:
        return json.load(fh)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight",
    # Times-compatible serif matching the manuscript body text; STIX for math;
    # embed as TrueType (fonttype 42) so no Type-3 glyphs leak into the PDF/PS.
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
C = {"score": "#0072B2", "oracle": "#009E73", "random": "#D55E00",
     "accept": "#0072B2", "target": "#333333", "warn": "#D55E00",
     "clean": "#0072B2", "n25": "#56B4E9", "n15": "#E69F00", "n5": "#D55E00",
     "whisper": "#CC79A7"}


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ F1 teaser RC
# FIGCONTENT-2026-07-16 KEEP-JUSTIFIED (single series): risk_coverage carries 6
#   series (paraformer_clean + 3 musan SNRs + whisper_clean + whisper_thchs30),
#   but this is the paper's introductory teaser -- its job is to explain the
#   risk-coverage mechanism (gate vs. oracle vs. random deferral) and mark one
#   operating point, before any backbone/noise/cross-corpus comparison has been
#   introduced. Overlaying more series here would duplicate fig4 (noise arm)
#   and fig5 (backbone vacuity) and dilute the one-curve pedagogical read.
# Every other risk_coverage series is plotted elsewhere: musan5/15/25db feed
#   fig4's noise-condition x-axis; whisper_clean/whisper_thchs30 feed fig5's
#   backbone-vacuity comparison (paraformer_clean recurs there too).
def fig1():
    rc = NUM["risk_coverage"]["paraformer_clean"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    cov = np.array(rc["coverage"]) * 100
    ax.plot(cov, np.array(rc["risk"]) * 100, color=C["score"], lw=2,
            label=r"$s_1$ log-posterior gate")
    ax.plot(cov, np.array(rc["oracle_risk"]) * 100, color=C["oracle"], lw=1.3,
            ls="--", label="oracle (sort by true CER)")
    ax.axhline(rc["random_line"] * 100, color=C["random"], lw=1.3, ls=":",
               label="analytic random deferral")
    ax.axhline(2.0, color=C["target"], lw=1.0, alpha=0.6)
    ax.text(3, 2.15, r"$\alpha=2\%$ target", fontsize=7.5, color="black")
    # operating point: main run reseed-0 (85.7% cov, 1.00% accCER)
    op = NUM["main_alpha_frontier"]["0.02"]
    ax.plot(op["test_accepted_fraction"] * 100, op["test_accepted_macro_cer"] * 100,
            "o", color="#333", ms=6, zorder=5)
    # single panel: no in-figure title; operating-point detail lives in the caption.
    # REVIEWFIX-D3-2026-07-16: short annotation placed adjacent to the marked point
    #   (was xytext=(26,3.2), a long arrow spanning the panel that ran under the
    #   legend's third line); black text per the annotation-color rail.
    ax.annotate("certified\noperating point",
                (op["test_accepted_fraction"] * 100, op["test_accepted_macro_cer"] * 100),
                xytext=(55, 2.35), fontsize=7.5, ha="left", color="black",
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("coverage (% auto-accepted)")
    ax.set_ylabel("accepted-set macro-CER (%)")
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "fig1_teaser_rc.pdf")


# ------------------------------------------------------------------ F2 Holm m=12
def fig2():
    rows = NUM["holm_realized"]["rows"]
    conds, order = [], []
    # TABLEFIX-2026-07-16: shortened condition labels (were "Aishell clean" /
    #   "Aishell +noise 5 dB" / ... / "THCHS-30 (cross-corpus)") -- the repeated "Aishell"
    #   and long SNR phrasing pushed the tight bbox well past the other figures' native
    #   width, so at a matched includegraphics width the cell/tick text fell under the
    #   7pt-effective floor. Backbone name now carries the corpus context implicitly
    #   (Aishell is the default; THCHS-30 is called out).
    bbname = {"paraformer": "Paraformer", "whisper": "Whisper"}
    label = {"aishell_clean": "clean", "musan5db": "+5 dB", "musan15db": "+15 dB",
             "musan25db": "+25 dB", "thchs30_crosscorpus": "THCHS-30"}
    keyorder = [("paraformer", "aishell_clean"), ("paraformer", "musan5db"),
                ("paraformer", "musan15db"), ("paraformer", "musan25db"),
                ("whisper", "aishell_clean"), ("whisper", "thchs30_crosscorpus")]
    mat = np.zeros((len(keyorder), 2))
    rej = np.zeros((len(keyorder), 2), dtype=bool)
    ylabels = []
    for i, (bb, cond) in enumerate(keyorder):
        ylabels.append(f"{bbname[bb]} / {label[cond]}")
        for j, sc in enumerate(["s1", "s2"]):
            r = next(x for x in rows if x["backbone"] == bb and x["condition"] == cond and x["score"] == sc)
            mat[i, j] = r["excess_aurc"]
            rej[i, j] = r["reject_holm_global"]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=max(float(mat.max()), 0.055))
    ax.set_xticks([0, 1]); ax.set_xticklabels([r"$s_1$ log-post.", r"$s_2$ weak-link"])
    ax.set_yticks(range(len(keyorder))); ax.set_yticklabels(ylabels, fontsize=8)
    for i in range(len(keyorder)):
        for j in range(2):
            star = r"$^{*}$" if rej[i, j] else ""
            ax.text(j, i, f"{mat[i,j]:.3f}{star}", ha="center", va="center",
                    fontsize=8, color="black")
    # single panel: short centered title; Holm m and p_Holm detail moved to caption
    ax.set_title(r"excess-AURC over random deferral", fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7.5)
    save(fig, "fig2_holm_matrix.pdf")


# ------------------------------------------------------------------ F3 frontier
def fig3():
    fr = NUM["main_alpha_frontier"]
    alphas = [0.01, 0.02, 0.03, 0.05]
    cov = [fr[f"{a}"]["test_accepted_fraction"] * 100 for a in alphas]
    cer = [(fr[f"{a}"]["test_accepted_macro_cer"] or 0) * 100 for a in alphas]
    cert = [fr[f"{a}"]["certified"] for a in alphas]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    x = [a * 100 for a in alphas]
    # REVIEWFIX-D3-2026-07-16: widen x-limits so the vacuous label clears the axis
    #   spine (was rendering over the y-axis at the far-left tick).
    ax.set_xlim(0.2, 5.8)
    for xi, ci, ce, ok in zip(x, cov, cer, cert):
        col = C["accept"] if ok else C["warn"]
        ax.bar(xi, ci, width=0.35, color=col, alpha=0.85 if ok else 0.4)
        if ok:
            ax.text(xi, ci + 2, f"CER\n{ce:.2f}%", ha="center", fontsize=7.5)
        else:
            # REVIEWFIX-D3: black text (was warn/orange) lifted off the bottom axis
            #   into the empty vacuous column (was y=4, overlapping the x/y axes).
            ax.text(xi, 16, "vacuous\nat target", ha="center", fontsize=7.5, color="black")
    ax.set_xlabel(r"certified target $\alpha$ (% macro-CER)")
    ax.set_ylabel("test coverage (% auto-accepted)")
    ax.set_ylim(0, 108)
    ax.set_xticks(x); ax.set_xticklabels([f"{a}" for a in x])
    # REVIEWFIX-D3-2026-07-16: the reseed-attainment fact was a floating inset at
    #   [0.53,0.35,...] that sat on top of the tall alpha=3%/5% coverage bars; the
    #   bar it drew was 0/20 (zero height), so it conveyed only "below the delta
    #   line". Restyled to a compact black text note in the empty vacuous column --
    #   no overlap, same information (all text black per the annotation-color rail).
    at = NUM["main_attainment"]
    nviol = at["n_violations"]; nres = at["n_reseeds"]
    ax.text(1.35, 62,
            f"20-reseed attainment\n(certified $\\alpha=2\\%$):\n"
            f"{nviol}/{nres} violations\nwithin $\\delta=0.1$",
            ha="center", va="center", fontsize=7, color="black")
    save(fig, "fig3_certified_frontier.pdf")


# ------------------------------------------------------------------ F4 noise arm
def fig4():
    ea = NUM["expansion_attainment"]
    order = [("clean", None), ("25 dB", "musan25db"), ("15 dB", "musan15db"), ("5 dB", "musan5db")]
    lab, acccer, accfrac, viol, lo, hi = [], [], [], [], [], []
    # clean = main run
    at = NUM["main_attainment"]
    lab.append("clean")
    acccer.append(at["accepted_cer_mean"] * 100)
    lo.append(at["accepted_cer_min"] * 100); hi.append(at["accepted_cer_max"] * 100)
    accfrac.append(at["accepted_fraction_mean"] * 100)
    viol.append(at["n_violations"])
    for l, k in order[1:]:
        v = ea[k]
        lab.append(l)
        acccer.append(v["acc_set_macro_cer_mean"] * 100)
        lo.append(v["acc_set_macro_cer_min"] * 100); hi.append(v["acc_set_macro_cer_max"] * 100)
        accfrac.append(v["acc_fraction_mean"] * 100)
        viol.append(v["violations"])
    x = np.arange(len(lab))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    yerr = [np.array(acccer) - np.array(lo), np.array(hi) - np.array(acccer)]
    ax1.errorbar(x, acccer, yerr=yerr, fmt="o-", color=C["accept"], capsize=3, lw=1.5)
    ax1.axhline(2.0, color=C["target"], ls="--", lw=1)
    ax1.text(0.05, 2.08, r"$\alpha=2\%$ target", fontsize=7.5, color="black")
    # REVIEWFIX-D3-2026-07-16: violation counts in black (were warn/orange when
    #   nonzero) per the annotation-color rail -- the 5 dB 1/20 is now black too.
    for xi, vv in zip(x, viol):
        ax1.text(xi, 0.18, f"{vv}/20", ha="center", fontsize=7.5, color="black")
    ax1.text(x[-1] + 0.02, 0.42, "viol./20", ha="right", fontsize=7.5, color="black")
    ax1.set_xticks(x); ax1.set_xticklabels(lab)
    ax1.set_ylabel("accepted-set macro-CER (%)")
    ax1.set_xlabel("acoustic condition (ESC-50 additive)")
    ax1.set_ylim(0, 3.0)
    # short centered per-panel titles; "clean-calibrated gate" detail moved to caption
    ax1.set_title("certificate holds under noise", fontsize=8)
    ax2.plot(x, accfrac, "s-", color=C["oracle"], lw=1.5)
    ax2.set_xticks(x); ax2.set_xticklabels(lab)
    ax2.set_ylabel("coverage (% auto-accepted)")
    ax2.set_xlabel("acoustic condition (ESC-50 additive)")
    ax2.set_ylim(70, 100)
    ax2.set_title("coverage cost of noise", fontsize=8)
    # bold standalone panel labels, top-left of each panel
    ax1.text(-0.16, 1.04, "(a)", transform=ax1.transAxes, fontweight="bold",
             fontsize=10, va="bottom", ha="left")
    ax2.text(-0.16, 1.04, "(b)", transform=ax2.transAxes, fontweight="bold",
             fontsize=10, va="bottom", ha="left")
    save(fig, "fig4_noise_robustness.pdf")


# ------------------------------------------------------------------ F5 vacuity
# FIGCONTENT-2026-07-16 KEEP-JUSTIFIED (3 of 6 risk_coverage series): this panel's
#   claim is specifically backbone-driven vacuity at clean condition (certifiable
#   Paraformer vs. vacuous Whisper on two Mandarin corpora); the 3 omitted series
#   are Paraformer's musan5/15/25db noise arm, a different axis (noise level, not
#   backbone) already covered by fig4. Adding them here would overlay two unrelated
#   claims (noise robustness and backbone vacuity) on one axis.
def fig5():
    rc = NUM["risk_coverage"]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    series = [("paraformer_clean", "Paraformer / Aishell clean", C["clean"], "-"),
              ("whisper_clean", "Whisper / Aishell clean", C["whisper"], "--"),
              ("whisper_thchs30", "Whisper / THCHS-30", C["n15"], "-.")]
    for key, lab, col, ls in series:
        d = rc[key]
        ax.plot(np.array(d["coverage"]) * 100, np.array(d["risk"]) * 100,
                color=col, ls=ls, lw=1.8, label=lab)
    ax.axhline(2.0, color=C["target"], lw=1.0, ls=":")
    ax.text(52, 2.4, r"$\alpha=2\%$ certifiable ceiling", fontsize=7.5, color="black")
    ax.set_xlabel("coverage (% auto-accepted)")
    ax.set_ylabel("accepted-set macro-CER (%)")
    ax.set_ylim(0, 35)
    ax.set_xlim(0, 100)
    ax.legend(loc="center right", frameon=False, fontsize=7.5)
    # single panel: short centered title; full explanation moved to caption
    ax.set_title("certifiable vs vacuous backbones", fontsize=8)
    save(fig, "fig5_vacuity_rc.pdf")


# ------------------------------------------------------------------ F6 calibration budget
# Frozen records only: mandarin_calsweep (Paraformer seed-0 crossover),
# english_calsweep (wav2vec2-large crossover), calsize_power (Belle 20-reseed
# cert fraction). Same science as the calibration-size prose; no new decodes.
def fig6():
    man = _load_frozen("mandarin_calsweep_2026-07-13/results.json")
    eng = _load_frozen("english_calsweep_2026-07-13/results.json")
    bel = _load_frozen("calsize_power_2026-07-15/results.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.65))

    def _xy(crossover, alphas):
        xs, ys, vac = [], [], []
        for a in alphas:
            n = crossover.get(f"{a:g}", crossover.get(f"{a:.3f}"))
            # JSON keys are strings like "0.015"
            if n is None:
                key = f"{a:.3f}" if a < 0.1 else f"{a:.2f}"
                n = crossover.get(key)
            xs.append(a * 100)
            if n is None:
                ys.append(np.nan)
                vac.append(True)
            else:
                ys.append(n)
                vac.append(False)
        return np.array(xs), np.array(ys, dtype=float), vac

    man_alphas = [0.015, 0.019, 0.02, 0.03, 0.05]
    x, y, _ = _xy(man["crossover_seed0"], man_alphas)
    ax1.plot(x, y, "o-", color=C["clean"], lw=1.5, ms=5, label="Paraformer / Aishell-1")

    bel_head = bel["headline_min_n_cal_for_cert_frac_90pct"]
    bel_alphas = [0.015, 0.02, 0.03, 0.05]
    bx, by, bvac = _xy(bel_head, bel_alphas)
    ax1.plot(bx[~np.isnan(by)], by[~np.isnan(by)], "s-", color=C["n15"],
             lw=1.5, ms=5, label=r"Belle / Aishell-1 ($\geq$90\% of 20 reseeds)")
    # never-certifies: open marker at the full dev pool
    pool = bel["cal_pool_size"]
    for xi, yi, vac in zip(bx, by, bvac):
        if vac:
            ax1.plot(xi, pool, "s", ms=5, mfc="white", mec=C["n15"], mew=1.1)
    ax1.annotate("vac.\nat pool", (1.5, pool), xytext=(2.15, pool - 1800),
                 fontsize=7, color="black", ha="left",
                 arrowprops=dict(arrowstyle="->", lw=0.6))

    en_cross = eng["crossover"]["wav2vec2_large"]
    ex, ey, evac = _xy(en_cross, [0.02, 0.03, 0.05])
    ax1.plot(ex[~np.isnan(ey)], ey[~np.isnan(ey)], "^-", color=C["whisper"],
             lw=1.5, ms=5, label=r"wav2vec\,2.0 large / LibriSpeech")
    ax1.set_xlabel(r"certified target $\alpha$ (% macro-CER)")
    ax1.set_ylabel(r"min $n_{\mathrm{cal}}$ to certify")
    ax1.set_xlim(0.8, 5.6)
    ax1.set_ylim(0, 16000)
    ax1.legend(loc="upper right", frameon=False, fontsize=6.8)
    ax1.set_title("calibration budget vs target", fontsize=8)

    # Belle power curve: cert fraction vs n_cal
    for a, col, ls, lab in [
        (0.05, C["clean"], "-", r"Belle $\alpha=5\%$"),
        (0.03, C["n15"], "--", r"Belle $\alpha=3\%$"),
        (0.02, C["warn"], ":", r"Belle $\alpha=2\%$"),
    ]:
        rows = [c for c in bel["cells"] if abs(c["alpha"] - a) < 1e-9]
        rows = sorted(rows, key=lambda c: c["n_cal"])
        ax2.plot([c["n_cal"] for c in rows],
                 [c["cert_fraction"] * 100 for c in rows],
                 ls, color=col, lw=1.6, label=lab)
    ax2.axhline(90, color=C["target"], ls="--", lw=0.9)
    ax2.text(250, 93, r"90\% of reseeds", fontsize=7, color="black")
    ax2.set_xlabel(r"calibration size $n_{\mathrm{cal}}$")
    ax2.set_ylabel("% of reseeds that certify")
    ax2.set_ylim(-4, 108)
    ax2.set_xlim(0, 15000)
    ax2.legend(loc="center right", frameon=False, fontsize=6.8)
    ax2.set_title("Belle reseed attainment", fontsize=8)
    ax1.text(-0.16, 1.04, "(a)", transform=ax1.transAxes, fontweight="bold",
             fontsize=10, va="bottom", ha="left")
    ax2.text(-0.16, 1.04, "(b)", transform=ax2.transAxes, fontweight="bold",
             fontsize=10, va="bottom", ha="left")
    save(fig, "fig6_calsize_sweep.pdf")


# ------------------------------------------------------------------ F7 landscape heatmap
# Tightest non-vacuous alpha from landscape.attainment (grid {1.5,2,3,5,10}%),
# with Paraformer/Aishell overridden by the binding 1.9% band (same sources as
# the landscape table). aidatatang cells are withdrawn and omitted.
def fig7():
    att = NUM["landscape"]["attainment"]
    band = _load_frozen("alpha015_2026-07-13/results.json")["binding_band_sweep"]
    bind = next(r for r in band if abs(r["alpha"] - 0.019) < 1e-9)
    bbs = ["paraformer", "belle", "zipformer"]
    corps = ["aishell", "thchs30", "magicdata"]
    ylabels = ["Paraformer", "Belle", "zipformer"]
    xlabels = ["Aishell-1", "THCHS-30", "MagicData"]
    grid = [0.015, 0.02, 0.03, 0.05, 0.1]
    mat = np.full((3, 3), np.nan)
    labels = [[""] * 3 for _ in range(3)]
    vac = np.zeros((3, 3), dtype=bool)
    for i, bb in enumerate(bbs):
        for j, co in enumerate(corps):
            tight = None
            acc = None
            for a in grid:
                cell = att[f"{bb}_{co}_a{a}"]
                if (not cell["vacuous"]) and cell["violations"] == 0:
                    tight = a
                    acc = cell["acc_fraction_mean"]
                    break
            if bb == "paraformer" and co == "aishell":
                tight = 0.019
                acc = bind["mean_acceptance"]
            if tight is None:
                vac[i, j] = True
                labels[i][j] = "vac."
            else:
                mat[i, j] = tight * 100
                labels[i][j] = f"{tight*100:.1f}%\n{acc*100:.1f}% acc."
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="Blues_r", aspect="auto", vmin=1.5, vmax=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(xlabels)
    ax.set_yticks(range(3)); ax.set_yticklabels(ylabels)
    for i in range(3):
        for j in range(3):
            if vac[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="#EEEEEE", edgecolor="none"))
                ax.text(j, i, labels[i][j], ha="center", va="center",
                        fontsize=8, color="black")
            else:
                ax.text(j, i, labels[i][j], ha="center", va="center",
                        fontsize=7.5, color="black")
    ax.set_title(r"tightest certified $\alpha$ (0/20 violations)", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\alpha$ (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    save(fig, "fig7_landscape.pdf")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print("all figures regenerated from frozen result records")
