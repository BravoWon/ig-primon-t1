#!/usr/bin/env python
"""Drilling Execution Forecast — Branch 4 RR re-entry, Bowman County ND.
Geometry x geology, section by section, with a risk-by-phase split (mechanical vs geological)
and the offset-log items that retire each risk.  -> TOSCO_Drilling_Execution_Forecast.pdf"""
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

NAVY, GREEN, AMBER, RED, GREY = "#15293f", "#1e7d34", "#9a6a2f", "#b03a2e", "#5a6470"
FLUID = "#2c6fa6"                                        # fluid / well-control (loss & gain) axis = blue
ANH, PAY, WAT = "#d8dde3", "#cfe6d4", "#d7e6f2"          # anhydrite / pay / water tints
plt.rcParams.update({"font.size": 9, "savefig.dpi": 200})
fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor("white")

# ---- letterhead ----
hb = fig.add_axes([0, 0.925, 1, 0.075]); hb.axis("off")
hb.add_patch(Rectangle((0, 0), 1, 1, fc=NAVY, ec="none"))
hb.text(0.045, 0.58, "VALOR ENERGY PARTNERS", color="white", fontsize=12, weight="bold")
hb.text(0.045, 0.24, "Subsurface & Drilling Analytics", color="#9fb3c8", fontsize=7.5)
hb.text(0.955, 0.56, "CONFIDENTIAL", color="white", fontsize=8, weight="bold", ha="right")
hb.text(0.955, 0.25, "Branch 4 RR (#41700) · Bowman County, ND", color="#9fb3c8", fontsize=7, ha="right")
fig.add_artist(plt.Line2D([0, 1], [0.923, 0.923], color=RED, lw=2))

fig.text(0.045, 0.885, "Drilling Execution Forecast", fontsize=17, weight="bold", color=NAVY)
fig.text(0.045, 0.862, "How the short-radius re-entry should drill — geometry × geology, section by section, "
         "and where the residual risk sits", fontsize=8.5, color=RED)

# ============================ HERO: geometry-through-geology cross-section ============================
axx = fig.add_axes([0.045, 0.45, 0.59, 0.37]); axx.set_xlim(0, 10)
slope, hb_ = 0.16, 0.32
xg = np.linspace(0, 10, 400)
def zc(x): return 4.6 + slope * x + 0.30 * np.sin(0.6 * x + 0.4)
zcg = zc(xg)
top_an, bot_an = zcg - hb_ - 1.7, zcg - hb_
pay_t, pay_b = zcg - hb_, zcg + hb_
owc = zcg + hb_ + 0.05
# stratigraphy
axx.fill_between(xg, top_an, bot_an, color=ANH, ec="none")
axx.fill_between(xg, pay_t, pay_b, color=PAY, ec="none")
axx.fill_between(xg, owc, owc + 2.3, color=WAT, ec="none")
axx.plot(xg, owc, color="#2c6fa6", lw=1.0, ls="--")
# well path
x_v, R = 1.2, 1.3
ykick = zc(x_v) - hb_ - 0.9
axx.plot([x_v, x_v], [2.35, ykick], color="#222", lw=2.4)                 # cased vertical
th = np.linspace(np.pi, np.pi / 2, 60)
cx, cy = x_v + R, ykick
axx.plot(cx + R * np.cos(th), cy + R * np.sin(th), color="#222", lw=2.4)  # curve
xland = x_v + R
xl = np.linspace(xland, 9.6, 200)
yl = zc(xl) - 0.15                                                        # lateral in upper-middle pay
axx.plot(xl, yl, color="#222", lw=2.4)
axx.add_patch(Circle((x_v, ykick), 0.10, fc=RED, ec="white", lw=0.8, zorder=5))
axx.add_patch(Circle((xland, zc(xland) - 0.15), 0.10, fc=GREEN, ec="white", lw=0.8, zorder=5))
# labels
axx.text(x_v - 0.15, 2.75, "existing cased\nvertical (re-entry)", fontsize=6.8, color="#222", ha="left", va="top")
axx.text(x_v + 0.18, ykick - 0.10, "kickoff /\nwindow", fontsize=6.8, color=RED, ha="left", va="bottom", weight="bold")
axx.text(xland + 0.05, zc(xland) - 0.55, "land in 'B'", fontsize=6.8, color=GREEN, ha="left", weight="bold")
axx.text(6.4, zc(6.4) - 0.20, "lateral held in upper-middle 'B'", fontsize=7.2, color="#222", ha="center", weight="bold")
axx.text(8.9, zc(8.9) - hb_ - 0.85, "anhydrite seal (ceiling marker)", fontsize=6.8, color="#5b6166", ha="right", va="center")
axx.text(8.9, zc(8.9) + hb_ + 1.15, "OWC / tight lime — water leg (floor)", fontsize=6.8, color="#2c6fa6", ha="right", va="center")
axx.text(0.15, zc(0.4), "Red River 'B'\n~6 ft pay", fontsize=6.8, color=GREEN, ha="left", va="center", weight="bold")
axx.set_ylim(top_an.min() - 0.2, (owc + 2.3).max()); axx.invert_yaxis()
axx.set_xticks([]); axx.set_yticks([])
for s in axx.spines.values(): s.set_visible(False)
axx.set_title("Geometry through the geology: vertical re-entry → short-radius curve → thin-pay lateral",
              loc="left", color=NAVY, fontsize=9.3)

# ============================ risk-by-phase stacked bar ============================
axr = fig.add_axes([0.70, 0.47, 0.255, 0.33])
labels = ["Kickoff", "Curve", "Land", "Lateral", "Reach"]
mech  = np.array([0.100, 0.050, 0.012, 0.012, 0.025])
geo   = np.array([0.005, 0.018, 0.020, 0.020, 0.003])
fluid = np.array([0.000, 0.005, 0.012, 0.030, 0.010])    # loss & gain (see Fluid Balance sheet)
xb = np.arange(5)
axr.bar(xb, mech * 100, color=NAVY, width=0.66, label="mechanical")
axr.bar(xb, geo * 100, bottom=mech * 100, color=GREEN, width=0.66, label="geological")
axr.bar(xb, fluid * 100, bottom=(mech + geo) * 100, color=FLUID, width=0.66, label="fluid (loss/gain)")
axr.set_xticks(xb); axr.set_xticklabels(labels, fontsize=7)
axr.set_ylabel("residual risk (pts)", fontsize=7.5)
axr.tick_params(axis="y", labelsize=7)
for sp in ("top", "right"): axr.spines[sp].set_visible(False)
axr.legend(loc="upper right", fontsize=6.6, frameon=False)
tot = (mech + geo + fluid).sum() * 100
axr.set_title(f"Where the ~{tot:.0f}% residual risk sits\n(Σ ≈ {tot:.0f}% → P(success) ~ {100-tot:.0f}%)",
              loc="left", color=NAVY, fontsize=8.8)

# ============================ phase cards ============================
cards = [
 ("1 · Re-entry & Kickoff", RED, [
  ("GEOM", "cut window in casing; sidetrack"),
  ("GEOL", "steel & cement, not rock"),
  ("ROP",  "n/a"),
  ("RISK", "casing / cement-bond integrity")]),
 ("2 · Short-radius Curve", AMBER, [
  ("GEOM", "high build rate / DLS; tool-face fight"),
  ("GEOL", "abrasive anhydrite ↔ dolomite cycles"),
  ("ROP",  "slow; real bit wear"),
  ("RISK", "exit curve off-TVD; torque & drag")]),
 ("3 · Land in the 'B'", GREEN, [
  ("GEOM", "flatten into a ~6-ft window"),
  ("GEOL", "anhydrite ceiling, OWC floor"),
  ("ROP",  "—"),
  ("RISK", "land high (no pay) / low (water)")]),
 ("4 · Lateral in Pay", GREEN, [
  ("GEOM", "hold upper-middle 'B'"),
  ("GEOL", "fractured dolomite — drills fast"),
  ("ROP",  "fast"),
  ("RISK", "lost circulation; thin-zone steer")]),
 ("5 · Reach / Hole Mechanics", NAVY, [
  ("GEOM", "T&D limits lateral length"),
  ("GEOL", "easy cleaning at angle"),
  ("ROP",  "steady"),
  ("RISK", "torque/drag; ECD vs losses & OWC")]),
]
cw = (0.91 - 4 * 0.012) / 5
for i, (title, c, entries) in enumerate(cards):
    x = 0.045 + i * (cw + 0.012)
    ax = fig.add_axes([x, 0.165, cw, 0.235]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.05",
                                fc="#f6f8fa", ec=c, lw=1.4))
    ax.add_patch(Rectangle((0.0, 0.86), 1.0, 0.14, fc=c, ec="none", clip_on=True))
    ax.text(0.5, 0.93, title, ha="center", va="center", color="white", fontsize=7.5, weight="bold")
    yy = 0.78
    for tag, val in entries:
        ax.text(0.07, yy, tag, fontsize=6.3, color=c, weight="bold", va="top")
        wrapped = textwrap.wrap(val, width=19) or [""]
        for j, wl in enumerate(wrapped):
            ax.text(0.34, yy - j * 0.072, wl, fontsize=6.3, color="#222", va="top")
        yy -= 0.072 * len(wrapped) + 0.043

# ============================ retires-the-risk strip ============================
axb = fig.add_axes([0.045, 0.072, 0.91, 0.072]); axb.axis("off")
axb.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.008,rounding_size=0.03",
                             fc="#eef4ee", ec=GREEN, lw=1.2))
axb.text(0.012, 0.80, "WHAT RETIRES THE RISK — before the curve is cut", color=GREEN, fontsize=8.5,
         weight="bold", va="top")
items = [
 ("Original Branch 4 CBL / casing log", "retires kickoff & casing risk — the biggest single lever"),
 ("Pilot-hole logs ('B' / anhydrite / OWC TVD)", "retires landing risk — land to a known depth"),
 ("Offset / fleet dip model (sheaf prior)", "retires steering lag — holds ~95% in-zone"),
]
for i, (h, d) in enumerate(items):
    x = 0.012 + i * 0.335
    axb.text(x, 0.50, f"✓ {h}", fontsize=6.7, color="#143", weight="bold", va="top")
    axb.text(x + 0.012, 0.25, d, fontsize=6.2, color="#3a4a3a", va="top")

# ---- footer ----
fig.add_artist(plt.Line2D([0.045, 0.955], [0.055, 0.055], color="#cfd6de", lw=1.0))
fig.text(0.045, 0.040, "CONFIDENTIAL — Marlo Operating / Valor Energy Partners.  Residual-risk split is structured judgment on the "
         "execution framework; fluid loss/gain detailed in the companion Fluid Balance sheet; cross-section is a play-model schematic.",
         fontsize=6.4, color=GREY)
fig.text(0.045, 0.027, "Target zone NDIC-confidential to 12/19/2026; confirm zone + casing integrity before cutting the window. "
         "Prepared 23 Jun 2026.", fontsize=6.4, color=GREY)
fig.text(0.955, 0.033, "Page 1 of 1", fontsize=6.4, color=GREY, ha="right")

fig.savefig("TOSCO_Drilling_Execution_Forecast.pdf"); plt.close(fig)
print("wrote TOSCO_Drilling_Execution_Forecast.pdf")
