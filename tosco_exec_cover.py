#!/usr/bin/env python
"""One-page executive cover (letterhead) distilled from the two Tosco ND client deliverables."""
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

NAVY, ACC, GO, AMBER, GREY = "#15293f", "#c0392b", "#1e7d34", "#9a6a2f", "#5a6470"
FLUID = "#2c6fa6"                                        # fluid / well-control (loss & gain) layer
plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 200})
fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")

# ---- letterhead band ----
hb = fig.add_axes([0, 0.93, 1, 0.07]); hb.axis("off")
hb.add_patch(Rectangle((0, 0), 1, 1, fc=NAVY, ec="none"))
hb.text(0.06, 0.60, "VALOR ENERGY PARTNERS", color="white", fontsize=13, weight="bold")
hb.text(0.06, 0.24, "Subsurface & Drilling Analytics", color="#9fb3c8", fontsize=8.0)
hb.text(0.94, 0.58, "CONFIDENTIAL", color="white", fontsize=8.5, weight="bold", ha="right")
hb.text(0.94, 0.26, "For Client Review — 23 Jun 2026", color="#9fb3c8", fontsize=7.5, ha="right")
fig.add_artist(plt.Line2D([0, 1], [0.928, 0.928], color=ACC, lw=2.2))

# ---- masthead ----
fig.text(0.06, 0.875, "Executive Summary", fontsize=22, weight="bold", color=NAVY)
fig.text(0.06, 0.845, "Tosco Branch Re-Entry Wells — Production Potential, Geology & Drilling Execution",
         fontsize=12, color=ACC)
fig.text(0.06, 0.823, "Bowman County, North Dakota  ·  Williston Basin southern margin (Red River trend)",
         fontsize=8.5, color=GREY)
fig.text(0.06, 0.800, "Prepared for:  Marlo Operating Company · Valor Energy Partners      |      "
         "Wells:  Branch 4 RR (#41700) · Branch 2 (#41258)",
         fontsize=8.0, color="#333")
fig.add_artist(plt.Line2D([0.06, 0.94], [0.788, 0.788], color="#cfd6de", lw=1.0))

# ---- bottom line ----
fig.text(0.06, 0.762, "Bottom line", fontsize=11.5, weight="bold", color=NAVY)
fig.text(0.06, 0.745,
         "Both wells are RE-ENTRIES into a pool Marlo just discovered and is already producing (sister well\n"
         "Branch 5). That makes them DEVELOPMENT-CLASS step-outs, not wildcats: the reservoir is effectively\n"
         "proven, and the outcome turns on EXECUTION of the short-radius re-entry rather than on whether oil is\n"
         "present. We assess a ~64–72% probability of a successful producer (modest size), with the dominant\n"
         "risk being mechanical — and largely knowable in advance from the original wellbore's logs.",
         fontsize=9.3, color="#222", linespacing=1.5, va="top")

# ---- key metrics row ----
metrics = [("~64–72%", "Probability of a\nsuccessful producer", GO),
           ("Development-\nclass re-entry", "Not a wildcat\n(pool is producing)", NAVY),
           ("Execution", "Dominant risk:\nshort-radius mechanics", ACC),
           ("~65–150 bopd", "IP if successful · EUR 40–150 Mbbl\n(offset Branch 5 ≈65; discovery ~497 bopd)", GREY)]
for i, (big, sub, c) in enumerate(metrics):
    x = 0.06 + i * 0.225
    ax = fig.add_axes([x, 0.565, 0.205, 0.085]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="#f5f7fa", ec=c, lw=1.3))
    ax.text(0.5, 0.66, big, ha="center", va="center", fontsize=11.5, weight="bold", color=c)
    ax.text(0.5, 0.20, sub, ha="center", va="center", fontsize=7.0, color="#444")

# ---- probability band visual ----
axp = fig.add_axes([0.06, 0.45, 0.88, 0.075])
axp.add_patch(Rectangle((0, 0), 1, 1, fc="#eef1f5", ec="none"))
axp.axvspan(0.40, 0.85, color=GO, alpha=0.20)
axp.axvline(0.68, color=GO, lw=2.2)
axp.text(0.70, 0.80, "central\n~64–72%", color=GO, fontsize=7.6, weight="bold", ha="left", va="top")
axp.text(0.40, -0.32, "P10  40%", color=GREY, fontsize=7.5, ha="center")
axp.text(0.85, -0.32, "P90  85%", color=GREY, fontsize=7.5, ha="center")
axp.text(0.02, 0.5, "wildcat\n~20%", color=GREY, fontsize=6.6, va="center")
axp.axvline(0.20, color=GREY, lw=1.0, ls=":")
axp.set_xlim(0, 1); axp.set_ylim(0, 1); axp.set_yticks([])
axp.set_xticks([0, .2, .4, .6, .8, 1.0]); axp.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
fig.text(0.06, 0.537, "Chance of success  —  development-class, well above the ~20% wildcat line",
         color=NAVY, fontsize=9.5, weight="bold")

# ---- recommended action ----
axr = fig.add_axes([0.06, 0.315, 0.88, 0.085]); axr.axis("off")
axr.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.015,rounding_size=0.04",
                             fc="#fbf3e6", ec=AMBER, lw=1.4))
axr.text(0.025, 0.74, "RECOMMENDED ACTION", color=AMBER, fontsize=9, weight="bold", va="center")
axr.text(0.025, 0.36, "Pull the original Branch 4 wellbore's casing / cement-bond log and confirm the target zone. "
         "That one\nstep sets the dominant (mechanical) risk and collapses most of the 40–85% band toward the high "
         "end\nbefore committing the curve.", color="#3d2f12", fontsize=8.6, va="center", linespacing=1.4)

# ---- execution risk in three layers ----
fig.text(0.06, 0.298, "EXECUTION RISK — TRACKED IN THREE LAYERS", fontsize=9.5, weight="bold", color=NAVY)
layers = [
 (NAVY,  "Mechanical",
  "Front-loaded — kickoff & casing integrity, knowable in advance from the original wellbore logs."),
 (GO,    "Geological",
  "Landing and holding the thin 'B'; the offset / fleet dip prior keeps the lateral in zone."),
 (FLUID, "Fluid — loss & gain",
  "Mud lost to natural fractures; water gain at the OWC — held with MPD / LCM / ECD discipline."),
]
for i, (c, h, b) in enumerate(layers):
    x = 0.06 + i * 0.2967
    ax = fig.add_axes([x, 0.200, 0.278, 0.088]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.03,rounding_size=0.06",
                                fc="#f5f7fa", ec=c, lw=1.4))
    ax.text(0.5, 0.80, h, ha="center", va="center", color=c, fontsize=8.6, weight="bold")
    ax.text(0.5, 0.37, textwrap.fill(b, 40), ha="center", va="center", color="#333",
            fontsize=6.6, linespacing=1.32)

# ---- in this package: seven documents ----
fig.text(0.06, 0.170, "IN THIS PACKAGE — SEVEN DOCUMENTS", fontsize=10, weight="bold", color=NAVY)
fig.text(0.94, 0.170, "full index & page references on the Contents page", fontsize=7.2, color=GREY,
         ha="right", va="center")
titles = ["Executive Summary (this page)",
          "Production-Probability & Execution Report",
          "Production Expectation Matrix",
          "Geologic / Geosteering Window Report",
          "Drilling Execution Forecast",
          "Fluid Balance — Loss & Gain Risk",
          "Steering Anticipation at Scale"]
rows = [0.142, 0.115, 0.088, 0.061]
for i, t in enumerate(titles):
    x = 0.07 if i < 4 else 0.52
    y = rows[i] if i < 4 else rows[i - 4]
    fig.text(x, y, f"{i + 1}.  {t}", fontsize=8.0, color="#222")

# ---- footer ----
fig.add_artist(plt.Line2D([0.06, 0.94], [0.055, 0.055], color="#cfd6de", lw=1.0))
fig.text(0.06, 0.038, "CONFIDENTIAL — Marlo Operating / Valor Energy Partners.  Scope: probability of a producing well, "
         "not economics.", fontsize=6.8, color=GREY)
fig.text(0.06, 0.024, "Target formation NDIC-confidential to 12/19/2026; geosteering figures are a play-model template "
         "to calibrate to actual wellbore logs.", fontsize=6.8, color=GREY)
fig.text(0.94, 0.031, "Page 1 of 1", fontsize=6.8, color=GREY, ha="right")

fig.savefig("TOSCO_Executive_Cover.pdf"); plt.close(fig)
print("wrote TOSCO_Executive_Cover.pdf")
