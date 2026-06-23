#!/usr/bin/env python
"""Production Expectation Matrix — well/completion strategies for the Tosco / Bowman County setting,
anchored on the same execution-stage chance-of-success framework, with a geology-of-fracking analysis."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

NAVY, GREEN, AMBER, RED, GREY = "#15293f", "#1e7d34", "#9a6a2f", "#b03a2e", "#5a6470"
TINT = {GREEN: "#eaf3ec", AMBER: "#f7f0e3", RED: "#f6e8e6"}
plt.rcParams.update({"font.size": 9, "savefig.dpi": 200})
fig = plt.figure(figsize=(11, 8.5)); fig.patch.set_facecolor("white")

# letterhead
hb = fig.add_axes([0, 0.925, 1, 0.075]); hb.axis("off")
hb.add_patch(Rectangle((0, 0), 1, 1, fc=NAVY, ec="none"))
hb.text(0.045, 0.58, "VALOR ENERGY PARTNERS", color="white", fontsize=12, weight="bold")
hb.text(0.045, 0.24, "Subsurface & Drilling Analytics", color="#9fb3c8", fontsize=7.5)
hb.text(0.955, 0.56, "CONFIDENTIAL", color="white", fontsize=8, weight="bold", ha="right")
hb.text(0.955, 0.25, "Bowman County, ND · Red River trend", color="#9fb3c8", fontsize=7, ha="right")
fig.add_artist(plt.Line2D([0, 1], [0.923, 0.923], color=RED, lw=2))

fig.text(0.045, 0.885, "Production Expectation Matrix", fontsize=17, weight="bold", color=NAVY)
fig.text(0.045, 0.862, "By well & completion strategy — anchored on the execution-stage chance-of-success "
         "(same breakdown as the Production-Probability Report)", fontsize=8.5, color=RED)

# ---- matrix ----
cols = [(0.00, 0.205, "Scenario"), (0.205, 0.315, "P(success)*"),
        (0.315, 0.545, "Expected production"), (0.545, 0.690, "Geologic fit"),
        (0.690, 0.875, "Dominant risk"), (0.875, 1.00, "Verdict")]
rows = [
 ("Conventional re-entry\nopen-hole / acid  (Red River 'B')", "64–72%",
  "40–150 Mbbl · IP tens–low-100s bopd\nflat 5–15%/yr decline, long life", "EXCELLENT\nthin fractured dolomite",
  "short-radius re-entry\nmechanics (casing / curve)", "RECOMMEND", GREEN),
 ("New-drill horizontal\nacid  (Red River 'B')", "70–78%",
  "80–200 Mbbl\ncleaner lateral placement", "Excellent",
  "higher cost; loses the\nre-entry savings", "STRONG ALT", GREEN),
 ("Hydraulic frac\nmulti-stage proppant  (RR 'B')", "45–60%",
  "high variance;\nwater-cut risk dominates", "POOR\nthin carbonate over OWC",
  "frac propagates DOWN\ninto the water leg", "AVOID", RED),
 ("Madison / Mission Canyon\nconventional target", "50–62%",
  "facies-dependent\n(porosity-cycle controlled)", "Moderate\ntrap/porosity-controlled, patchy",
  "in / out of the\nporosity fairway", "CONDITIONAL", AMBER),
 ("Bakken / Three Forks\nfrac  (basin margin)", "25–40%",
  "sub-core EURs\non the fairway edge", "Marginal\nedge of the Bakken fairway",
  "play immature here;\nhigh capex", "AVOID HERE", RED),
 ("Vertical re-entry\nno lateral", "70–80%",
  "low (8–15% recovery factor)\nthin pay", "n/a",
  "low rate /\nsub-commercial", "LAST RESORT", AMBER),
]
ax = fig.add_axes([0.045, 0.355, 0.91, 0.475]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
hh = 0.115                                       # header height
rh = (1 - hh) / len(rows)
# header
ax.add_patch(Rectangle((0, 1 - hh), 1, hh, fc=NAVY, ec="none"))
for xl, xr, name in cols:
    ax.text((xl + xr) / 2, 1 - hh / 2, name, ha="center", va="center", color="white",
            fontsize=8.3, weight="bold")
# data rows
for i, (sc, p, prod, geo, risk, verd, c) in enumerate(rows):
    yt = 1 - hh - i * rh
    ax.add_patch(Rectangle((0, yt - rh), 1, rh, fc=TINT[c], ec="white", lw=1.2))
    cells = [sc, p, prod, geo, risk]
    for (xl, xr, _), txt in zip(cols[:5], cells):
        wt = "bold" if xl == 0 else "normal"
        ax.text(xl + 0.008, yt - rh / 2, txt, ha="left", va="center",
                fontsize=7.0, color="#222", weight=wt, linespacing=1.25)
    # verdict badge
    bx0, bx1 = cols[5][0], cols[5][1]
    ax.add_patch(FancyBboxPatch((bx0 + 0.006, yt - rh + 0.012), (bx1 - bx0) - 0.012, rh - 0.024,
                                boxstyle="round,pad=0.004,rounding_size=0.02", fc=c, ec="none"))
    ax.text((bx0 + bx1) / 2, yt - rh / 2, verd, ha="center", va="center", color="white",
            fontsize=7.0, weight="bold")
# column separators
for xl, _, _ in cols[1:]:
    ax.plot([xl, xl], [0, 1 - hh], color="white", lw=1.2)

fig.text(0.045, 0.345, "* P(success) = execution-stage chance of success (reservoir → casing → curve → "
         "lateral placement → rate). For conventional scenarios the dominant lever is short-radius re-entry "
         "execution, NOT reservoir presence.", fontsize=7.2, color=GREY, va="top")

# ---- fracking geology analysis ----
axf = fig.add_axes([0.045, 0.115, 0.62, 0.175]); axf.axis("off")
axf.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.03",
                             fc="#f6e8e6", ec=RED, lw=1.3))
axf.text(0.022, 0.90, "GEOLOGY OF FRACKING HERE — why it is the wrong tool for the 'B'",
         color=RED, fontsize=9, weight="bold", va="top")
axf.text(0.022, 0.70,
         "The Red River 'B' is a thin (~6–10 ft) NATURALLY-FRACTURED dolomite sitting directly above tight\n"
         "limestone and the oil-water contact. It produces through its natural fractures + acid — not proppant.\n"
         "A multi-stage hydraulic frac in a carbonate this thin and this close to the OWC propagates fractures\n"
         "DOWN into the water leg: you trade oil for water cut, and add mechanical and cost risk for little upside.\n"
         "Hydraulic fracking is essential in the tight Bakken / Three Forks shale — but Bowman County is on the\n"
         "MARGIN of that fairway, where Bakken economics are sub-core. Right tool here: lateral exposure + acid,\n"
         "geosteered to stay in the upper 'B' above the OWC (see the Geologic Window report).",
         color="#3a2320", fontsize=7.6, va="top", linespacing=1.45)

# ---- 'other like wells' note ----
axo = fig.add_axes([0.685, 0.115, 0.27, 0.175]); axo.axis("off")
axo.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.04",
                             fc="#eef1f5", ec=NAVY, lw=1.1))
axo.text(0.06, 0.90, "READING THE MATRIX", color=NAVY, fontsize=9, weight="bold", va="top")
axo.text(0.06, 0.66,
         "Green = recommended completion\nfor this rock.\n\n"
         "Amber = conditional — only with\nseismic / log support.\n\n"
         "Red = wrong tool or wrong place\n(fracking the 'B'; Bakken on the\nmargin).",
         color="#222", fontsize=7.6, va="top", linespacing=1.4)

# footer
fig.add_artist(plt.Line2D([0.045, 0.955], [0.075, 0.075], color="#cfd6de", lw=1.0))
fig.text(0.045, 0.058, "CONFIDENTIAL — Marlo Operating / Valor Energy Partners.  P(success) figures are structured "
         "judgment on the execution framework; production ranges are SW-ND Red River / Williston analogs (not economics).",
         fontsize=6.6, color=GREY)
fig.text(0.045, 0.044, "Target zone NDIC-confidential to 12/19/2026; confirm zone + casing integrity before selecting "
         "a completion. Prepared 23 Jun 2026.", fontsize=6.6, color=GREY)

fig.savefig("TOSCO_Production_Expectation_Matrix.pdf"); plt.close(fig)
print("wrote TOSCO_Production_Expectation_Matrix.pdf")
