#!/usr/bin/env python
"""Fluid Balance — Loss & Gain Risk.  Drilling-system fluid (mud / ECD) vs formation pressure:
the mud-weight window pinches inside the Red River 'B' because the natural fractures (deliverability)
drop the loss ceiling while the OWC just below raises the gain floor.  -> TOSCO_Fluid_Balance_Risk.pdf"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

NAVY, GREEN, AMBER, RED, GREY = "#15293f", "#1e7d34", "#9a6a2f", "#b03a2e", "#5a6470"
FLUID = "#2c6fa6"                                   # well-control / fluid axis = blue
GTINT, ATINT, RTINT = "#e7f1e9", "#f6efdf", "#f6e4e1"
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

fig.text(0.045, 0.885, "Fluid Balance — Loss & Gain Risk", fontsize=17, weight="bold", color=NAVY)
fig.text(0.045, 0.862, "Drilling-system fluid (mud / ECD) vs formation pressure — the mud-weight window pinches "
         "inside the pay, where losses and gains meet", fontsize=8.5, color=RED)

# ============================ HERO: mud-weight window vs depth ============================
axw = fig.add_axes([0.055, 0.37, 0.50, 0.45])
zg = np.linspace(8300, 9200, 400)
owc = 9000.0; btop, bbase = 8965.0, 9000.0; bcen = 8985.0
pore = 8.55 + 0.00030 * (zg - 8300)
pore = np.where(zg > owc, pore + 0.12, pore)                 # water leg slightly higher
frac_m = 13.2 + 0.00090 * (zg - 8300)                        # matrix fracture gradient (high)
notch = (frac_m - 9.70) * np.exp(-0.5 * ((zg - bcen) / 60.0) ** 2)
loss_ceil = frac_m - notch                                   # natural-fracture loss threshold dips in 'B'
mw = np.full_like(zg, 8.95)                                  # planned static mud weight (slight overbalance)
ecd = 8.95 + 0.35 + 0.00040 * (zg - 8300)                    # circulating (dynamic) density
XL, XR = 8.2, 11.5
# zones
axw.fill_betweenx(zg, XL, pore, color=RTINT)                 # under pore -> GAIN
axw.fill_betweenx(zg, pore, np.clip(loss_ceil, XL, XR), color=GTINT)   # safe corridor
axw.fill_betweenx(zg, np.clip(loss_ceil, XL, XR), XR, color=ATINT)     # over ceiling -> LOSS
# pay band + OWC
axw.axhspan(btop, bbase, color=GREEN, alpha=0.10)
axw.plot([XL, XR], [owc, owc], color=FLUID, lw=1.0, ls="--")
# lines
axw.plot(pore, zg, color=FLUID, lw=1.8, label="pore pressure (gain floor)")
axw.plot(loss_ceil, zg, color=AMBER, lw=1.8, label="fracture / loss ceiling")
axw.plot(mw, zg, color=NAVY, lw=1.6, ls="--", label="planned mud wt (static)")
axw.plot(ecd, zg, color=NAVY, lw=1.9, label="ECD (circulating)")
# annotations
axw.text(8.30, 8420, "GAIN\nkick / water\nfrom OWC", color=RED, fontsize=7.2, weight="bold", va="top")
axw.text(11.35, 8420, "LOSS\nmud into\nfractures", color=AMBER, fontsize=7.2, weight="bold", va="top", ha="right")
beak_x = loss_ceil[np.argmin(np.abs(zg - bcen))]
axw.annotate("pinch in the 'B':\nnatural fractures drop\nthe loss ceiling",
             xy=(beak_x, bcen), xytext=(9.82, 9045), color=AMBER, fontsize=6.8,
             weight="bold", va="top", ha="left",
             arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.0))
axw.annotate("Red River 'B'\n~6 ft pay", xy=(8.30, bcen), xytext=(8.28, 8700),
             color=GREEN, fontsize=6.6, weight="bold", va="center", ha="left",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
axw.text(XR - 0.05, owc - 6, "OWC", color=FLUID, fontsize=7.0, ha="right", va="bottom", weight="bold")
axw.text(ecd[60] + 0.04, zg[60], "ECD", color=NAVY, fontsize=6.8)
axw.text(mw[60] - 0.04, zg[60], "MW", color=NAVY, fontsize=6.8, ha="right")
axw.set_xlim(XL, XR); axw.set_ylim(zg.min(), zg.max()); axw.invert_yaxis()
axw.set_xlabel("equivalent mud weight (ppg)", fontsize=8); axw.set_ylabel("TVD (ft)", fontsize=8)
axw.tick_params(labelsize=7)
axw.legend(loc="lower left", fontsize=6.4, frameon=True, framealpha=0.9)
axw.set_title("The mud-weight window narrows inside the pay", loc="left", color=NAVY, fontsize=9.3)

# ============================ R1: two failure modes ============================
fig.text(0.625, 0.815, "TWO WAYS THE FLUID BALANCE FAILS", color=NAVY, fontsize=9, weight="bold")
modes = [
 (0.715, ATINT, AMBER, "OVERBALANCED  —  ECD above formation pressure",
  "→  LOSSES: mud invades the natural fractures; you lose returns, burn mud,\n"
  "     and can drill blind to a kick building below."),
 (0.585, RTINT, RED, "UNDERBALANCED  —  ECD below formation pressure",
  "←  GAIN / KICK: formation fluid enters the well; here that means WATER\n"
  "     pulled up from the OWC sitting just under the 'B'."),
]
for y, fc, ec, head, body in modes:
    ax = fig.add_axes([0.625, y, 0.33, 0.115]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01,rounding_size=0.05",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(0.04, 0.80, head, color=ec, fontsize=7.4, weight="bold", va="top")
    ax.text(0.04, 0.50, body, color="#2a2320", fontsize=6.8, va="top", linespacing=1.35)

# ============================ R2: loss/gain risk by phase ============================
fig.text(0.625, 0.520, "LOSS / GAIN RISK BY PHASE", color=NAVY, fontsize=9, weight="bold")
axt = fig.add_axes([0.625, 0.375, 0.33, 0.125]); axt.axis("off")
axt.set_xlim(0, 1); axt.set_ylim(0, 1)
LV = {"Low": (GTINT, GREEN), "Med": (ATINT, AMBER), "High": (RTINT, RED)}
prows = [("Curve", "Low", "Low"),
         ("Landing", "Med", "Med"),
         ("Lateral", "High", "Med"),
         ("Trips / connections", "Med", "Med")]
cx = [0.0, 0.52, 0.76]; cw = [0.50, 0.22, 0.22]
axt.text(cx[0] + 0.01, 0.92, "phase", fontsize=6.8, weight="bold", color=NAVY, va="center")
axt.text(cx[1] + cw[1] / 2, 0.92, "LOSS", fontsize=6.8, weight="bold", color=AMBER, va="center", ha="center")
axt.text(cx[2] + cw[2] / 2, 0.92, "GAIN", fontsize=6.8, weight="bold", color=RED, va="center", ha="center")
rh = 0.20
for i, (ph, lo, ga) in enumerate(prows):
    yy = 0.80 - i * rh
    axt.text(cx[0] + 0.01, yy - rh / 2, ph, fontsize=6.8, color="#222", va="center")
    for j, lvl in ((1, lo), (2, ga)):
        f, e = LV[lvl]
        axt.add_patch(Rectangle((cx[j], yy - rh + 0.02), cw[j], rh - 0.04, fc=f, ec=e, lw=1.0))
        axt.text(cx[j] + cw[j] / 2, yy - rh / 2, lvl, fontsize=6.6, color=e, weight="bold",
                 ha="center", va="center")

# ============================ bottom: mitigation toolkit ============================
tk = [
 (0.055, AMBER, "LOSS MANAGEMENT",
  "• LCM sized to the fractures, staged & ready\n"
  "• hold ECD down: flow rate, hole cleaning, RPM,\n   slow connections & trips (surge)\n"
  "• cure losses before drilling ahead — don't chase\n• air-mist / UBD only where clear of the OWC"),
 (0.375, RED, "GAIN / WELL CONTROL",
  "• flow checks + active kick detection\n"
  "• trip & connection discipline (avoid swab)\n"
  "• keep a trip margin; PWD to watch BHP live\n"
  "• geosteer to hold standoff ABOVE the OWC —\n   the cheapest gain insurance there is"),
 (0.695, FLUID, "NARROW-WINDOW TOOL  ·  MPD",
  "Managed-pressure drilling: closed-loop, holds a\n"
  "precise bottomhole pressure and reacts in seconds\n"
  "to BOTH losses and gains — the right tool where the\n"
  "loss ceiling and gain floor pinch in the pay. With the\n"
  "OWC right below, hold a precise slight overbalance."),
]
for x, ec, head, body in tk:
    ax = fig.add_axes([x, 0.105, 0.26, 0.205]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.012,rounding_size=0.04",
                                fc="#f6f8fa", ec=ec, lw=1.4))
    ax.add_patch(Rectangle((0.0, 0.84), 1.0, 0.16, fc=ec, ec="none"))
    ax.text(0.5, 0.92, head, ha="center", va="center", color="white", fontsize=7.6, weight="bold")
    ax.text(0.05, 0.74, body, fontsize=6.6, color="#222", va="top", linespacing=1.45)

# ---- footer ----
fig.add_artist(plt.Line2D([0.055, 0.955], [0.075, 0.075], color="#cfd6de", lw=1.0))
fig.text(0.055, 0.058, "CONFIDENTIAL — Marlo Operating / Valor Energy Partners.  Pressure curves are a play-model template "
         "(Williston SW Red River); calibrate to the Branch 4 pilot-hole pressures, mud logs and any offset LOT/FIT.",
         fontsize=6.4, color=GREY)
fig.text(0.055, 0.044, "Loss/gain is a third risk axis alongside mechanical and geological (see Drilling Execution Forecast). "
         "Prepared 23 Jun 2026.", fontsize=6.4, color=GREY)
fig.text(0.955, 0.050, "Page 1 of 1", fontsize=6.4, color=GREY, ha="right")

fig.savefig("TOSCO_Fluid_Balance_Risk.pdf"); plt.close(fig)
print("wrote TOSCO_Fluid_Balance_Risk.pdf")
