#!/usr/bin/env python
"""Geologic / geosteering report: drilling the Tosco Branch 4 RR lateral WITHIN the Red River 'B'
pay window for best effect (max oil-saturated reservoir contact). Schematic PLAY MODEL (public Cedar
Hills Red River 'B' character + this well's parameters) — to be calibrated to the actual original-
wellbore logs + Branch 5 tops (confidential to 12/19/2026)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyArrow, Polygon

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 130, "savefig.dpi": 200})
INK, ACC, OIL, WIN, SEAL, WAT = "#1a2b3c", "#c0392b", "#1e7d34", "#27ae60", "#7d6fb0", "#2e86c1"

# ---- schematic stratigraphy (representative TVD, ft; FLAGGED schematic) ----
STRAT = [  # (top, base, name, color, lith)
 (9250, 9335, "Stony Mtn / Stonewall", "#cfcabb", "argillaceous carbonate / shale"),
 (9335, 9362, "Red River 'A'",          "#e3d4a8", "tight dolomite"),
 (9362, 9374, "ANHYDRITE  (top seal)",  "#cfc6e8", "anhydrite — geosteer DOWN if seen"),
 (9374, 9384, "RED RIVER 'B'  TARGET",  "#bfe6c6", "porous dolomite  8-14% phi  (PAY)"),
 (9384, 9420, "Tight limestone (seal)", "#cdd9e3", "tight lime — geosteer UP if seen"),
 (9420, 9485, "Red River 'C' / 'D'",    "#d9c3a3", "dolomite + kukersite source"),
]
SUB = (9375.5, 9379.5)          # geosteering sub-target inside the 'B' (upper-middle)
OWC = 9389.0                    # oil-water contact (downdip, below 'B' base here = full oil column)

def footer(fig, pg):
    fig.text(0.5, 0.022, "PLAY-MODEL SCHEMATIC (public Cedar Hills Red River 'B' + well parameters) — "
             "calibrate to actual wellbore logs.  Confidential — Marlo / Valor.", ha="center",
             fontsize=6.2, color="#999")
    fig.text(0.93, 0.022, f"p.{pg}", fontsize=6.2, color="#999")

def band(fig, t, sub):
    fig.text(0.06, 0.945, t, fontsize=15, weight="bold", color=INK)
    fig.text(0.06, 0.917, sub, fontsize=8.5, color=ACC)
    fig.add_artist(plt.Line2D([0.06, 0.94], [0.907, 0.907], color=INK, lw=1.2))

pdf = PdfPages("TOSCO_4RR_Geologic_Window_Report.pdf")

# ============================================================ PAGE 1 — model + section ====
fig = plt.figure(figsize=(11, 8.5)); band(fig,
    "Tosco Branch 4 RR — Geologic Window for Drilling",
    "Keep the lateral inside the Red River 'B' porous dolomite — between the anhydrite seal above and the tight lime / OWC below — for maximum oil contact.")

# -- type-log / stratigraphic column (left) --
axL = fig.add_axes([0.06, 0.10, 0.30, 0.74]); axL.set_title("Stratigraphic column + schematic GR", loc="left", color=INK)
for top, base, nm, c, lith in STRAT:
    axL.add_patch(Rectangle((0, top), 1.0, base - top, fc=c, ec="#888", lw=0.6))
    axL.text(0.04, (top + base) / 2 - 1, nm, fontsize=7.3, weight="bold", va="center", color=INK)
    axL.text(0.04, (top + base) / 2 + 6, lith, fontsize=6.0, va="center", color="#555", style="italic")
axL.add_patch(Rectangle((0, SUB[0]), 1.0, SUB[1] - SUB[0], fc="none", ec=ACC, lw=2.0, ls="--"))
axL.text(0.5, np.mean(SUB) - 13, "steer band", fontsize=6.4, color=ACC, ha="center", weight="bold")
# schematic GR curve (low=carbonate/anhydrite, high=shale)
z = np.linspace(9250, 9485, 400)
gr = (40 + 70 * np.exp(-((z - 9292) / 25) ** 2)          # Stony Mtn shale high
      + 10 * (z > 9420) - 25 * np.exp(-((z - 9368) / 6) ** 2)  # anhydrite low
      + 8 * np.exp(-((z - 9379) / 5) ** 2))               # B slight
axL.plot(0.15 + 0.7 * (gr - gr.min()) / (gr.max() - gr.min()), z, color="#333", lw=1.0)
axL.set_ylim(9485, 9250); axL.set_xlim(0, 1.0); axL.set_xticks([]); axL.set_ylabel("TVD (ft, schematic)")

# -- geosteering cross-section (right) --
axR = fig.add_axes([0.43, 0.10, 0.52, 0.74]); axR.set_title("Geosteering cross-section — stay in the 'B' window", loc="left", color=INK)
X = np.linspace(-300, 2600, 600)
dip = 0.0032                                              # gentle structural dip (ft/ft)
btop = 9374 + dip * (X)                                   # 'B' top dips down to the right
bbase = 9384 + dip * X; owc = OWC + dip * X
axR.fill_between(X, btop, bbase, color=WIN, alpha=0.22, label="Red River 'B' window (pay)")
axR.fill_between(X, 9362 + dip * X, btop, color=SEAL, alpha=0.30, label="anhydrite seal")
axR.fill_between(X, bbase, owc, color="#cdd9e3", alpha=0.5)
axR.fill_between(X, owc, owc + 25, color=WAT, alpha=0.18)
axR.plot(X, owc, color=WAT, lw=1.3, ls="-.");  axR.text(1350, owc[int(len(owc)*0.6)] + 8, "OIL–WATER CONTACT", color=WAT, fontsize=7.5, weight="bold")
# wellbore: vertical re-entry -> short-radius curve -> lateral in mid-'B'
kop_x, kop_z = -250, 9300
cz = np.linspace(0, 1, 60); curve_x = kop_x + 250 * np.sin(cz * np.pi / 2)
curve_z = kop_z + (9379 - kop_z) * (1 - np.cos(cz * np.pi / 2))
latx = np.linspace(0, 2550, 200); latz = 9377.5 + dip * latx          # tracks mid-'B'
axR.plot([kop_x, kop_x], [9250, kop_z], color="k", lw=2.2)
axR.plot(curve_x, curve_z, color="k", lw=2.2)
axR.plot(latx, latz, color="k", lw=2.6)
axR.plot(latx[::18], latz[::18], "o", color=ACC, ms=2.5)
axR.annotate("re-entry\n(existing vertical)", (kop_x, 9275), (kop_x - 40, 9268), fontsize=7, ha="center")
axR.annotate("short-radius curve\n25-30°/100 ft", (curve_x[30], curve_z[30]), (650, 9305),
             fontsize=7, ha="center", arrowprops=dict(arrowstyle="->", color="#444"))
axR.annotate("LANDING POINT\nupper-middle 'B'", (latx[5], latz[5]), (300, 9362),
             fontsize=7, ha="center", color=ACC, arrowprops=dict(arrowstyle="->", color=ACC))
axR.text(1300, 9376.0, "lateral steered in pay window, above OWC", fontsize=7.5, color=INK, weight="bold")
axR.set_ylim(9430, 9250); axR.set_xlim(-320, 2620)
axR.set_xlabel("along-hole offset within Sec 19 NE/4 (ft, schematic)"); axR.set_ylabel("TVD (ft)")
axR.legend(loc="lower left", fontsize=6.8, frameon=False)
footer(fig, 1); pdf.savefig(fig); plt.close(fig)

# ============================================================ PAGE 2 — playbook ====
fig = plt.figure(figsize=(11, 8.5)); band(fig,
    "Geosteering Playbook — staying in window for best effect",
    "Marker signatures, steering rules, and the optimization that maximizes oil-saturated reservoir contact.")

fig.text(0.06, 0.85, "Marker beds & log signatures (what the MWD gamma is telling you)", fontsize=10.5, weight="bold", color=INK)
rows = [("Anhydrite SEAL (above 'B')", "very low GR, high density/PE, no porosity", "you are HIGH — steer DOWN into the 'B'"),
        ("Red River 'B' PAY (target)", "low-moderate GR, neutron-density porosity separation (dolomite)", "ON TARGET — hold the steer band"),
        ("Tight limestone (below 'B')", "low GR, no porosity, density tight", "you are LOW — steer UP; OWC risk below"),
        ("Oil–water contact (OWC)", "resistivity drop / increasing water on shows", "STOP descending — stay updip / above")]
y = 0.80
fig.text(0.07, y, "Marker", fontsize=8, weight="bold", color="#666"); fig.text(0.30, y, "Log signature", fontsize=8, weight="bold", color="#666"); fig.text(0.66, y, "Steering action", fontsize=8, weight="bold", color="#666")
fig.add_artist(plt.Line2D([0.06, 0.94], [y - 0.008, y - 0.008], color="#bbb", lw=0.8))
for i, (m, s, a) in enumerate(rows):
    yy = y - 0.04 - i * 0.045
    fig.text(0.07, yy, m, fontsize=8, color=INK, weight="bold"); fig.text(0.30, yy, s, fontsize=7.6, color="#444"); fig.text(0.66, yy, a, fontsize=7.8, color=ACC if "OWC" not in m else WAT)

fig.text(0.06, 0.555, "Steering rules — the window", fontsize=10.5, weight="bold", color=INK)
rules = ["Target the UPPER-MIDDLE of the 'B' (≈2-6 ft below the anhydrite seal): maximum standoff from the OWC while keeping a buffer below the seal.",
         "Maintain steer band within the ±2-3 ft 'sweet spot'; the 'B' is only ~8-10 ft thick, so small dip changes move you out of zone.",
         "FOLLOW STRUCTURAL DIP using the MWD gamma + correlation to the original wellbore and Branch 5 tops — adjust inclination to track the dipping 'B'.",
         "Never drill below the OWC: descending into the water leg trades oil footage for water cut (the #1 re-entry failure mode in this play)."]
for i, r in enumerate(rules):
    fig.text(0.07, 0.515 - i * 0.045, f"•  {r}", fontsize=8.2, color="#333", va="top")

fig.text(0.06, 0.315, "Best effect for the client", fontsize=10.5, weight="bold", color=OIL)
fig.text(0.07, 0.288,
         "Maximize productive lateral footage inside the porous, oil-saturated 'B' dolomite. Every foot held in the steer band\n"
         "is reservoir contact; every foot in the seal or water leg is wasted hole. A clean in-zone lateral in the upper 'B',\n"
         "updip of the OWC, delivers the field's best per-well recovery (Cedar Hills 'B' analog: low initial water, 5-15%/yr\n"
         "decline, long life). The geologic upside is realized in the STEERING, not just the target pick.",
         fontsize=8.4, color="#333", linespacing=1.5, va="top")

fig.text(0.06, 0.165, "Calibration required (honest scope)", fontsize=9, weight="bold", color="#777")
fig.text(0.07, 0.138,
         "Depths, tops and the OWC above are a PLAY-MODEL SCHEMATIC (public Cedar Hills Red River 'B' character + this well's\n"
         "parameters). Before drilling, tie to: (1) the original Branch 4 wellbore logs (actual 'B' top, thickness, porosity);\n"
         "(2) Branch 5 tops + fluid contacts (the OWC datum); (3) a structural dip map. Target zone confidential to 12/19/2026.",
         fontsize=7.8, color="#777", linespacing=1.5, va="top")
footer(fig, 2); pdf.savefig(fig); plt.close(fig)

pdf.close()
print("wrote TOSCO_4RR_Geologic_Window_Report.pdf (2 pages)")
