#!/usr/bin/env python
"""Bundle the seven Tosco Branch 4 RR deliverables into one client hand-off PDF,
with a branded Contents page (front matter) listing each document and its page."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import fitz

NAVY, ACC, GREEN, AMBER, GREY = "#15293f", "#c0392b", "#1e7d34", "#9a6a2f", "#5a6470"

# (file, title, one-line description) in hand-off order
DOCS = [
 ("TOSCO_Executive_Cover.pdf", "Executive Summary",
  "One-page distilled view — probability of a producer, dominant risk, recommended action."),
 ("TOSCO_Wells_Execution_Report.pdf", "Production-Probability & Execution Report",
  "CoS triangulation, risk-retirement dynamics, sensitivity, cross-domain V&V, full citations."),
 ("TOSCO_Production_Expectation_Matrix.pdf", "Production Expectation Matrix",
  "Well / completion strategies scored on the execution framework; why fracking is wrong for the 'B'."),
 ("TOSCO_4RR_Geologic_Window_Report.pdf", "Geologic / Geosteering Window Report",
  "Stratigraphic pay window, geosteering cross-section, MWD-gamma steering playbook."),
 ("TOSCO_Drilling_Execution_Forecast.pdf", "Drilling Execution Forecast",
  "Geometry × geology section-by-section; risk-by-phase (mechanical / geological / fluid)."),
 ("TOSCO_Fluid_Balance_Risk.pdf", "Fluid Balance — Loss & Gain Risk",
  "Mud / ECD vs formation pressure; the narrow window in the pay; loss & gain mitigation toolkit."),
 ("TOSCO_Steering_Anticipation.pdf", "Steering Anticipation at Scale",
  "Anticipatory geosteering with a fleet-learned dip prior; ~95% footage held in-zone."),
]
OUT = "TOSCO_Branch4RR_Client_Package.pdf"
CONTENTS = "TOSCO_Package_Contents.pdf"

# ---- page-reference map (Contents is page 1; docs follow) -------------------
pages, cur = [], 2                                   # first doc starts on page 2
for f, _, _ in DOCS:
    pages.append(cur)
    cur += fitz.open(f).page_count
total = cur - 1

# ============================ Contents page (portrait) ============================
plt.rcParams.update({"font.size": 9.5, "savefig.dpi": 200})
fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")

hb = fig.add_axes([0, 0.93, 1, 0.07]); hb.axis("off")
hb.add_patch(Rectangle((0, 0), 1, 1, fc=NAVY, ec="none"))
hb.text(0.06, 0.60, "VALOR ENERGY PARTNERS", color="white", fontsize=13, weight="bold")
hb.text(0.06, 0.24, "Subsurface & Drilling Analytics", color="#9fb3c8", fontsize=8.0)
hb.text(0.94, 0.58, "CONFIDENTIAL", color="white", fontsize=8.5, weight="bold", ha="right")
hb.text(0.94, 0.26, "For Client Review — 23 Jun 2026", color="#9fb3c8", fontsize=7.5, ha="right")
fig.add_artist(plt.Line2D([0, 1], [0.928, 0.928], color=ACC, lw=2.2))

fig.text(0.06, 0.875, "Client Package — Contents", fontsize=22, weight="bold", color=NAVY)
fig.text(0.06, 0.845, "Tosco Branch Re-Entry Wells — Production Potential, Geology & Drilling Execution",
         fontsize=12, color=ACC)
fig.text(0.06, 0.823, "Bowman County, North Dakota  ·  Williston Basin southern margin (Red River trend)",
         fontsize=8.5, color=GREY)
fig.text(0.06, 0.800, "Prepared for:  Marlo Operating Company · Valor Energy Partners      |      "
         "Wells:  Branch 4 RR (#41700) · Branch 2 (#41258)", fontsize=8.0, color="#333")
fig.add_artist(plt.Line2D([0.06, 0.94], [0.788, 0.788], color="#cfd6de", lw=1.0))

# entries
y0, dy = 0.745, 0.092
for i, ((_, title, desc), pg) in enumerate(zip(DOCS, pages)):
    y = y0 - i * dy
    bx = fig.add_axes([0.06, y - 0.018, 0.045, 0.045]); bx.axis("off")
    bx.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=NAVY, ec="none"))
    bx.text(0.5, 0.5, str(i + 1), ha="center", va="center", color="white", fontsize=13, weight="bold")
    fig.text(0.135, y + 0.012, title, fontsize=12.5, weight="bold", color=NAVY)
    fig.text(0.135, y - 0.014, desc, fontsize=8.3, color=GREY)
    fig.text(0.94, y + 0.012, f"p. {pg}", fontsize=10.5, weight="bold", color=ACC, ha="right")
    if i < len(DOCS) - 1:
        fig.add_artist(plt.Line2D([0.135, 0.94], [y - 0.034, y - 0.034], color="#e6e9ee", lw=0.8))

# how-to-read note
axn = fig.add_axes([0.06, 0.075, 0.88, 0.10]); axn.axis("off")
axn.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.015,rounding_size=0.04",
                             fc="#fbf3e6", ec=AMBER, lw=1.3))
axn.text(0.022, 0.80, "HOW TO READ THIS PACKAGE", color=AMBER, fontsize=9, weight="bold", va="top")
axn.text(0.022, 0.55,
         "Start with the Executive Summary (1) for the bottom line. Documents 2–3 answer “will it produce and "
         "with which completion”;\ndocument 4 answers “where to drill”; documents 5–7 answer “how the well will "
         "drill” — mechanics, fluid balance, and steering.\nScope is the probability of a producing well and how "
         "to execute it — not economics.", color="#3d2f12", fontsize=8.0, va="top", linespacing=1.45)

fig.add_artist(plt.Line2D([0.06, 0.94], [0.055, 0.055], color="#cfd6de", lw=1.0))
fig.text(0.06, 0.038, "CONFIDENTIAL — Marlo Operating / Valor Energy Partners.  Target formation NDIC-confidential "
         "to 12/19/2026; figures calibrate to actual wellbore logs.", fontsize=6.8, color=GREY)
fig.text(0.94, 0.038, f"{total} pages", fontsize=6.8, color=GREY, ha="right")
fig.savefig(CONTENTS); plt.close(fig)
print(f"wrote {CONTENTS}")

# ============================ merge ============================
out = fitz.open()
out.insert_pdf(fitz.open(CONTENTS))
for f, _, _ in DOCS:
    out.insert_pdf(fitz.open(f))
# bookmarks / outline for navigation
toc = [[1, "Contents", 1]] + [[1, title, pg] for (_, title, _), pg in zip(DOCS, pages)]
out.set_toc(toc)
out.set_metadata({"title": "Tosco Branch 4 RR — Client Package",
                  "author": "Valor Energy Partners — Subsurface & Drilling Analytics",
                  "subject": "Production potential, geology & drilling execution (Bowman County, ND)"})
out.save(OUT, garbage=4, deflate=True)
print(f"wrote {OUT}  ({out.page_count} pages)")
out.close()
