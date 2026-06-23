# Valor Energy Partners — Prospect Value Screen (Public Data)

*Re: Prospects Purchase Agreement, eff. 8 Dec 2025 (Hood / Bay, Valor — Crawford / McAndrew / Crump,
sellers). Method: the value-geometry discipline from `FINDINGS.md` applied with the correct value
dimension per play, grounded only in public KGS records (`lease_prod.csv`, `ks_tops.txt`). Scripts:
`recon_valor.py`, `valor_prospects.py`. **This is a screen, not a valuation** — see "What this can and
cannot tell you."*

## The deal at a glance
Valor acquired the sellers' maps / seismic / geology (their IP) on four prospects, with leasing at
Valor's sole discretion (nominate which to pursue). Economics are **burdened by seller ORRIs**:
**Kansas 7.5%** (Crawford 4.5 + McAndrew 1.5 + Crump 1.5), **Ohio 5.5%** (4.0 + 0.75 + 0.75), plus
$2,500/well rig move-in after the first well in each prospect (first well free). After ORRI +
customary landowner royalty, Valor's NRI is ~73–78% — **the prospects must be good enough to carry
that load.**

| Prospect | Play | Where | Public-data verdict |
|---|---|---|---|
| Old Paint / Algrim | **Morrow sandstone** | Finney+Gray Co, KS | Productive area, **open running room in Gray**; sand fairway not publicly mappable |
| Butler Arbuckle | **Arbuckle carbonate** | Butler Co, KS | Mature area; footprint structurally **low**; structural edge real but underpowered |
| Harpers Station | **Trenton carbonate** | Ross Co, **Ohio** | **Outside KGS data** — needs ODNR pull |

## Method note — right dimension per play
Our verified edge (structure → productivity) is a *carbonate, structural-trap* result. It transfers
to **Butler Arbuckle** (same rock, same trap style) — **not** to the Morrow (a **sandstone** play
whose value is sand presence/thickness in incised-valley fairways) and not to Ohio Trenton (fractured
dolomite, different control, different state's data). We apply the *method* to all; we claim the
*structural edge* only where the rock matches.

## Prospect 1–2 — Old Paint / Algrim (Morrow, Finney + Gray)
- **Footprint (T23–24S, R29–30W):** 53 oil leases / 92 wells; median best12 **4,153 bbl**, median
  cum **~20,000 bbl** — solid, and an **oil-only floor**: SW-Kansas Morrow is gas-prone and the KGS
  *oil* record barely registers it — only **5 Finney oil leases carry a Morrow zone tag** (and even
  that tag is the combined "Mississippian, Morrow"), against **2,995 Morrow tops** mapped across the
  Finney/Gray area.
- **Running room:** Finney is developed (874 wells); **Gray is open (90 wells)** — the Gray acreage
  is the under-drilled upside. Multi-pay column present (Morrow + Mississippian + Lansing tops).
- **Limit:** the *value driver* — Morrow channel-sand fairways — is **not resolvable from public
  data** (20 wells with Morrow thickness + oil; thickness→oil is noise, rho −0.32 n=20). This is
  exactly what the sellers' seismic is for. Public data confirms the *area* is productive and partly
  open; it cannot rank the *sand sweet spots*.
- **Opportunity:** highest running room of the KS prospects, gas upside not in these numbers, proven
  oil. **Risk:** depends on the sellers' sand maps; gas economics need a separate KGS gas pull.

## Prospect 3 — Butler County Arbuckle (Arbuckle, Butler)
- **Play behaves as expected:** Arbuckle structure → log best12 rho **+0.30** (n=31, p=0.10) — same
  sign/size as our CKU edge (+0.41 shelf), but **underpowered/not significant** on this small subset.
  Mississippian structure here is null (+0.06) — the edge is specifically on the deep Arbuckle, as in
  our work.
- **Footprint (T28S, R5–6E):** 53 leases / **140 wells — mature**; median best12 2,082 bbl. The
  acquired sections sit at the **26th structural percentile** of the county (structurally *low*). If
  structure pays here, the footprint is on the **weaker** side of the trend on public data.
- **Caveat (important):** n is small and the public Arbuckle tops are sparse; the sellers' seismic may
  define a specific Arbuckle structural/stratigraphic trap (nose, dolomitized zone) the public data
  can't see. Treat the "structurally low" flag as **a question to put to the seller's data**, not a
  disqualification.
- **Opportunity:** proven Arbuckle trend, fast first well (free move-in). **Risk:** mature/competitive
  acreage + below-median public structural position → the seller's structural case must carry it.

## Prospect 4 — Harpers Station (Trenton, Ross Co, Ohio)
- **Outside KGS data** — KGS is Kansas-only; this extract contains **no Ohio production or tops** at
  all, so Harpers Station cannot be screened here. Trenton here is a Cincinnati-Arch /
  Lima-Indiana-style fractured-dolomite play; value is controlled by fracture/dolomite fairways, not
  simple structure. Exhibit A defines an **ORRI boundary** (this prospect may be an override position
  rather than a working-interest play — different economics).
- **Next step:** replicate the method on **Ohio DNR (ODNR) public well/production data** — the same
  wringer + OOS-verify discipline, with the Trenton-appropriate dimension. Not doable from KGS.

## Leasing priority (the actionable call)
Leasing is at Valor's discretion. On **public screening alone** (seller seismic will reorder this):
1. **Gray-County Morrow (Old Paint/Algrim, Gray portion)** — open acreage, productive trend, gas
   upside, multi-pay. Best public risk/reward; nominate first.
2. **Finney-County Morrow** — proven but developed; infill / bypassed-pay; pursue where seller maps
   show untested channel sand.
3. **Butler Arbuckle** — proven play but mature + footprint structurally low on public data; **gate
   on the seller's seismic** before committing lease capital.
4. **Harpers Station (Ohio)** — cannot screen here; run the ODNR analysis before ranking; ORRI nature
   may make it a low-cost option position regardless.

## What this can and cannot tell you
- **Can:** confirm each area is productive, measure development density (open vs drilled), locate the
  footprint's structural position, and confirm the play-type controls — all from free public records.
- **Cannot:** rank the seller's seismic-defined sweet spots, value the gas (oil-only data), or assess
  Ohio. The sellers' IP is the valuation basis; this screen tells you **where it must be strongest to
  justify the 7.5% ORRI load**, and which acreage has public-confirmed running room.

## Recommended next public pulls
1. **KGS gas production** (Morrow is gas) — turns the Morrow oil *floor* into a real value.
2. **ODNR Ohio** well/production for Ross Co — to screen Harpers Station at all.
3. **Exact PLSS footprints** (county GIS / KGS section centroids) — replace the approximate lat/lon
   boxes used here with the precise acquired sections.
