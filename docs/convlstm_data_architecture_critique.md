# ConvLSTM Wildfire Data Architecture — Critical Review

**Role of this document:** a skeptical-collaborator review of the proposed ConvLSTM input architecture, not an implementation guide. Where a claim is a citable fact (dataset resolution, coverage, licence), a source is given. Where a claim is a methodological judgment (which representation is "best," whether ConvLSTM is the right model), it is labeled as a recommendation, not an established fact, and the confidence level is stated.

---

## 1. Executive verdict

The overall pipeline shape (static + dynamic rasters → common grid → spatiotemporal model → future fire state) is **standard practice** and matches two real, peer-reviewed precedents: Google's *Next Day Wildfire Spread* dataset (Huot et al., 2022) and *WildfireSpreadTS* (NeurIPS 2023). So the architecture is not naive — it is the right shape.

But three specific choices as originally stated are **not defensible as written** and need to change before any data is downloaded:

1. **"Fuel density" as an invented scalar** (`forest=1.0, grass=0.3, shrub=0.6`) is scientifically unjustified. Canada already has an authoritative categorical fuel classification (CFFDRS FBP System, 16 fuel types) and a real continuous biomass/canopy dataset (SCANFI). Use those instead of inventing a number.
2. **Categorical vegetation labels as raw integers** (`forest=1, grass=2, shrub=3`) are invalid CNN input — they impose a false ordinal relationship (shrub is not "between" forest and grass). This must be one-hot encoded or embedded, not fed as a scalar.
3. **`wind_speed + wind_direction` as two raw channels** is a modeling mistake: direction is circular (0° and 360° are identical points but numerically maximally distant), and a CNN has no reason to learn that discontinuity from data. Decompose to `u, v` components — this is the meteorological-standard representation and it is what ERA5 and every operational weather product actually stores natively.

None of these are optional style preferences — they are standard-practice fixes documented in both the reanalysis data format itself (ERA5 stores `u10`/`v10`, not speed/direction) and the CFFDRS literature (fuel type, not an ad hoc density scalar, is the operational fuel input for Canadian fire behaviour prediction).

The fire-state representation (`0=unburned, 1=burning, 2=burned`) is defensible as a *label* but is too poor as an *input* — it collapses fire history into a single categorical snapshot with no memory of ignition timing or growth trend, which is exactly the information a spread predictor needs from its own past. Section 9 gives a stronger, still-simple alternative.

Overall confidence: **the concept is sound, the initial channel design is not ready for implementation.**

---

## 2. What was originally proposed

```
Static:  elevation, slope, vegetation/land-cover type, fuel density
Dynamic: fire state (0/1/2), wind speed, wind direction, temperature, relative humidity
Grid:    common raster, X[t, C, H, W]
```

## 3. What is scientifically correct

- **Elevation and slope matter physically** and are standard terrain inputs in every operational fire-behaviour system (CFFDRS FBP, LANDFIRE/FARSITE). Confirmed as correct.
- **Wind and temperature/humidity as dynamic drivers** is correct — CFFDRS FWI, LANDFIRE/FARSITE, and every ML wildfire-spread paper reviewed (Next Day Wildfire Spread, WildfireSpreadTS) use these as core dynamic channels.
- **A common-grid rasterization approach with `X[t, C, H, W]`** is exactly the tensor shape used by the two closest public benchmarks (see §6), so the framing is right.
- **Static vs. dynamic separation** is correct in principle — terrain doesn't change at wildfire timescales; fuel condition, weather, and fire state do.

## 4. What is questionable or wrong

| Item | Problem | Fix |
|---|---|---|
| Fuel density as invented scalar | No physical basis, not reproducible, not defensible in a review | CFFDRS FBP fuel type (categorical) + SCANFI biomass/canopy (continuous) — §11 |
| Vegetation as raw integer label | False ordinal structure fed to a CNN | One-hot or embedding — §11 |
| Wind speed + direction as 2 raw channels | Direction is circular; discontinuity at 0°/360° | u, v components — §10 |
| Fire state as single categorical snapshot only | No memory of ignition timing; weak signal for spread *rate* | Add binary active-fire + cumulative-burned + time-since-ignition channels — §9 |
| Relative humidity alone as the moisture proxy | RH is a weaker predictor of dead fuel moisture than vapor pressure deficit (VPD) | Add VPD explicitly (§5) |
| No stated resolution or temporal cadence | Grid resolution and timestep were never pinned down against what the underlying data actually supports | §7, §8 |
| No stated leakage-prevention procedure | Sliding-window sequences from the same event risk event leakage if split naively | §14, §18 |

---

## 5. Feature-by-feature validation

For each feature: physical relevance, established use, real observability, static/dynamic classification, leakage risk, and a better representation where relevant.

**Elevation** — Physically relevant (affects fuel type distribution, wind channeling, drainage of cold air, and fireline intensity via updraft). Standard in FARSITE, CFFDRS, and every ML wildfire paper reviewed. Observable nationally via CDEM (NRCan) or Copernicus DEM. Static. No leakage risk (elevation doesn't change). Representation: raw elevation in metres, normalized; fine as one channel.

**Slope** — Physically critical: fire spreads faster upslope (radiative/convective preheating of upslope fuels) — this is one of the strongest terms in the Rothermel spread-rate model and in CFFDRS FBP. Standard. Not directly "observed" — it is *derived* from the DEM (a terrain-analysis operation, not a separate dataset download). Static. No leakage. Representation: slope magnitude in degrees; consider also **aspect** (slope direction, as sin/cos to avoid the same circularity problem as wind direction) since south-facing boreal slopes in the northern hemisphere receive more solar loading and dry out faster — this is a legitimate secondary feature, not mandatory for a first prototype.

**Vegetation / land-cover type** — Physically relevant only as a proxy for fuel *type*, which is what actually controls spread rate and fire behaviour (Rothermel model parameters are fuel-type-specific, not land-cover-specific). Generic land-cover classes (ESA WorldCover: "tree cover," "grassland," "shrubland," etc.) are coarser than what fire behaviour prediction actually needs. Observable nationally at 10 m (WorldCover) or 30 m (SCANFI, CFFDRS FBP fuel layer). Static (updated infrequently — annually at best; treat as static within a fire season). Leakage risk: **low but non-zero** — if a land-cover product used post-fire burn scars in its own training/labeling (some products are annual composites), a "vegetation" value assigned *after* the fire event could implicitly encode that the area burned. Mitigation: use the land-cover/fuel layer's vintage from *before* the fire season being modeled, never the same-year product if it postdates the fire. Representation: **do not use raw land-cover class** — use the CFFDRS FBP fuel type directly (§11), because it is already fire-behaviour-specific rather than a generic land-cover proxy requiring a second translation step.

**Fuel density** — This is the most misspecified feature in the original proposal (see §11 for full treatment). "Density" is not a standard fire-science term on its own; the field uses **fuel load** (mass per unit area, e.g. t/ha or kg/m²) or **fuel type classification** (which implicitly encodes typical load, structure, and moisture-of-extinction). Physically relevant — fuel load determines fire intensity and residence time. Observable via SCANFI biomass (continuous, R² ≈ 0.76 validated against forest inventory plots) or CFFDRS FBP fuel type (categorical, operationally used in every Canadian fire-behaviour calculation). Static within a season (biomass changes over years, not days). No meaningful leakage risk if using pre-fire-season data. Representation: SCANFI aboveground biomass (continuous) **and** CFFDRS FBP fuel type (categorical) as complementary channels — not a single invented scalar.

**Temperature** — Physically relevant: higher air temperature reduces fuel moisture and is a direct FWI System input. Standard. Observable via ERA5 (global, hourly, ~31 km) or ECCC HRDPS (Canada, ~2.5 km, 4×/day, 48 h forecast horizon). Dynamic. Leakage risk: **real and specific** — using *forecast* temperature for a time window that includes timestamps after the prediction target would leak future information; must use only analysis/reanalysis (or historical forecast valid-at-time) data for training, never a forecast issued after the target time (§9, §18).

**Relative humidity** — Physically relevant as a moisture-related driver, and it is the term CFFDRS FWI actually uses. But the literature is fairly clear that **vapor pressure deficit (VPD) is a better predictor of dead fine fuel moisture than RH or temperature alone** (Sedano & Randerson-style analyses; VPD-based dead-fuel-moisture models were found to outperform RH/T models in the dataset reviewed — see citation below). Recommendation: keep RH for FWI-compatibility, but **add VPD as an explicit derived channel** rather than relying on the network to reconstruct it implicitly from T and RH. This is cheap (VPD is a closed-form function of T and RH/dewpoint) and has direct literature support.

**Wind speed / wind direction** — Physically the single most important dynamic driver of fire *direction and rate* of spread (wind-driven convective preheating dominates over slope in most fire regimes). Standard in every model reviewed. Observable via ERA5 u10/v10 globally or HRDPS regionally over Canada. Dynamic. The representation is wrong as stated — see §10.

**Fire state** — Physically it's the model's own memory of "where is the fire and how has it been behaving," which is what lets a spread model extrapolate a *rate*, not just a static shape. The proposed 3-class snapshot is directionally correct but under-informative — see §9 for the recommended multi-channel replacement. Observable via VIIRS/MODIS/GOES active-fire and burned-area products, each with real resolution/latency tradeoffs (§6). Dynamic. **This is the single highest-leakage-risk feature** — see §9 and §18 for exactly how to construct it without leaking future fire state into the input window.

---

## 6. Real datasets, with official sources

Every entry below was checked against an official/authoritative source found via web search on 2026-08-11. Where a claim could not be independently confirmed beyond a search snippet, that is stated explicitly.

### Elevation

**Canadian Digital Elevation Model (CDEM)** — NRCan. Full Canadian landmass coverage. Native resolution 0.75 arc-second (N–S) to 3 arc-seconds (E–W, varies by latitude) — roughly 20–70 m depending on latitude. Format: GeoTIFF. Open Government Licence – Canada (free). Appropriate for research; realistically alignable to a common grid via standard GDAL reprojection/resampling.
[CDEM product specification (NRCan)](https://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/doc/CDEM_product_specs.pdf) · [Open Government Portal listing](https://open.canada.ca/data/en/dataset/7f245e4d-76c2-4caa-951a-45d1d2051333)

**Copernicus DEM (GLO-30 / GLO-90)** — ESA/Copernicus. Global coverage including Canada. GLO-30 = 30 m, GLO-90 = 90 m. GLO-90 fully free worldwide; GLO-30 free for most tiles via the "Public" instance, with a restricted-access "R" instance for full coverage under a separate authorised-user licence. Format: COG/GeoTIFF. Recommended as the **primary** elevation source over CDEM for a first prototype because it has a single consistent global processing pipeline (useful if the project later needs non-Canadian data) and unambiguous free licensing at 30 m.
[Copernicus Data Space Ecosystem — DEM collections](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) · [Licence terms, COP-DEM-GLO-30](https://docs.sentinel-hub.com/api/latest/static/files/data/dem/resources/license/License-COPDEM-30.pdf)

### Land cover / vegetation

**ESA WorldCover** — ESA. Global, including Canada. 10 m resolution, 11 land-cover classes, versions for 2020 and 2021 (overall accuracy 74.4% / 76.7% respectively). Free, open license (CC BY 4.0). Format: COG. Good for a generic land-cover backdrop, but see §11 — not the primary fuel-relevant layer for this project.
[ESA WorldCover official site](https://esa-worldcover.org/en) · [Registry of Open Data on AWS](https://registry.opendata.aws/esa-worldcover-vito/)

**SCANFI (Spatialized Canadian National Forest Inventory)** — NRCan / Canadian Forest Service. Canada-wide (non-Arctic landmass). 30 m resolution. Provides land-cover type, canopy height, crown closure, aboveground biomass, and species composition, validated against NFI photo-plots (biomass R²≈0.76, crown closure R²≈0.82, height R²≈0.78). v2 provides a 5-year interval national time series 1985–2025. Free, Open Government Licence – Canada. **This is the recommended primary vegetation/fuel-load dataset** for this project (§11) — it's Canada-specific, validated, and gives continuous biomass rather than a coarse category.
[Peer-reviewed methods paper (Can. J. Forest Res.)](https://cdnsciencepub.com/doi/10.1139/cjfr-2023-0118) · [Open Government Portal — SCANFI v2](https://open.canada.ca/data/en/dataset/07653869-f303-46c2-a04e-9ab479b73cbf)

### Fuel type

**CFFDRS FBP Fuel Types (2024), 30 m** — NRCan / Canadian Forest Service. Canada-wide. 30 m resolution. 16 operational fuel types (e.g. C-1…C-7 conifer, D-1/2 deciduous, M-1…M-4 mixedwood, S-1…S-3 slash, O-1 grass) derived from SCANFI, ecozones, and NBAC. This **is** the Canadian operational standard for fire-behaviour fuel classification — it is not a research convenience layer, it's what Canadian fire agencies actually use to run FBP calculations. Free, Open Government Licence – Canada. Format: GeoTIFF/vector.
[Open Government Portal listing](https://open.canada.ca/data/en/dataset/4e66dd2f-5cd0-42fd-b82c-a430044b31de) · [NRCan — Canada's Fire Behaviour Prediction System fuel types](https://natural-resources.canada.ca/forests-forestry/wildland-fires/canada-fire-behaviour-prediction-system-fuel-types)

**LANDFIRE FBFM13 / FBFM40** — USGS/USFS. US coverage only (**does not cover Canada**). 30 m native resolution. Mentioned here only because it's the closest US analogue and useful if the project ever needs a US-side benchmark comparison; not usable directly for a Canadian study area.
[LANDFIRE FBFM40](https://landfire.gov/fuel/fbfm40) · [LANDFIRE FBFM13](https://www.landfire.gov/fuel/fbfm13)

### Weather

**ERA5 (hourly, single levels)** — ECMWF / Copernicus Climate Change Service. Global, including Canada. ~31 km (0.25°) native grid, hourly, 1940–present, updated with a few days' lag. Free, open access (Copernicus Climate Data Store, registration required, no cost). Provides `u10`, `v10` (10 m wind components — already u/v, not speed/direction), 2 m temperature, dewpoint (for RH/VPD derivation), precipitation. Coarse relative to a fire-front's actual size, but the only option with full historical hourly reanalysis coverage back decades, which matters for building enough training sequences.
[ECMWF ERA5 dataset page](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) · [Climate Data Store — ERA5 hourly single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)

**HRDPS (High Resolution Deterministic Prediction System)** — Environment and Climate Change Canada. Covers "most of Canada" (not full Arctic/full national extent — check the specific domain polygon before assuming full coverage). ~2.5 km resolution, run 4×/day, forecasts to 48 h. This is a **forecast** product, not a reanalysis — for training data you must use the archived forecast *valid at* or *before* the target time, never a forecast issued after it (leakage risk, §18). Free via ECCC MSC Open Data / GeoMet, Open Government Licence – Canada.
[ECCC HRDPS readme](https://eccc-msc.github.io/open-data/msc-data/nwp_hrdps/readme_hrdps_en/) · [Open Government Portal listing](https://open.canada.ca/data/en/dataset/5b401fa0-6c29-57f0-b3d5-749f301d829d)

**Recommendation:** use ERA5 (or ERA5-Land, ~9 km, not independently verified here but documented by ECMWF) for historical training-sequence construction (consistent reanalysis, no forecast-leakage ambiguity), and treat HRDPS as a future upgrade path for finer-grained Canada-specific weather once the leakage-safe historical-forecast-archive question is solved.

### Fire observations

**NASA FIRMS / VIIRS active fire (375 m)** — NASA. Global, including Canada. 375 m pixel resolution, ~2 overpasses/day per satellite (S-NPP, NOAA-20, NOAA-21 combined give better revisit), refreshed multiple times daily in near-real-time. Free, public. Good balance of resolution and near-daily cadence; the standard choice in both Next Day Wildfire Spread and WildfireSpreadTS.
[NASA Earthdata — FIRMS](https://www.earthdata.nasa.gov/data/tools/firms) · [VIIRS 375 m NRT product, NASA Open Data Portal](https://data.nasa.gov/dataset/viirs-noaa-21-i-band-375-m-active-fire-product-nrt-vector-data-e73a2)

**MODIS MCD64A1 (burned area)** — NASA/USGS LP DAAC. Global. 500 m, **monthly** composite — too coarse temporally for next-day/next-hour spread prediction, but useful for post-hoc burned-area validation/ground truth of *final* fire extent, and for constructing the "cumulative burned" long-term static context.
[NASA Earthdata catalog page](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd64a1-061)

**GOES-16/17/18/19 ABI Fire Detection and Characterization (FDC)** — NOAA. Geostationary, so **temporal cadence is excellent (down to ~10 minutes for full disk)**, but two important caveats for a Canadian project: (1) spatial resolution is coarser than VIIRS (≈2 km at nadir, worse toward the edge of the disk), and (2) view-angle/parallax distortion **worsens significantly at high Canadian latitudes**, since Canada sits well off-nadir from both GOES-East and GOES-West — this is a real, documented limitation of geostationary fire products at northern latitudes, not just a theoretical caveat. Free, public (NOAA CLASS / AWS Open Data).
[GOES-R fire/hot-spot product page](https://www.goes-r.gov/products/baseline-fire-hot-spot.html)

**Canadian National Fire Database (NFDB)** — NRCan / CWFIS. Canada-wide, all provinces/territories + Parks Canada. Point and polygon fire records, updated annually, Open Government Licence – Canada. Explicitly documented by NRCan as **incomplete and variable in positional accuracy** across agencies — treat as ground-truth *ignition/perimeter* records for event selection, not as a pixel-accurate daily-progression product.
[CWFIS / NFDB](https://cwfis.cfs.nrcan.gc.ca/ha/nfdb) · [NFDB point shapefile metadata](https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point_shapefile_metadata.pdf)

**National Burned Area Composite (NBAC)** — NRCan / CWFIS / FireMARS. Canada-wide, annual, combines provincial perimeter data with satellite-derived mapping to produce the best-available annual burned-area polygons since 1972. Free, Open Government Licence – Canada. Good source for final/near-final fire perimeters, and for computing the CFFDRS fuel-type layer's own training inputs — **not** a within-event daily progression product either.
[CWFIS NBAC](https://cwfis.cfs.nrcan.gc.ca/ha/nfdb?type=nbac) · [FireMARS overview (NRCan)](https://natural-resources.canada.ca/forests-forestry/wildland-fires/fire-monitoring-accounting-reporting-system)

**Canadian gridded Fire Weather Index (FWI)** — CWFIS interpolates ~2,500 weather stations (IDW, 12 nearest stations) to produce daily FWI System component grids; a newer automated **ERA5-FWI-SN** dataset provides a fully gridded FWI product at ~31 km, 1950–present, updated daily with a ~6-day lag. Free, NRCan/Copernicus-derived. Useful as an optional physics-informed derived feature (§17), not a primary raw-weather substitute.
[CWFIS FWI methodology](https://cwfis.cfs.nrcan.gc.ca/index.php/background/dsm/fwi) · [ESSD preprint, gridded FWI dataset](https://essd.copernicus.org/preprints/essd-2025-535/)

### Closest existing public benchmarks (precedent, not something to redownload wholesale)

**Next Day Wildfire Spread** (Huot, Hu, Ihme, Wang; arXiv:2112.02447, published as a Google Research dataset) — US-only, 18,545 fire-day samples, 64×64 pixels at 1 km, 12 input channels (topography, vegetation index, weather, drought index, population density, previous fire mask), next-day fire mask as target. This is the closest existing implementation of exactly the architecture originally proposed, and it validates the "static + dynamic → common grid → next-day mask" shape.
[arXiv:2112.02447](https://arxiv.org/abs/2112.02447) · [Google Research GitHub](https://github.com/google-research/google-research/blob/master/simulation_research/next_day_wildfire_spread/README.md)

**WildfireSpreadTS** (NeurIPS 2023 Datasets & Benchmarks) — US-only, 607 fire events, 13,607 daily images at 375 m, 23 channels (active fire, weather, topography, vegetation), rigorous 12-fold cross-validation designed specifically to prevent event leakage. This is the strongest available precedent for §14's event-level split requirement — it was built by people solving the exact leakage problem this project has, at a US scale, with US data.
[NeurIPS 2023 paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ebd545176bdaa9cd5d45954947bd74b7-Abstract-Datasets_and_Benchmarks.html)

**Canada-specific precedent found but only partially verifiable:** a 2026 arXiv preprint, *"Spatio-Temporal Wildfire Spread Prediction in Canada using a Video Swin-Hybrid-U-Net and Satellite Imagery"* (arXiv:2606.20693), reports training on Canadian wildfire events 2014–2023 using exclusively public Google Earth Engine data sources, 3-day input sequences, next-day fire-incidence-map output. I could not independently confirm its exact channel list or resolution beyond the abstract (the fetch returned only metadata, not full text) — flagging this explicitly rather than guessing at numbers. It is worth reading in full before finalizing the channel list, since it is (as far as this search found) the only public Canada-specific ConvLSTM/spatiotemporal deep-learning wildfire paper. A second Canada-focused preprint, *BCWildfire* (arXiv:2511.17597, boreal wildfire risk dataset), was found but its full content also could not be extracted from the PDF fetch — same caveat applies.

---

## 7. Spatial resolution recommendation

| Resolution | Fire-front fidelity | DEM support | Land-cover/fuel support | Weather support | Fire-obs support | Compute cost |
|---|---|---|---|---|---|---|
| 10 m | Excellent | Yes (resample from 30m DEM — not native) | Yes (WorldCover native) | No (weather is 100–1000× coarser) | No (no fire product this fine at useful revisit) | Very high (H×W explodes) |
| 30 m | Good | Yes (native, CDEM/Copernicus) | Yes (native, SCANFI/FBP fuel/LANDFIRE) | No | No (VIIRS is 375 m, best case) | High |
| 100–250 m | Adequate for regional fronts | Downsampled from 30m | Downsampled from 30m | Still coarser than this | MODIS burned area (500m) close, VIIRS finer | Moderate |
| 375 m–1 km | Coarse fire-front, but matches actual sensor limits | Downsampled | Downsampled | ERA5≈31km still much coarser; HRDPS≈2.5km closer | **Matches VIIRS (375m) and both benchmark datasets (375m, 1km)** | Low–moderate |

The honest constraint is this: **fuel and terrain data support 30 m, but the fire *observations* that become your training labels do not.** VIIRS active fire is 375 m at best; MODIS burned area is 500 m monthly; GOES FDC is ~2 km. There is no publicly available Canada-wide fire-observation product with real daily temporal cadence at better than ~375 m. Choosing 30 m or 10 m for the grid would create a false sense of precision — you'd be upsampling a 375 m fire-observation pixel into sixteen or more phantom 30 m cells with no actual independent information in them, and a ConvLSTM would happily learn spurious sub-pixel patterns that are actually resampling artifacts, not real fire-front structure.

**Recommendation: 375 m for a first research prototype**, matching VIIRS native resolution and both public benchmark datasets (WildfireSpreadTS uses exactly 375 m; Next Day Wildfire Spread uses 1 km — 375 m sits at the finer, defensible end of established practice, not an untested compromise). Resample the 30 m fuel/terrain layers *down* to 375 m (aggregation, e.g. modal fuel type or mean elevation per 375 m cell) rather than trying to fake 30 m fire observations. This keeps every channel "real" — max spatial fidelity is the constraint of the sparsest input (fire observation), not a resolution chosen for aesthetics.

## 8. Temporal resolution recommendation

| Cadence | Fire obs support | Weather support | Realistic? |
|---|---|---|---|
| 5–15 min | GOES FDC only, with coarse resolution + high-latitude distortion | HRDPS is 4×/day, not sub-hourly | Not realistic for Canada as primary cadence |
| 1 hour | No polar-orbiting product gives hourly revisit; only GOES (with above caveats) | ERA5 is hourly — well supported | Feasible only if GOES is the fire-obs backbone, which forces its resolution/latitude weaknesses onto the whole model |
| 3–6 hours | Still no polar product at this cadence | ERA5/HRDPS both support this | Feasible but arbitrary — no dataset naturally aligns here |
| 24 hours | **VIIRS gives near-daily (day+night composite) revisit; matches both benchmark datasets** | ERA5/HRDPS both trivially support daily aggregation | **Realistic and precedented** |

**Recommendation: 24-hour (daily) timesteps** for the first prototype, exactly matching both public benchmarks (Next Day Wildfire Spread and WildfireSpreadTS are both daily). This is not a compute-driven compromise — it is what the fire-observation data actually supports at usable spatial resolution over Canada. A ConvLSTM step should represent **one calendar day**, with the input sequence being the preceding N days (a reasonable starting point, matching the Canada-specific 2026 preprint found above, is a 3-day lookback window) and the target being the next day's fire state.

Sub-daily prediction is a legitimate *future* research direction (useful operationally — a UAV suppression system genuinely wants better than 24 h latency) but should be pursued only after the daily model is validated, using GOES with its stated caveats made explicit rather than silently ignored.

---

## 9. Fire representation

The proposed 3-class snapshot (`unburned / burning / burned`) is a **reasonable label for the prediction target** but a **weak representation as a model input**, because it discards two things a spread model needs: how *fast* the fire has been moving (rate information, implicit in *change* between consecutive snapshots) and how *long* a cell has been burning (residual heat/intensity, relevant to whether adjacent unburned fuel will actually ignite).

Comparison of alternatives:
- **Binary burned/unburned** — simpler than the 3-class version, but throws away "currently actively burning vs. already burned out," which matters (a burned-out cell won't re-ignite the same way an actively-burning cell threatens neighbors). Worse than the original 3-class proposal.
- **Active-fire probability** (from VIIRS/GOES confidence values, rather than a hard threshold) — better than a hard binary/categorical mask because it preserves detection uncertainty instead of silently discarding it at an arbitrary threshold. Recommended as a refinement, not mandatory for v1.
- **Fire intensity / fireline intensity (FRP — fire radiative power)** — physically the most informative single fire-state quantity (available from VIIRS/MODIS/GOES as a per-pixel product), but noisier and prone to missed detections/pixel saturation. Worth adding as an auxiliary channel once the baseline works; risky as sole representation for v1 because of missing-data patterns.
- **Rate of spread** — you cannot observe this directly; it would have to be *derived* from consecutive burned-area snapshots, which means it's a function of exactly the inputs the model already has (redundant, not leaking, but also not new information unless explicitly engineered as a finite-difference feature).
- **Time since ignition** — cheap to compute from your own event's history (first day a cell was detected as burning), physically meaningful (older burn = less residual fuel/heat), and **not derivable by the network from a single categorical snapshot**, so it's a genuine information addition, not redundant engineering.
- **Cumulative burned area** — monotonic, low-noise (once a cell is marked burned, later noisy detections don't need to un-mark it), a good stabilizing complement to noisy same-day active-fire detections.

**Recommendation:** replace the single 3-class channel with **three channels**: (1) binary/probabilistic *currently active* fire mask, (2) monotonic *cumulative burned* mask, (3) *time-since-first-detection* (in days, 0 for never-burned cells). This is still simple (3 channels, no exotic data source — all three are derivable from the same VIIRS/GOES time series already needed for the 3-class version) but gives the model the rate and persistence information the single-snapshot version cannot express.

---

## 10. Weather representation

**Wind:** decompose to `u = speed × cos(direction), v = speed × sin(direction)` — do **not** keep speed and direction as two separate raw channels. This is not a stylistic preference; it's the representation ERA5 and every operational NWP product actually store natively (`u10`, `v10`), specifically because direction is circular and a raw degree value creates a false discontinuity at 0°/360° that a convolutional/recurrent network has no architectural reason to know is fake. This is standard, uncontroversial practice, not an aggressive recommendation.

**Altitude:** use **10 m wind** (the standard near-surface meteorological reference height, and what ERA5/HRDPS provide as `u10`/`v10`) rather than a higher-altitude level. Near-surface wind is what actually drives surface fire spread (the FBP System's wind input is a 10 m-equivalent open wind speed); higher-altitude wind is more relevant to smoke plume transport, not surface fire-front advance, and isn't what's needed here.

**Spatial interpolation:** yes, appropriate — weather grids (ERA5 ~31 km, HRDPS ~2.5 km) are coarser than the 375 m fire grid, so each fire-grid cell inherits its containing weather-grid cell's value via nearest-neighbor or bilinear resampling. This is standard downscaling-by-broadcast, not a scientifically risky step, as long as it's applied identically to train and test data and doesn't use any spatial information not available at prediction time.

**Temporal interpolation:** appropriate **only within already-available data** (e.g. interpolating HRDPS's 4×/day cadence to hourly), never interpolating *across* the train/target boundary using future timestamps. If your target timestep needs weather at a time not yet available in the analysis archive, that's a leakage risk, not an interpolation opportunity (§18).

**Realistic weather resolution:** ~31 km (ERA5) to ~2.5 km (HRDPS) — both far coarser than the 375 m fire grid. This is a real, unavoidable resolution mismatch; state it explicitly as a project limitation rather than implying the weather channels carry 375 m-equivalent information (they don't — you're broadcasting one coarse value across many fine cells).

**Additional variables — evaluated individually, not added by default:**
- **Precipitation** — include. Directly suppresses spread and drives short-term fuel moisture; both ERA5 and HRDPS provide it natively at no extra dataset cost.
- **Vapor pressure deficit (VPD)** — include, computed from temperature + humidity. Literature evidence found (§5, §6) that VPD outperforms RH/T alone as a dead-fuel-moisture predictor; cheap to derive, not a separate download.
- **Solar radiation** — do not include in v1. Plausible secondary effect (surface heating, fuel drying) but weaker and more redundant with temperature than VPD is; adds a dataset dependency for a second-order effect. Revisit only if v1 shows systematic underprediction on sunny/dry days.
- **Dew point** — include only as an intermediate used to compute VPD/RH, not as a separate model input channel (redundant with RH+temperature once VPD is derived).
- **Fuel moisture (dead)** — do not include a *direct measurement* (there isn't a usable gridded Canada-wide product of this granularity); instead let VPD proxy for it, as the cited literature explicitly supports. Revisit if a specific Canadian dead-fuel-moisture gridded product is found later.
- **Soil moisture** — do not include. Its physical link to *surface* fire spread rate (as opposed to drought/ignition-probability context on multi-week timescales) is weak at the daily fire-spread timescale this project targets; adding it risks a noisy, low-signal channel that mostly just correlates with season.

---

## 11. Fuel / vegetation representation

This is the section the user specifically asked to be challenged hardest, so the verdict is stated plainly: **the original `forest=1.0, grass=0.3, shrub=0.6` style scalar mapping should not be used.** It fails on every count — it's not derived from any measurement, it can't be cited, it imposes an arbitrary numeric ordering, and a real, better alternative already exists for exactly this geography.

**What "fuel density" should actually mean:** in wildland fire science, the closest real quantities are **fuel load** (mass of combustible material per unit area — what actually determines fire intensity and energy release) and **fuel type** (a categorical classification that implicitly encodes typical load, arrangement, moisture-of-extinction, and expected fire behaviour for that vegetation/fuel complex — this is what CFFDRS FBP and LANDFIRE both use operationally, not raw biomass alone, because *arrangement* matters as much as *quantity*: a windrow of dead branches burns very differently than the same mass of standing green timber).

**Recommended representation, two complementary channels:**
1. **CFFDRS FBP fuel type (16 classes), one-hot encoded (16 channels)** — this is Canada's actual operational fire-behaviour fuel classification, used by real fire agencies to run FBP calculations. It is *the* physically meaningful categorical fuel representation for this geography, not a proxy for one.
2. **SCANFI aboveground biomass (continuous, Mg/ha)** — a real, validated (R²≈0.76) continuous fuel-load proxy, giving within-fuel-type variation the categorical layer alone can't express (a C-2 stand at 40 Mg/ha vs. 80 Mg/ha behaves differently even though both are "C-2").

Vegetation-type categorical encoding, evaluated:
- **Raw integer label** — wrong, as stated above (false ordinal structure).
- **One-hot encoding** — correct default for a small, fixed, physically distinct class set like the 16 CFFDRS fuel types. Interpretable, no learned parameters needed for the encoding itself, and 16 extra channels is a modest, affordable cost. **Recommended for v1.**
- **Learned embedding** — reduces channel count (e.g. 16 classes → 4–8 dim embedding) and lets the network discover which fuel types behave similarly, but adds a learned component that needs its own regularization/interpretation, and with only 16 classes the dimensionality savings barely matter. Worth revisiting once the model needs to scale to a much larger fuel taxonomy (e.g. if later fusing LANDFIRE's 40-class FBFM40 for a cross-border extension); not needed for v1's 16-class Canadian fuel system.
- **Continuous vegetation properties alone (no categorical channel)** — insufficient on its own; canopy height/biomass/crown closure describe structure but not fuel *type* behaviour (e.g., a grass fuel type and a young conifer stand can have similar low biomass but completely different spread-rate behaviour). Use continuous properties as a *complement* to, not a replacement for, the categorical fuel type.
- **Multiple vegetation/fuel channels (the recommended approach)** — one-hot fuel type (16 ch) + SCANFI biomass (1 ch) + SCANFI canopy height (1 ch) + SCANFI crown closure (1 ch). This is the strongest defensible representation: physically grounded, Canada-specific, and each channel adds genuinely non-redundant information (type, load, vertical structure, canopy continuity).

---

## 12. Final tensor / channel specification

Two tensors: a static tensor broadcast across the sequence dimension, and a dynamic tensor that varies per timestep. Both share the same `H × W` grid at **375 m resolution** (§7).

```
X_static.shape  = [C_static, H, W]                       # constant across the whole sequence
X_dynamic.shape = [T_seq, C_dynamic, H, W]                # T_seq daily steps, e.g. 3-day lookback
Y_target.shape  = [C_target, H, W]                        # next-day fire state
```

**Static channels (C_static = 20):**
| # | Channel | Source |
|---|---|---|
| 0 | Elevation (m, normalized) | Copernicus DEM GLO-30 (aggregated to 375m) |
| 1 | Slope (degrees) | derived from DEM |
| 2 | Aspect — cos(aspect) | derived from DEM (optional for v1) |
| 3 | Aspect — sin(aspect) | derived from DEM (optional for v1) |
| 4–19 | CFFDRS FBP fuel type, one-hot (16 classes) | NRCan CFFDRS FBP Fuel Type layer, 30m→375m modal aggregation |

**Additional static channels (append after the 16 fuel-type channels):**
| # | Channel | Source |
|---|---|---|
| 20 | Aboveground biomass (Mg/ha, normalized) | SCANFI |
| 21 | Canopy height (m) | SCANFI |
| 22 | Crown closure (%) | SCANFI |

**Dynamic channels (C_dynamic = 10, per timestep):**
| # | Channel | Source |
|---|---|---|
| 0 | Active fire mask (binary or probability) | VIIRS 375m active fire |
| 1 | Cumulative burned mask (monotonic) | derived from VIIRS time series within event |
| 2 | Time since first detection (days, 0 if unburned) | derived |
| 3 | Wind u-component (m/s, 10m) | ERA5 `u10` (or HRDPS) |
| 4 | Wind v-component (m/s, 10m) | ERA5 `v10` (or HRDPS) |
| 5 | 2m air temperature (°C) | ERA5 `t2m` (or HRDPS) |
| 6 | Relative humidity (%) | derived from ERA5 `t2m`/`d2m` (or HRDPS) |
| 7 | Vapor pressure deficit (kPa) | derived from temperature + humidity |
| 8 | Precipitation (mm, 24h accumulated) | ERA5 (or HRDPS) |
| 9 | Day-of-year (sin or cos, seasonal signal) | calendar, cheap, no dataset needed |

**Target (`Y`):** next-day active fire mask + next-day cumulative burned mask (2 channels), mirroring dynamic channels 0–1 one day ahead — this keeps the target format identical to two of the input channels, which is convenient for both loss computation and for chaining multi-step rollout predictions later.

This is a **research-prototype specification**, not a final one — every number above (channel count, lookback length, resolution) should be revisited once actual data availability is confirmed event-by-event, particularly the Canada-specific weather archive question (§18).

---

## 13. Data preprocessing pipeline

1. **Download** — pull each raw dataset from its official source (§6) for the specific study-area bounding box and date range. Store raw files unmodified (never edit in place) so provenance is always recoverable.
2. **Coordinate-system normalization** — reproject every raster to a single common CRS. Recommendation: a Canada-appropriate equal-area or conformal projection (e.g. Lambert Conformal Conic, the standard for pan-Canadian analysis, or a UTM zone if the study area is regional) — NAD83(CSRS) is what CDEM itself uses, so it's a natural anchor if using CDEM.
3. **Spatial clipping** — clip every layer to the study-area bounding box (with a buffer, since fires spread and you don't want edge artifacts at the exact crop boundary — see §16, edge effects).
4. **Resampling/reprojection to 375 m** — categorical layers (fuel type) use **modal/nearest** resampling, never bilinear (bilinear would invent fractional fuel-type values that don't exist); continuous layers (elevation, biomass, weather) use **bilinear or area-weighted mean**.
5. **Temporal alignment** — resample all dynamic sources to the common daily timestep. Fire observations: aggregate same-day passes into one daily active-fire mask. Weather: daily mean/accumulation from hourly/sub-daily source, using only data with a valid time at or before the target daily cutoff.
6. **Missing-data handling** — cloud-covered/no-observation fire pixels must be flagged as *missing*, not silently coded as "unburned" (a cloud-obscured burning pixel coded as unburned is a direct label-corruption bug). Carry an explicit per-pixel observation-quality/mask channel through the pipeline, and exclude masked pixels from the loss function rather than imputing a guessed fire state.
7. **Feature normalization** — compute normalization statistics (mean/std or min/max) **from the training split only**, then apply the same fixed statistics to validation/test — recomputing statistics per split is a leakage bug (§16, normalization leakage).
8. **Common grid assembly** — stack all static and dynamic channels into the final `X_static` / `X_dynamic` tensors per event, verified pixel-for-pixel aligned (a sanity check: overlay elevation contours against a known landmark, or check that every channel shares identical affine transform/shape before stacking).
9. **Training sequences** — slide an N-day window per event to produce `(X_static, X_dynamic[t-N:t]) → Y[t+1]` samples (§14 covers exactly how to split these).
10. **ConvLSTM** — final tensors fed to the model; static channels broadcast/concatenated at every timestep or injected once via a separate encoder branch (implementation detail, not a data-architecture concern).

---

## 14. Training/validation split strategy

**Do not randomly split individual timesteps from the same wildfire between train and test.** Consecutive days of the same fire are highly autocorrelated (today's fire shape is a near-superset of yesterday's) — a model that has seen day 5 of Fire A in training and is tested on day 6 of the same Fire A is not being tested on generalization, it's being tested on interpolation within a sequence it has already partially memorized. This inflates reported accuracy and is exactly the failure mode the WildfireSpreadTS authors built their 12-fold cross-validation specifically to avoid (§6), and it's the same failure mode a Western-Canada wildfire-risk paper found in this search avoided via a strict chronological (not random) split.

**Correct approach: event-level (and where possible, also time-block) splitting.** Every sequence sample from a given fire event must go entirely into train, entirely into validation, or entirely into test — never split across. A reasonable default: hold out entire fire *events* (not fire-days) for validation/test, stratified so validation/test still cover a representative range of fire sizes, fuel types, and regions rather than being a biased subset.

**Dataset size, honestly assessed:**
- **How many real events are realistically needed:** the two closest public benchmarks used 607 events (WildfireSpreadTS, US) and roughly thousands of fire-day samples across 18,545 fire-days (Next Day Wildfire Spread, US, though this counts fire-days not distinct fires). Canada's National Fire Database contains many more raw fire records, but only a subset will have clean, cloud-free, well-observed *daily progression* sequences suitable for this task — realistically, a first Canadian prototype should expect **on the order of tens to low hundreds of usable well-observed events**, not thousands, unless a substantial manual/automated QA effort is invested in screening NFDB/NBAC records for usable satellite coverage.
- **Sliding windows create inflated sample *counts*, not inflated *information*.** A 30-day fire event sliced into 3-day windows produces ~27 training samples, but they are not 27 independent pieces of evidence about fire-spread behaviour — they are 27 highly correlated views of one underlying process. Report both "number of sequences" and "number of independent events" when describing dataset size, and split/report metrics at the event level, not the sequence level, to avoid an inflated sense of statistical power.
- **Split ratios:** a reasonable starting point given a small event count is roughly 70/15/15 event-level, but with so few events, prefer k-fold cross-validation over a single fixed split (as WildfireSpreadTS does with 12-fold) so the small-sample variance in any single split doesn't dominate the reported result.

---

## 15. Real vs. simulation strategy

Four options as posed: (A) simulation-only, (B) real-only, (C) simulation-pretrain → real-finetune, (D) hybrid mixed.

- **(A) Simulation-only** — will not generalize to real fires unless the simulator itself is an extremely faithful physical model (most accessible research-grade simulators, including simple percolation/cellular-automata-style generators, encode simplified spread rules that do not capture the full complexity of real fuel heterogeneity, real weather variability, and real suppression interference). Useful for architecture debugging and unit-testing the pipeline, not as a final model.
- **(B) Real-only** — scientifically the cleanest (no risk of the model learning simulator-specific artifacts), but constrained by how few well-observed real Canadian events exist (§14). With only tens-to-low-hundreds of usable events, a data-hungry ConvLSTM trained from scratch risks underfitting/overfitting depending on model size.
- **(C) Simulation-pretrain → real-finetune** — the user's stated intuition, and a reasonable one *if* the simulator's spread physics are close enough to reality that pretraining teaches useful, transferable structure (e.g., "fire tends to spread contiguously, faster downwind, faster upslope") rather than simulator-specific quirks. This is a real risk, not a hypothetical one: if the custom simulator uses a spread rule meaningfully different from real fire behaviour (e.g., simplistic radial/isotropic growth without proper wind/slope coupling), pretraining could actively **bias** the network toward wrong priors that fine-tuning then has to unlearn — which can be slower than training from scratch on real data alone. **This needs to be tested empirically, not assumed.**
- **(D) Hybrid mixed** — training on a blended real+simulated set risks the model learning to distinguish (and exploit) simulation-vs-real distributional differences as a shortcut, unless the simulator is validated to be visually/statistically indistinguishable from real fire progression at the pixel level — a strong claim this project has not yet made.

**Recommendation:** treat (C) as a hypothesis to validate, not a default. Concretely: train (B) real-only and (C) sim-pretrain+real-finetune on the *same* real-event train/val/test split, and compare validation performance before investing further engineering effort into elaborating the simulator. If (C) doesn't measurably beat (B), the honest conclusion is that the current simulator's physics aren't a good match for real fire behaviour, and effort is better spent finding/preprocessing more real events (or a public real dataset, transfer-learned from WildfireSpreadTS/Next-Day-Wildfire-Spread on US data as a cross-domain pretraining source instead of synthetic data) than on improving the simulator.

---

## 16. ConvLSTM vs. alternatives

- **ConvLSTM** — established, moderate parameter count, explicit recurrent memory state well-suited to a genuinely sequential process like fire spread. Reasonable default given a small real-event dataset.
- **ConvGRU** — fewer parameters than ConvLSTM (no separate cell state), often comparable performance on sequence-to-sequence spatiotemporal tasks. Worth an empirical ablation against ConvLSTM given the project's small-data regime — cheaper and sometimes better on limited data, but not guaranteed, hence "ablation" not "replacement."
- **U-Net + temporal component** — the strongest literature precedent found: WildfireSpreadTS baselines and the Canada-specific 2026 preprint both use U-Net-family architectures (the latter, a Video Swin Transformer encoder + convolutional decoder). U-Net's skip connections are particularly good at preserving fine spatial detail (fire-front boundary sharpness) that a pure ConvLSTM bottleneck can blur. **This is a legitimate, precedented alternative to plain ConvLSTM for this exact task**, not a hypothetical one.
- **PredRNN** — designed for more general video-prediction benchmarks (Moving MNIST, traffic flow); more complex than ConvLSTM with more parameters to tune; not found in the wildfire-specific literature reviewed here, so its advantage for this specific problem is unverified.
- **Earthformer / spatiotemporal Transformer** — the review explicitly asked not to recommend this just because it's newer, and that instruction is scientifically correct here: transformer-family models are typically more data-hungry than convolutional-recurrent hybrids, and this project has a small, real-event-constrained dataset (§14). The one Canada-specific precedent found *does* use a Transformer component (Video Swin encoder), but notably wrapped inside a U-Net with a convolutional decoder and trained on GEE-scale public data spanning a full decade of Canadian fires — a substantially larger effective dataset than this project's likely early-stage real-event count. Recommend deferring a Transformer-heavy architecture until (a) the U-Net/ConvLSTM baseline is working and (b) the real training set is large enough that transformer data-hunger stops being the dominant risk.
- **CNN + temporal model (non-recurrent, e.g. 3D-CNN or CNN+MLP over flattened time)** — simpler than ConvLSTM, sometimes competitive, but loses the explicit gated-memory mechanism that helps with the very non-stationary bursts of behaviour (wind shifts, spotting) fire spread exhibits. A reasonable ablation baseline, not a primary recommendation.
- **Physics-informed neural networks (PINNs)** — embedding a PDE constraint (e.g. a level-set/Huygens-principle spread equation) directly into the loss is conceptually attractive for a physically-governed process like fire spread, but adds real implementation complexity and requires the physics constraint itself to be a reasonably accurate description of the phenomenon — for a first prototype, treat as a stretch goal, not baseline.
- **Hybrid physics + ML** — see §17, treated as feature-engineering rather than architecture choice.
- **Cellular automata + ML** — CA-based fire models (e.g. Cell2Fire-style) are fast and interpretable but their learned/ML component is typically a small correction on top of a hand-specified spread rule, not a general spatiotemporal predictor; a reasonable comparison baseline for the simulator itself (§15), not a competitor to ConvLSTM as the *learned* predictor.

**Recommendation for this project's stated constraints (limited real data):** a **U-Net-style encoder/decoder with a ConvLSTM (or ConvGRU) bottleneck/temporal module** — not a pure vanilla ConvLSTM stack, and explicitly not a full spatiotemporal Transformer for v1. This combines the two strongest pieces of literature evidence found (U-Net's precedent in WildfireSpreadTS and the Canada-specific paper; ConvLSTM/ConvGRU's parameter-efficiency for small real-event datasets) rather than picking one in isolation.

---

## 17. Physics-informed feature integration

Using a physics-based wildfire model (e.g. Rothermel-derived spread rate, spread direction, or a risk index like FWI) to generate *intermediate* features and feed them into the neural model is a legitimate, precedented idea (the literature search found ML wildfire papers explicitly using FWI-derived features as inputs), but it needs to be evaluated for redundancy, not assumed helpful.

- **Redundancy risk:** FWI System components (and by extension most physics-derived risk indices) are themselves closed-form functions of temperature, humidity, wind, and precipitation — exactly the raw weather channels already in the tensor. Feeding both the raw weather **and** its FWI-derived transform is not leakage (no future information), but it *is* redundant unless the specific nonlinear transform the physics model applies (e.g., exponential fuel-moisture decay curves) is hard for a CNN to learn implicitly from raw weather within the available training-data budget. This is genuinely uncertain without an ablation — recommend testing "raw weather only" vs. "raw weather + FWI components" as an explicit experiment rather than assuming the physics features help.
- **Spread rate/direction from a physics model** — more interesting than FWI, because a Rothermel-style spread-rate/direction estimate is a genuinely different computation than what raw weather+fuel channels directly express (it's a *nonlinear coupling* of wind, slope, and fuel type, not a simple per-channel function) — this has a better a priori case for adding new, non-redundant information than a scalar risk index does.
- **Where it should enter the architecture, if used:** as an additional **input channel** (computed once per timestep, stacked alongside the raw dynamic channels), not as a loss-function constraint (that would be the PINN approach, §16, a heavier design commitment) and not as a post-hoc correction on the network's output (that reintroduces the same "is the physics model's spread rule actually accurate" risk flagged in §15 for simulation pretraining). Treat it as one more engineered feature to ablate, with the same empirical-validation-before-adoption discipline recommended for the FWI channels and for simulation pretraining.

---

## 18. Data leakage risks — and exactly how to avoid each

**Future weather leaking into inputs.** If HRDPS forecast archives are used, a forecast *issued* after the target timestamp (even if it's a forecast *for* an earlier time, reprocessed/corrected later) must never be used for a training example whose target is at or before that issue time. Concretely: for a training example predicting day `t+1` from days `t-N..t`, every weather value used must come from an analysis (ERA5) or a forecast *issued* at or before day `t`, never a later-issued analysis/reanalysis that incorporates observations from after day `t`.

**Burned-area products generated after the event.** Some burned-area/land-cover products are annual composites that implicitly use post-fire-season imagery to finalize the year's map. Using that finalized layer as a "vegetation type" input for a *within-season* prediction risks leaking the fact that an area burned (post-fire vegetation classes often differ from pre-fire) into what's meant to be a static pre-fire fuel description. **Fix:** always use the fuel/vegetation layer vintage from *before* the fire season being modeled (e.g., last year's SCANFI/FBP layer for a fire predicted mid-season), never the same-year product.

**Vegetation products derived from future observations.** Same failure mode as above, generalized: any "static" layer must be verified to have been generated using only data available before the earliest date in the training window, not silently assumed static.

**Fire masks including information after the prediction timestamp.** The single highest-risk item. When building the "cumulative burned" and "time since ignition" channels (§9), make absolutely sure the cumulative mask used as *input* at time `t` only aggregates detections up to and including `t`, never detections from `t+1` onward — an off-by-one error here (e.g., using an end-of-day-inclusive mask when the model should only see up-to-start-of-day) is an easy, silent bug that directly leaks the answer.

**Exact construction rule:** for an input window `t-N..t` and target `t+1..t+K`, every single value in every channel of the input tensor must be verifiable as "known no later than end-of-day `t`." A concrete implementation safeguard: timestamp-tag every raw value at ingestion with its true availability time (not just its "valid for" time — a forecast's *valid* time and *issued* time can differ), and assert at pipeline-build time that `availability_time <= t` for every value entering the input tensor for that example.

---

## 19. Failure modes and safeguards

| Failure mode | Prevention |
|---|---|
| Coordinate-system mismatch | Enforce a single canonical CRS at ingestion; assert CRS equality before any raster operation, don't rely on visual inspection |
| Incorrect raster alignment | After reprojection, verify identical affine transform (origin, pixel size, rotation) across all channels for a given tile before stacking; fail loudly, don't silently reproject-on-the-fly per channel |
| Different pixel sizes across sources | Resample everything to the single target grid (§13 step 4) as an explicit, logged pipeline stage — never assume two "30m" products share the same actual grid |
| Temporal mismatch | Tag every value with both valid-time and availability-time (§18); reject any example where these are inconsistent with the leakage rule |
| Missing weather | Explicit missing-data mask channel; never silently zero-fill or forward-fill across a gap larger than a defined threshold without flagging it |
| Cloud-covered satellite imagery | Carry the source product's own QA/cloud-mask band through the pipeline; exclude cloud-flagged pixels from the loss, don't impute a guessed fire state |
| Fire detection errors (false positives, e.g. industrial heat sources; false negatives, e.g. small/cool fires under canopy) | Cross-check VIIRS detections against NFDB/NBAC event records for the same date/area before accepting a pixel as ground truth for a *named* fire event |
| Class imbalance (most pixels never burn) | Standard techniques (weighted loss, focal loss, or evaluating with metrics robust to imbalance like IoU/F1 on the fire class rather than raw pixel accuracy) — flag but don't over-engineer for v1 |
| Data leakage (general) | §18's explicit availability-time rule, enforced as an automated assertion in the pipeline, not a manual review step |
| Unrealistic fuel mapping | Use CFFDRS FBP + SCANFI (§11), not an invented scalar; if a project-specific fuel adjustment is ever needed, document the physical justification explicitly rather than picking convenient numbers |
| Circular wind encoding | u/v decomposition (§10), enforced at the feature-derivation stage, never store raw direction-in-degrees downstream of that stage |
| Normalization leakage | Compute normalization statistics from the training split only, freeze them, apply identically to val/test (§13 step 7) |
| Spatial train/test leakage | Event-level (not pixel/timestep-level) splitting (§14); additionally consider a spatial buffer between training and test event bounding boxes if any events are geographically adjacent/overlapping |
| Wildfire-event leakage | Same fire event's sequences must never appear in more than one of train/val/test (§14) |
| Edge effects | Buffer the clip region beyond the nominal study area (§13 step 3) so fire fronts near the tile boundary have real neighboring context rather than an artificial "no fuel beyond here" edge |
| Invalid interpolation | Categorical layers: nearest/modal resampling only, never bilinear (§13 step 4); never interpolate weather across the train/target time boundary (§10, §18) |
| Future information entering static/dynamic channels | The availability-time assertion (§18) applied uniformly to every channel, static and dynamic alike — "static" describes update frequency, not immunity from the leakage rule |

---

## 20. Final recommended architecture

**Grid:** 375 m common raster (§7), Canada-appropriate projected CRS.
**Temporal step:** 1 day, 3-day lookback window as a starting point (§8).
**Static channels (20):** elevation, slope, sin/cos aspect, CFFDRS FBP fuel type (16-class one-hot), SCANFI aboveground biomass, canopy height, crown closure (§12).
**Dynamic channels (10):** active-fire mask, cumulative-burned mask, time-since-detection, wind u, wind v, temperature, relative humidity, VPD, precipitation, day-of-year (§12).
**Model:** U-Net-style encoder/decoder with a ConvLSTM (or ablated ConvGRU) temporal module at the bottleneck — not a plain ConvLSTM stack, not a full spatiotemporal Transformer for v1 (§16).
**Training data strategy:** real Canadian events as the primary target (§6's NFDB/NBAC/VIIRS pipeline), with simulation-pretraining treated as a hypothesis to validate empirically against a real-only baseline before committing to it, not assumed beneficial by default (§15).
**Splitting:** strict event-level splitting, k-fold if the real-event count is small, never timestep-level random splitting (§14).
**Physics-informed features:** optional FWI/spread-rate channels, added only if an ablation shows they measurably help beyond what the raw weather+fuel channels already express (§17).

### Falsification check

The strongest argument against this recommendation: *given how few well-observed real Canadian wildfire events likely exist with clean daily progression data (§14 estimates tens to low hundreds), any data-hungry deep spatiotemporal model — U-Net-ConvLSTM hybrid included — may be fundamentally under-constrained regardless of which specific architecture is chosen, and a much smaller-parameter approach (a physics-informed or cellular-automata-hybrid model with far fewer learned weights) might generalize better with this little real data than any of the deep-learning options compared in §16.*

This argument does not overturn the recommendation, but it does **change its emphasis**: it's the reason §15 insists simulation-pretraining be validated empirically rather than assumed, why §16 explicitly rules out the Transformer-heavy option for v1, and why §17 treats physics-informed features as a serious candidate rather than a nice-to-have. If, after building the real-event dataset, the usable event count turns out to be at the low end of the tens-to-low-hundreds estimate, the honest next step is not to make the neural architecture more sophisticated — it's to either invest more heavily in validated-simulation pretraining (§15) or to seriously reconsider a smaller-parameter, more physics-constrained model as the primary approach rather than a stretch goal. That reconsideration should be driven by the actual real-event count once measured, not decided in advance.

---

## 21. Step-by-step implementation plan

1. **Inventory real events.** Query NFDB/NBAC for Canadian wildfire events in the target region/period; for each, check VIIRS active-fire coverage overlap and cloud-free fraction. Produce a concrete count of "usable, well-observed events" before writing any model code — this number directly determines whether §15/§16's data-hungriness concerns are theoretical or immediate.
2. **Build the static layers once** for the study area: reproject/resample CDEM or Copernicus DEM, CFFDRS FBP fuel type, and SCANFI biomass/canopy/crown-closure to the common 375 m grid (§13 steps 1–4). Verify pixel alignment with an explicit affine-transform equality check.
3. **Build the per-event dynamic layers**, event by event: VIIRS-derived active/cumulative/time-since-detection masks, ERA5 (or HRDPS) weather resampled to daily/375 m, with every value tagged with availability-time per §18.
4. **Assemble training sequences** with the leakage-safe availability-time assertion (§18) as an automated pipeline check, not a manual review.
5. **Split at the event level** (§14), k-fold if the usable-event count from step 1 is small.
6. **Baseline model:** train the plain ConvLSTM stack first as the simplest working baseline, using the channel spec in §12 minus the optional aspect/FWI channels — get an end-to-end pipeline working before adding architectural sophistication.
7. **Upgrade to U-Net+ConvLSTM/ConvGRU** (§16, §20) and compare against the baseline on the same event-level split.
8. **Ablate the optional channels** (VPD vs. RH-only, FWI-derived features, aspect) individually against the step-7 model, keeping everything else fixed, to decide which are worth their added complexity per §17's redundancy concern.
9. **Run the simulation-pretraining experiment** (§15): train real-only vs. sim-pretrain+real-finetune on the identical event-level split; only proceed with simulation-based pretraining as a standing part of the pipeline if it measurably beats real-only.
10. **Re-evaluate architecture choice** using the falsification check in §20 once the real usable-event count and the step-9 result are both known — this is the point at which "stick with U-Net-ConvLSTM" vs. "shift toward a smaller physics-constrained model" should be decided on evidence, not assumption.

---

## Sources referenced

- [FIRMS | NASA Earthdata](https://www.earthdata.nasa.gov/data/tools/firms)
- [VIIRS (NOAA-21) 375 m Active Fire Product NRT, NASA Open Data Portal](https://data.nasa.gov/dataset/viirs-noaa-21-i-band-375-m-active-fire-product-nrt-vector-data-e73a2)
- [Canadian Wildland Fire Information System (CWFIS)](https://cwfis.cfs.nrcan.gc.ca/en)
- [CWFIS Datamart / National Burned Area Composite](https://cwfis.cfs.nrcan.gc.ca/ha/nfdb?type=nbac)
- [Fire Monitoring, Accounting and Reporting System — NRCan](https://natural-resources.canada.ca/forests-forestry/wildland-fires/fire-monitoring-accounting-reporting-system)
- [CFFDRS FBP Fuel Types 2024, 30m — Open Government Portal](https://open.canada.ca/data/en/dataset/4e66dd2f-5cd0-42fd-b82c-a430044b31de)
- [Canada's Fire Behaviour Prediction System — Fuel Types, NRCan](https://natural-resources.canada.ca/forests-forestry/wildland-fires/canada-fire-behaviour-prediction-system-fuel-types)
- [40 Scott and Burgan Fire Behavior Fuel Models — LANDFIRE](https://landfire.gov/fuel/fbfm40)
- [13 Anderson Fire Behavior Fuel Models — LANDFIRE](https://www.landfire.gov/fuel/fbfm13)
- [European Forest Fire Information System — Copernicus](https://www.copernicus.eu/en/european-forest-fire-information-system)
- [EFFIS Rapid Damage Assessment](https://forest-fire.emergency.copernicus.eu/about-effis/technical-background/rapid-damage-assessment)
- [ESA WorldCover](https://esa-worldcover.org/en)
- [ESA WorldCover — Registry of Open Data on AWS](https://registry.opendata.aws/esa-worldcover-vito/)
- [Canadian Digital Elevation Model — product specification, NRCan](https://ftp.maps.canada.ca/pub/nrcan_rncan/elevation/cdem_mnec/doc/CDEM_product_specs.pdf)
- [Canadian Digital Elevation Model, 1945-2011 — Open Government Portal](https://open.canada.ca/data/en/dataset/7f245e4d-76c2-4caa-951a-45d1d2051333)
- [Copernicus DEM — Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM)
- [Licence for Copernicus DEM instance COP-DEM-GLO-30](https://docs.sentinel-hub.com/api/latest/static/files/data/dem/resources/license/License-COPDEM-30.pdf)
- [ECMWF Reanalysis v5 (ERA5) dataset](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
- [ERA5 hourly data on single levels — Copernicus Climate Data Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
- [High Resolution Deterministic Prediction System — ECCC readme](https://eccc-msc.github.io/open-data/msc-data/nwp_hrdps/readme_hrdps_en/)
- [HRDPS — Open Government Portal](https://open.canada.ca/data/en/dataset/5b401fa0-6c29-57f0-b3d5-749f301d829d)
- [A new approach for spatializing the Canadian National Forest Inventory (SCANFI) — Can. J. Forest Res.](https://cdnsciencepub.com/doi/10.1139/cjfr-2023-0118)
- [SCANFI v2 — Open Government Portal](https://open.canada.ca/data/en/dataset/07653869-f303-46c2-a04e-9ab479b73cbf)
- [MODIS/Terra+Aqua MCD64A1 Burned Area Monthly — NASA Earthdata](https://www.earthdata.nasa.gov/data/catalog/lpcloud-mcd64a1-061)
- [GOES-R Fire / Hot Spot Characterization products](https://www.goes-r.gov/products/baseline-fire-hot-spot.html)
- [Canadian National Fire Database (NFDB) — CWFIS](https://cwfis.cfs.nrcan.gc.ca/ha/nfdb)
- [NFDB point shapefile metadata](https://cwfis.cfs.nrcan.gc.ca/downloads/nfdb/fire_pnt/current_version/NFDB_point_shapefile_metadata.pdf)
- [CWFIS Fire Weather Index methodology](https://cwfis.cfs.nrcan.gc.ca/index.php/background/dsm/fwi)
- [A daily gridded FWI dataset for Canada — ESSD preprint](https://essd.copernicus.org/preprints/essd-2025-535/)
- [VPD-based dead fine fuel moisture models — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0168192323005580)
- [Next Day Wildfire Spread — arXiv:2112.02447](https://arxiv.org/abs/2112.02447)
- [Next Day Wildfire Spread — Google Research GitHub](https://github.com/google-research/google-research/blob/master/simulation_research/next_day_wildfire_spread/README.md)
- [WildfireSpreadTS — NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ebd545176bdaa9cd5d45954947bd74b7-Abstract-Datasets_and_Benchmarks.html)
- [Spatio-Temporal Wildfire Spread Prediction in Canada using a Video Swin-Hybrid-U-Net — arXiv:2606.20693](https://arxiv.org/html/2606.20693) *(abstract-level detail only — full text not independently verified)*
- [BCWildfire dataset — arXiv:2511.17597](https://arxiv.org/pdf/2511.17597) *(abstract-level detail only — full text not independently verified)*
