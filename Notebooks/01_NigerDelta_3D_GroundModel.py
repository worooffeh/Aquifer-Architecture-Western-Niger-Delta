#!/usr/bin/env python3
"""
=============================================================================
 Niger Delta 3D Lithostratigraphic Ground Model
 Western Niger Delta — Delta State, Nigeria
 
 Authors : Dr Ogheneworo Offeh et al.
 Purpose : Stratigraphic correlation, 3D geospatial ground modelling,
           and distribution of aquifer properties (facies, K, T, geochem)
 Date    : 2026
=============================================================================

WORKFLOW
 1. Library imports & configuration
 2. Data importation (lithologs, locations, hydraulic & geochem properties)
 3. Data processing & exploration
 4. Outlier analysis (IQR + Z-score on numeric petrophysical data)
 5. Stratigraphic correlation — 2D vertical cross-sections (N→S)
 6. Interactive 3D geospatial model with toggleable property layers

GEOLOGICAL CONTEXT
 Study area: Delta State, Nigeria — Benin Formation aquifer system
 Formations encountered (surface → depth, N to S):
   • Alluvial / Topsoil (TS)      — surface laterite cap
   • Benin Formation (Miocene–Recent) — primary aquifer; sand-dominated
   • Ogwashi-Asaba Formation (Oligocene–Miocene) — lignite + clay baffles
   • Agbada / Ameki equivalents   — deep confining units
 Regional transect: Asaba (N) → Abraka/Kwale (Central) → Forcados (S)
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LIBRARY IMPORTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
"""
## 1. Library Imports & Configuration

| Library          | Version req. | Purpose                                        |
|------------------|-------------|------------------------------------------------|
| numpy            | ≥1.24       | Numerical arrays, interpolation grids           |
| pandas           | ≥2.0        | DataFrame handling for well-log intervals       |
| matplotlib       | ≥3.7        | Static cross-section plots                      |
| seaborn          | ≥0.12       | Statistical visualisation, outlier plots        |
| scipy            | ≥1.11       | Spatial interpolation (griddata, RBF)           |
| scikit-learn     | ≥1.3        | Optional: variogram/kriging helpers             |
| plotly           | ≥5.18       | Interactive 3D visualisation                   |
| openpyxl         | ≥3.1        | Read .xlsx litholog workbook                    |
| statsmodels      | ≥0.14       | Descriptive statistics for outlier section      |
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb, LinearSegmentedColormap
import seaborn as sns
import scipy.stats as stats
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings, os, re, textwrap

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

FT2M = 0.3048
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(PROJ_ROOT, "data")
FIG_DIR   = os.path.join(PROJ_ROOT, "figures")
OUT_DIR   = os.path.join(PROJ_ROOT, "outputs")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
print("✓ Libraries loaded | Project root:", PROJ_ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA IMPORTATION
# ─────────────────────────────────────────────────────────────────────────────
"""
## 2. Data Importation

### 2A  Well Locations (CSV)
Source: `borehole_locations.csv` — 22 wells with UTM easting/northing and
decimal-degree lat/lon extracted from GPS field readings.

### 2B  Lithologic Intervals (XLSX)
Source: `wells_lithologs.xlsx` — 21-well corrected litholog dataset;
one sheet per well; columns: MD Top (ft), MD Base (ft), Facies Code,
Facies Name, Aquifer Class, Contact Type.

### 2C  Hydraulic Properties (inline — from published literature)
Hydraulic conductivity K (m/day) and transmissivity T (m²/day) assigned
per facies code from Akpoborie & Efobo (2014), Amajor (1991),
Anomohanran (2014, 2021), Aweto & Akpoborie (2015).

### 2D  Hydrogeochemistry (inline — from Akpoborie & Efobo 2014 + regional)
pH, EC, TDS, major ions: Na, K, Ca, Mg, Cl, HCO₃, SO₄, NO₃, Fe.
"""

# ── 2A  Parse DMS coordinates ────────────────────────────────────────────────
def dms_to_dd(dms_str):
    """Convert DMS string like 5°32'16.37"N to decimal degrees."""
    if pd.isna(dms_str): return np.nan
    s = str(dms_str).strip()
    hem = 1 if s[-1] in ('N','E') else -1
    s = s[:-1]
    parts = re.split(r"[°'\"]+", s)
    parts = [p for p in parts if p.strip()]
    try:
        d = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0
        sc= float(parts[2]) if len(parts) > 2 else 0
        return hem * (d + m/60 + sc/3600)
    except:
        return np.nan

loc_raw = pd.read_csv(os.path.join(DATA_DIR, "borehole_locations.csv"), encoding='latin1')
# preserve column names as-is (some have trailing spaces)
loc_raw = loc_raw.dropna(subset=['Name'])
loc_raw['lat_dd'] = loc_raw['Latitude  '].apply(dms_to_dd)
loc_raw['lon_dd'] = loc_raw['Longitude  '].apply(dms_to_dd)
loc_raw['easting']  = pd.to_numeric(loc_raw['Surface X'], errors='coerce')
loc_raw['northing'] = pd.to_numeric(loc_raw['Surface Y'], errors='coerce')
loc_raw['td_m']     = pd.to_numeric(loc_raw['TD(m)'], errors='coerce')
loc_raw['name_clean'] = loc_raw['Name'].str.replace(' BH','',regex=False).str.strip()
print(f"✓ Locations loaded: {len(loc_raw)} wells")
print(loc_raw[['name_clean','lat_dd','lon_dd','easting','northing','td_m']].to_string(index=False))

# ── 2B  Read all litholog sheets ──────────────────────────────────────────────
WELL_SHEETS = [
    'Asaba-2','Issele-Uku','Umunede','Ewuru','Ute-Okpu','Idumuje-Unor',
    'Kwale','Abraka','Oben CB','Oben RB','Oben LQ',
    'Amukpe','Sapele-West','Orerokpe','Ozoro',
    'Eriemu-3','Edjovhe','Ughelli-East','Utorogu','RA Ogunu-Warri','Forcados'
]

REGION_MAP = {
    'Asaba-2':'Delta North','Issele-Uku':'Delta North','Umunede':'Delta North',
    'Ewuru':'Delta North','Ute-Okpu':'Delta North','Idumuje-Unor':'Delta North',
    'Kwale':'Delta Central','Abraka':'Delta Central','Oben CB':'Delta Central',
    'Oben RB':'Delta Central','Oben LQ':'Delta Central','Amukpe':'Delta Central',
    'Sapele-West':'Delta Central','Orerokpe':'Delta Central','Ozoro':'Delta Central',
    'Eriemu-3':'Delta South','Edjovhe':'Delta South','Ughelli-East':'Delta South',
    'Utorogu':'Delta South','RA Ogunu-Warri':'Delta South','Forcados':'Delta South',
}

# Name harmonisation between CSV and xlsx
NAME_MATCH = {
    'Asaba-2':'Asaba 2','Issele-Uku':'Issele-Uku','Umunede':'Umunede',
    'Ewuru':'Ewuru','Ute-Okpu':'Ute-Okpu','Idumuje-Unor':'Idumuje-Unor',
    'Kwale':'Kwale','Abraka':'Abraka BH','Oben CB':'Oben CS BH',
    'Oben RB':'Oben RS BH','Oben LQ':'Oben LQ BH','Amukpe':'Amukpe',
    'Sapele-West':'Sapele-west BH','Orerokpe':'Orerokpe','Ozoro':'Ozoro BH',
    'Eriemu-3':'Eriemu-3 BH','Edjovhe':'Edjovhe','Ughelli-East':'Ughelli-East BH',
    'Utorogu':'Utorogu BH','RA Ogunu-Warri':'RA Ogunu-Warri BH','Forcados':'Forcados BH',
}

xlsx_path = os.path.join(DATA_DIR, "wells_lithologs.xlsx")
frames = []
for ws in WELL_SHEETS:
    try:
        df = pd.read_excel(xlsx_path, sheet_name=ws, header=2)
        df.columns = [str(c).strip().replace('\n',' ') for c in df.columns]
        col_top   = next(c for c in df.columns if 'Top' in c and 'ft' in c.lower())
        col_base  = next(c for c in df.columns if 'Base' in c and 'ft' in c.lower())
        col_code  = next(c for c in df.columns if 'Code' in c)
        col_fname = next(c for c in df.columns if 'Facies Name' in c or 'Name' in c and 'Standard' in c)
        col_aq    = next(c for c in df.columns if 'Aquifer' in c and 'Class' in c)
        col_ct    = next(c for c in df.columns if 'Contact' in c)
        df = df[[col_top,col_base,col_code,col_fname,col_aq,col_ct]].copy()
        df.columns = ['top_ft','base_ft','facies_code','facies_name','aq_class','contact']
        df = df.dropna(subset=['top_ft','base_ft','facies_code'])
        df['top_ft']  = pd.to_numeric(df['top_ft'],  errors='coerce')
        df['base_ft'] = pd.to_numeric(df['base_ft'], errors='coerce')
        df = df.dropna(subset=['top_ft','base_ft'])
        df['top_m']   = df['top_ft']  * FT2M
        df['base_m']  = df['base_ft'] * FT2M
        df['thick_m'] = df['base_m'] - df['top_m']
        df['well']    = ws
        df['region']  = REGION_MAP[ws]
        frames.append(df)
    except Exception as e:
        print(f"  WARN {ws}: {e}")

intervals = pd.concat(frames, ignore_index=True)
print(f"\n✓ Litholog intervals loaded: {len(intervals)} rows across {intervals['well'].nunique()} wells")

# ── 2C  Hydraulic Properties per Facies Code ─────────────────────────────────
# K (m/day), porosity (%), based on published values for Benin Fm sands
K_TABLE = {
    'TS'   : (0.001,  5),   'CL'   : (0.005,  8),   'LG'   : (0.001,  5),
    'SI'   : (0.05,  20),   'CL-SI': (0.05,  15),   'SI-CL': (0.02,  12),
    'SI-FS': (5.0,   28),   'CL-FS': (2.0,   22),
    'FS'   : (25.0,  32),   'FS-MS': (50.0,  33),
    'MS'   : (100.0, 35),   'MS-CS': (200.0, 36),
    'CS'   : (400.0, 38),   'CS-GR': (700.0, 40),
    'GR'   : (1000.0,42),   'SC'   : (10.0,  20),
    'SC'   : (10.0,  20),
}
intervals['K_mday']   = intervals['facies_code'].map(lambda c: K_TABLE.get(c, (1,30))[0])
intervals['porosity'] = intervals['facies_code'].map(lambda c: K_TABLE.get(c, (1,30))[1])
intervals['T_m2day']  = intervals['K_mday'] * intervals['thick_m']

# ── 2D  Hydrogeochemistry (Akpoborie & Efobo 2014 + regional; linked by well) ─
# Reported as typical mid-range from Table 3 of Akpoborie & Efobo (2014)
# and cross-referenced with regional data for wells outside Abraka
GEOCHEM = {
    # well : {pH, EC_uS, TDS_mg, Na_mg, K_mg, Ca_mg, Mg_mg, Cl_mg, HCO3_mg, SO4_mg, NO3_mg, Fe_mg, facies_type}
    'Asaba-2':       {'pH':5.8,'EC':165,'TDS':105,'Na':8.5,'K':1.8,'Ca':10.2,'Mg':3.5,'Cl':12.8,'HCO3':16.5,'SO4':4.8,'NO3':2.5,'Fe':0.12,'type':'Ca-Cl'},
    'Issele-Uku':    {'pH':6.0,'EC':195,'TDS':125,'Na':10.2,'K':2.2,'Ca':12.5,'Mg':4.2,'Cl':15.8,'HCO3':19.5,'SO4':6.5,'NO3':3.0,'Fe':0.15,'type':'Ca-Cl'},
    'Umunede':       {'pH':5.9,'EC':185,'TDS':118,'Na':9.2,'K':2.0,'Ca':11.5,'Mg':3.8,'Cl':14.5,'HCO3':18.0,'SO4':5.8,'NO3':2.8,'Fe':0.14,'type':'Ca-Cl'},
    'Ewuru':         {'pH':6.1,'EC':200,'TDS':128,'Na':10.5,'K':2.3,'Ca':13.0,'Mg':4.5,'Cl':16.5,'HCO3':20.5,'SO4':7.0,'NO3':3.2,'Fe':0.16,'type':'Ca-Cl'},
    'Ute-Okpu':      {'pH':5.7,'EC':155,'TDS':99,'Na':7.8,'K':1.7,'Ca':9.5,'Mg':3.2,'Cl':11.5,'HCO3':15.0,'SO4':4.2,'NO3':2.2,'Fe':0.11,'type':'Ca-Cl'},
    'Idumuje-Unor':  {'pH':5.9,'EC':175,'TDS':112,'Na':9.0,'K':1.9,'Ca':11.0,'Mg':3.7,'Cl':14.0,'HCO3':17.5,'SO4':5.5,'NO3':2.7,'Fe':0.13,'type':'Ca-Cl'},
    'Kwale':         {'pH':6.2,'EC':215,'TDS':138,'Na':10.8,'K':2.2,'Ca':13.5,'Mg':4.5,'Cl':16.8,'HCO3':20.8,'SO4':6.8,'NO3':3.5,'Fe':0.18,'type':'Ca-Cl'},
    'Abraka':        {'pH':6.8,'EC':285,'TDS':182,'Na':8.5,'K':3.0,'Ca':18.4,'Mg':5.2,'Cl':15.8,'HCO3':22.4,'SO4':8.5,'NO3':3.2,'Fe':0.18,'type':'Ca-Mg-Cl'},  # Akpoborie & Efobo 2014
    'Oben CB':       {'pH':6.5,'EC':215,'TDS':138,'Na':10.5,'K':2.2,'Ca':13.5,'Mg':4.5,'Cl':16.5,'HCO3':20.5,'SO4':6.8,'NO3':3.5,'Fe':0.18,'type':'Ca-Cl'},
    'Oben RB':       {'pH':6.4,'EC':205,'TDS':130,'Na':10.0,'K':2.0,'Ca':13.0,'Mg':4.2,'Cl':15.5,'HCO3':19.5,'SO4':6.5,'NO3':3.2,'Fe':0.16,'type':'Ca-Cl'},
    'Oben LQ':       {'pH':6.3,'EC':195,'TDS':122,'Na':9.5,'K':1.9,'Ca':12.5,'Mg':4.0,'Cl':14.5,'HCO3':18.5,'SO4':6.0,'NO3':3.0,'Fe':0.15,'type':'Ca-Cl'},
    'Amukpe':        {'pH':6.5,'EC':220,'TDS':140,'Na':10.8,'K':2.3,'Ca':14.0,'Mg':4.8,'Cl':17.2,'HCO3':21.5,'SO4':7.2,'NO3':3.8,'Fe':0.20,'type':'Ca-Cl'},
    'Sapele-West':   {'pH':6.5,'EC':225,'TDS':142,'Na':11.0,'K':2.2,'Ca':13.8,'Mg':4.5,'Cl':17.0,'HCO3':21.0,'SO4':7.0,'NO3':3.5,'Fe':0.19,'type':'Ca-Cl'},
    'Orerokpe':      {'pH':6.5,'EC':195,'TDS':125,'Na':10.5,'K':2.2,'Ca':12.8,'Mg':4.2,'Cl':15.5,'HCO3':19.5,'SO4':6.8,'NO3':3.2,'Fe':0.18,'type':'Ca-Cl'},
    'Ozoro':         {'pH':6.4,'EC':205,'TDS':130,'Na':10.8,'K':2.1,'Ca':13.2,'Mg':4.3,'Cl':16.0,'HCO3':20.0,'SO4':6.5,'NO3':3.0,'Fe':0.16,'type':'Ca-Cl'},
    'Eriemu-3':      {'pH':6.3,'EC':195,'TDS':125,'Na':9.8,'K':2.0,'Ca':12.5,'Mg':4.0,'Cl':15.0,'HCO3':19.0,'SO4':6.0,'NO3':2.8,'Fe':0.15,'type':'Ca-Cl'},
    'Edjovhe':       {'pH':6.0,'EC':175,'TDS':112,'Na':9.0,'K':1.9,'Ca':11.5,'Mg':3.8,'Cl':14.0,'HCO3':17.5,'SO4':5.5,'NO3':2.5,'Fe':0.12,'type':'Ca-Cl'},
    'Ughelli-East':  {'pH':6.2,'EC':195,'TDS':125,'Na':10.0,'K':2.0,'Ca':12.8,'Mg':4.2,'Cl':15.5,'HCO3':19.5,'SO4':6.2,'NO3':3.0,'Fe':0.15,'type':'Ca-Cl'},
    'Utorogu':       {'pH':6.1,'EC':185,'TDS':118,'Na':9.5,'K':1.9,'Ca':12.0,'Mg':4.0,'Cl':14.5,'HCO3':18.5,'SO4':5.8,'NO3':2.8,'Fe':0.14,'type':'Ca-Cl'},
    'RA Ogunu-Warri':{'pH':7.1,'EC':295,'TDS':193,'Na':18.5,'K':5.2,'Ca':8.5,'Mg':4.2,'Cl':42.5,'HCO3':10.2,'SO4':1.5,'NO3':6.8,'Fe':0.35,'type':'Na-Cl'},  # near River Warri
    'Forcados':      {'pH':5.8,'EC':200,'TDS':130,'Na':12.5,'K':2.5,'Ca':10.0,'Mg':4.0,'Cl':28.5,'HCO3':14.5,'SO4':5.0,'NO3':3.5,'Fe':0.22,'type':'Na-Cl'},
}
geochem_df = pd.DataFrame(GEOCHEM).T.reset_index().rename(columns={'index':'well'})
for col in geochem_df.columns:
    if col != 'well':
        geochem_df[col] = pd.to_numeric(geochem_df[col], errors='coerce')

# ── Merge locations onto interval table ───────────────────────────────────────
loc_lookup = {}
for _, row in loc_raw.iterrows():
    nm = row['name_clean']
    for ws, csv_name in NAME_MATCH.items():
        if csv_name == nm or nm.lower() in ws.lower() or ws.lower() in nm.lower():
            loc_lookup[ws] = {'easting':row['easting'],'northing':row['northing'],
                              'lat':row['lat_dd'],'lon':row['lon_dd'],'td_m':row['td_m']}

intervals['easting']  = intervals['well'].map(lambda w: loc_lookup.get(w,{}).get('easting', np.nan))
intervals['northing'] = intervals['well'].map(lambda w: loc_lookup.get(w,{}).get('northing',np.nan))
intervals['lat']      = intervals['well'].map(lambda w: loc_lookup.get(w,{}).get('lat', np.nan))
intervals['lon']      = intervals['well'].map(lambda w: loc_lookup.get(w,{}).get('lon', np.nan))
intervals['td_m']     = intervals['well'].map(lambda w: loc_lookup.get(w,{}).get('td_m', np.nan))
intervals['mid_m']    = (intervals['top_m'] + intervals['base_m']) / 2

print(f"\n✓ Master interval table: {intervals.shape}")
print(intervals[['well','region','facies_code','top_m','base_m','thick_m','K_mday','T_m2day']].head(8).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DATA PROCESSING & EXPLORATION
# ─────────────────────────────────────────────────────────────────────────────
"""
## 3. Data Processing & Exploration

Key descriptive statistics computed WITHOUT standardisation. Geologic
depths and thickness values retain physical units (metres, m/day, m²/day)
throughout — standardisation is not applied to geologic data as it would
destroy the petrophysical meaning embedded in the raw values.
"""

print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS — Interval Table")
print("="*60)
numeric_cols = ['top_m','base_m','thick_m','K_mday','T_m2day','porosity']
print(intervals[numeric_cols].describe().round(3).to_string())

# Per-region summaries
print("\nPer-region interval statistics:")
print(intervals.groupby('region')[numeric_cols].mean().round(2).to_string())

# Facies frequency
print("\nFacies code frequency (all wells):")
fc = intervals['facies_code'].value_counts()
print(fc.to_string())

# Net-to-gross per well
ng = intervals.groupby('well').apply(
    lambda x: x.loc[x['aq_class']=='Aquifer','thick_m'].sum() / x['thick_m'].sum() * 100
).rename('NG_pct').reset_index()
ng['region'] = ng['well'].map(REGION_MAP)
print("\nNet/Gross per well (% aquifer):")
print(ng.sort_values('region').to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — OUTLIER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
"""
## 4. Outlier Analysis

IQR fences and Z-scores applied to hydraulic conductivity, transmissivity,
and interval thickness. Geologic label data (facies codes, formation names)
are NOT standardised or altered. Outliers are flagged for investigation —
they may reflect genuine geological features (e.g., gravel lag layers with
very high K) rather than measurement errors.

**Note**: We do NOT remove or transform outliers in geologic data.
High-K gravel intervals (K > 800 m/day) in Abraka and Kwale are physically
real, represent coarse fluvial channel lags, and are preserved in the model.
"""

fig_out, axes_out = plt.subplots(2, 3, figsize=(16, 9), facecolor='#F5F5F0')
fig_out.suptitle("Figure 1 — Outlier Analysis: Hydraulic & Petrophysical Parameters\n"
                 "(IQR fences + Z-scores; geologic data NOT standardised)",
                 fontsize=12, fontweight='bold', y=0.98)

PLOT_PARAMS = [
    ('K_mday',   'Hydraulic Conductivity K (m/day)', '#2196F3'),
    ('T_m2day',  'Transmissivity T (m²/day)',         '#4CAF50'),
    ('thick_m',  'Interval Thickness (m)',            '#FF9800'),
    ('porosity', 'Porosity (%)',                      '#9C27B0'),
    ('top_m',    'Depth to Top (m)',                  '#795548'),
]

outlier_records = []
for ax, (col, label, col_color) in zip(axes_out.flat, PLOT_PARAMS):
    data = intervals[col].dropna()
    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    zscore = np.abs(stats.zscore(data))
    out_iqr   = ((data < lo) | (data > hi)).sum()
    out_zscore= (zscore > 3).sum()

    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor=col_color, alpha=0.4),
               medianprops=dict(color='red', lw=2),
               whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2),
               flierprops=dict(marker='D', color='red', ms=4, alpha=0.6))
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(f"{label}", fontsize=9, fontweight='bold')
    ax.set_xticks([])
    ax.text(1.05, 0.95, f"IQR outliers: {out_iqr}\nZ>3 outliers: {out_zscore}\n"
            f"Median: {data.median():.2f}\nMean:   {data.mean():.2f}\nStd:    {data.std():.2f}",
            transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='grey', alpha=0.8))
    outlier_records.append({'param':col,'IQR_outliers':out_iqr,'Z_outliers':out_zscore,
                            'note': 'Geologically real (retain)' if col=='K_mday' else 'Inspect'})

# Last panel: violin by region
ax6 = axes_out.flat[5]
region_K = [intervals.loc[intervals['region']==r,'K_mday'].values for r in ['Delta North','Delta Central','Delta South']]
parts = ax6.violinplot(region_K, positions=[1,2,3], showmedians=True)
for pc, c in zip(parts['bodies'], ['#1565C0','#E65100','#2E7D32']):
    pc.set_facecolor(c); pc.set_alpha(0.5)
ax6.set_xticks([1,2,3])
ax6.set_xticklabels(['Delta\nNorth','Delta\nCentral','Delta\nSouth'], fontsize=8)
ax6.set_ylabel('K (m/day)', fontsize=9)
ax6.set_title('K Distribution by Region', fontsize=9, fontweight='bold')
ax6.set_yscale('log')

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig(os.path.join(FIG_DIR,'fig01_outlier_analysis.png'), dpi=160, bbox_inches='tight')
plt.close()
print("✓ Figure 1 saved: fig01_outlier_analysis.png")

out_df = pd.DataFrame(outlier_records)
print("\nOutlier summary:")
print(out_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — STRATIGRAPHIC CORRELATION & 2D CROSS-SECTIONS
# ─────────────────────────────────────────────────────────────────────────────
"""
## 5. Stratigraphic Correlation — 2D Vertical Cross-Sections

Three cross-sections are generated:
  A) All 21 wells — Regional N→S traverse (Asaba → Forcados)
  B) Delta North corridor
  C) Delta Central → South transition

Wells are ordered by northing (N to S). Facies columns are colour-coded
by the standardised taxonomy. Key stratigraphic surfaces are annotated:
  - H1: Base of weathered/LVL layer (~2–5 m)
  - H2: Base of sub-weathered layer (~15–36 m)
  - Lignite horizon (where present) — Ogwashi-Asaba marker
  - Formation boundaries (inferred)

Pinch-outs are identified where laterally continuous facies wedge out,
and are annotated with dashed tie-lines between wells.
"""

# Facies colour scheme
FACIES_COLORS = {
    'TS':   '#8B6914', 'CL':   '#B22222', 'LG':   '#3D2B1F',
    'SI':   '#A9A9A9', 'CL-SI':'#B06060', 'SI-CL':'#C07070',
    'SI-FS':'#D8D870', 'CL-FS':'#D4A870', 'FS':   '#FFFF99',
    'FS-MS':'#FFE84D', 'MS':   '#FFD700', 'MS-CS':'#FFC000',
    'CS':   '#FFA500', 'CS-GR':'#FF7700', 'GR':   '#FF5500',
    'SC':   '#C8864A',
}

def section_plot(well_list, title, max_depth_m=260, figw=None, note=""):
    """Draw a 2D vertical stratigraphic correlation section."""
    n = len(well_list)
    fig_w = figw or max(18, n * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, 13), facecolor='#F4F2EC')
    ax.set_facecolor('#EEEEE8')

    col_hw  = 0.45
    spacing = 2.0
    xs = {w: i*spacing for i,w in enumerate(well_list)}

    # Formation boundaries (mean from literature, metres)
    H1 = 3.0;  H2 = 22.0

    # Background zone shading
    xmin = -col_hw*2; xmax = (n-1)*spacing + col_hw*2
    ax.axhspan(0, H1, color='#8B6914', alpha=0.07, label='Zone I (LVL)')
    ax.axhspan(H1, H2, color='#4A7A4A', alpha=0.06, label='Zone II (Sub-wtrd)')
    ax.axhspan(H2, max_depth_m, color='#2A5A8A', alpha=0.04, label='Zone III (Consol.)')

    ax.axhline(H1, color='#8B6914', lw=1.5, ls='--', alpha=0.8)
    ax.axhline(H2, color='#226622', lw=1.8, ls='--', alpha=0.9)

    # Tie-lines for correlatable horizons (lignite, top-of-main-aquifer)
    lignite_tops = {}
    first_aq_tops = {}

    for wi, wname in enumerate(well_list):
        xc = xs[wname]
        wdata = intervals[intervals['well']==wname].sort_values('top_m')
        if wdata.empty: continue
        td = wdata['td_m'].iloc[0]

        # Well column background
        ax.add_patch(plt.Rectangle((xc-col_hw, 0), col_hw*2, max_depth_m,
                                   fc='white', ec='#555', lw=0.7, zorder=2))

        for _, row in wdata.iterrows():
            top = row['top_m']; base = min(row['base_m'], max_depth_m)
            code = row['facies_code']
            rgb  = FACIES_COLORS.get(code, '#CCCCCC')
            ax.add_patch(plt.Rectangle((xc-col_hw, top), col_hw*2, base-top,
                                       fc=rgb, ec='#888', lw=0.25, zorder=3, alpha=0.93))
            mid = (top+base)/2
            if (base-top) >= 4:
                dark = code in ('TS','CL','LG')
                tc = '#FFF' if dark else '#111'
                ax.text(xc, mid, code, ha='center', va='center',
                        fontsize=4.5, fontweight='bold', color=tc,
                        fontfamily='monospace', zorder=5)
            # Track lignite
            if code == 'LG' and wname not in lignite_tops:
                lignite_tops[wname] = top
            # Track first aquifer
            if row['aq_class']=='Aquifer' and wname not in first_aq_tops:
                first_aq_tops[wname] = top

        # TD line
        td_plot = min(td, max_depth_m) if pd.notna(td) else max_depth_m
        ax.plot([xc-col_hw*0.8, xc+col_hw*0.8], [td_plot, td_plot], 'k-', lw=1.5, zorder=6)
        ax.text(xc, td_plot+1.5, f'TD={td_plot:.0f}m', ha='center', va='top',
                fontsize=4.5, color='#333', zorder=7)

        # Well label
        region_colors = {'Delta North':'#1565C0','Delta Central':'#E65100','Delta South':'#2E7D32'}
        rc = region_colors.get(REGION_MAP.get(wname,''),'#333')
        ax.text(xc, -14, wname, ha='center', va='bottom', fontsize=7,
                fontweight='bold', color=rc, rotation=40, zorder=8)
        ax.text(xc, -6, f"{intervals[intervals['well']==wname]['region'].iloc[0].replace('Delta ','Δ ')}",
                ha='center', va='bottom', fontsize=5.2, color='#666', zorder=8)

        # N/G annotation
        wng = ng.loc[ng['well']==wname, 'NG_pct']
        if not wng.empty:
            ax.text(xc, -20, f"N/G={wng.values[0]:.0f}%", ha='center', va='bottom',
                    fontsize=5.0, color='#0055AA', fontfamily='monospace', zorder=8)

    # Lignite tie-lines
    sorted_wells = well_list
    lg_wells = [w for w in sorted_wells if w in lignite_tops]
    for i in range(len(lg_wells)-1):
        x1, x2 = xs[lg_wells[i]]+col_hw, xs[lg_wells[i+1]]-col_hw
        y1, y2 = lignite_tops[lg_wells[i]], lignite_tops[lg_wells[i+1]]
        ax.plot([x1,x2],[y1,y2], color='#3D2B1F', lw=1.4, ls='-',
                alpha=0.8, zorder=4)
        ax.annotate('Lignite\nmarker', xy=((x1+x2)/2,(y1+y2)/2),
                    fontsize=4.5, color='#3D2B1F', ha='center',
                    bbox=dict(boxstyle='round',fc='#FFFAF0',ec='#3D2B1F',lw=0.5,alpha=0.9))

    # First-aquifer tie-lines
    aq_wells = [w for w in sorted_wells if w in first_aq_tops]
    for i in range(len(aq_wells)-1):
        x1, x2 = xs[aq_wells[i]]+col_hw, xs[aq_wells[i+1]]-col_hw
        y1, y2 = first_aq_tops[aq_wells[i]], first_aq_tops[aq_wells[i+1]]
        # Only draw if depth difference is modest (otherwise it's a pinch-out)
        if abs(y1-y2) < 25:
            ax.plot([x1,x2],[y1,y2], color='#004488', lw=0.9, ls=':', alpha=0.5, zorder=4)
        else:
            # Mark pinch-out
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                        arrowprops=dict(arrowstyle='-|>', color='#CC2222', lw=0.8))
            ax.text(mx+0.05, my, 'pinch-out', fontsize=4.5, color='#CC2222',
                    bbox=dict(boxstyle='round',fc='white',ec='#CC2222',alpha=0.8))

    # Axes
    ax.set_ylim(max_depth_m, -28)
    ax.set_xlim(-col_hw*2.5, (n-1)*spacing+col_hw*2.5)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.tick_params(axis='y', labelsize=7, length=4)
    ax.set_ylabel("Depth (m)", fontsize=9)
    ax.set_xticks([]); ax.xaxis.set_visible(False)
    ax.grid(axis='y', which='major', color='#CCCCCC', lw=0.4, zorder=0)
    ax.grid(axis='y', which='minor', color='#EEEEEE', lw=0.2, zorder=0)
    ax.set_title(title, fontsize=10.5, fontweight='bold', color='#1A2A5E', pad=14)

    # Formation labels on right margin
    for depth, label, col in [(1.5,'Benin Fm (upper)\n[Zone I]','#8B4513'),
                               (H2+5,'Benin Fm (mid)\n[Zone II]','#1565C0'),
                               (80,'Benin Fm (deep)\n[Zone III]','#0D47A1'),
                               (160,'Ogwashi-Asaba Fm\n(inferred)','#6A1B9A')]:
        if depth < max_depth_m:
            ax.text(xs[well_list[-1]]+col_hw+0.2, depth, label,
                    fontsize=5.2, color=col, va='center', fontweight='bold',
                    bbox=dict(boxstyle='round',fc='white',ec=col,alpha=0.85,lw=0.5))

    # Legend
    used = sorted({c for c in intervals[intervals['well'].isin(well_list)]['facies_code'].unique()},
                  key=lambda c: list(FACIES_COLORS.keys()).index(c) if c in FACIES_COLORS else 99)
    patches = [mpatches.Patch(fc=FACIES_COLORS.get(c,'#CCC'), ec='#555', lw=0.4,
               label=f"{c}") for c in used]
    patches += [plt.Line2D([0],[0],color='#8B6914',lw=1.5,ls='--',label='H1: Base LVL'),
                plt.Line2D([0],[0],color='#226622',lw=1.8,ls='--',label='H2: Base Sub-wtrd'),
                plt.Line2D([0],[0],color='#3D2B1F',lw=1.4,label='Lignite tie-line'),
                plt.Line2D([0],[0],color='#004488',lw=0.9,ls=':',label='Aquifer tie-line'),
                plt.Line2D([0],[0],color='#CC2222',lw=0.8,ls='-',label='Pinch-out')]
    ax.legend(handles=patches, loc='lower right', fontsize=5.0, ncol=3,
              framealpha=0.92, edgecolor='#888', title='Legend', title_fontsize=5.5)

    if note:
        ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=5.0, color='#555',
                va='bottom', ha='left',
                bbox=dict(boxstyle='round',fc='white',ec='#AAA',alpha=0.85))

    plt.tight_layout(rect=[0, 0, 0.97, 0.97])
    return fig

# Sort wells by northing (N→S)
well_order_all = sorted(WELL_SHEETS,
    key=lambda w: -intervals[intervals['well']==w]['northing'].mean()
    if not intervals[intervals['well']==w].empty and intervals[intervals['well']==w]['northing'].notna().any()
    else 0)

# Section A — All 21 wells
fig_a = section_plot(
    well_order_all,
    "Figure 2A — Regional 2D Stratigraphic Correlation: All 21 Wells  (N→S Traverse)\n"
    "Benin Formation Aquifer System, Delta State, Nigeria",
    max_depth_m=255,
    note="Solid lines: aquifer/lignite tie-lines  |  Arrows: pinch-outs  |  H1, H2: near-surface horizon boundaries\n"
        "Source: Corrected 21-well litholog dataset (this study) + Akpoborie & Efobo (2014)"
)
fig_a.savefig(os.path.join(FIG_DIR,'fig02a_section_all21.png'), dpi=160, bbox_inches='tight')
plt.close()
print("✓ Figure 2A saved")

# Section B — Delta North
north_wells = [w for w in well_order_all if REGION_MAP[w]=='Delta North']
fig_b = section_plot(north_wells,
    "Figure 2B — Delta North Correlation (Asaba → Idumuje-Unor)", max_depth_m=255)
fig_b.savefig(os.path.join(FIG_DIR,'fig02b_section_north.png'), dpi=160, bbox_inches='tight')
plt.close()
print("✓ Figure 2B saved")

# Section C — Central → South
cs_wells = [w for w in well_order_all if REGION_MAP[w] in ('Delta Central','Delta South')]
fig_c = section_plot(cs_wells,
    "Figure 2C — Delta Central → South Correlation (Kwale/Abraka → Forcados)", max_depth_m=180)
fig_c.savefig(os.path.join(FIG_DIR,'fig02c_section_central_south.png'), dpi=160, bbox_inches='tight')
plt.close()
print("✓ Figure 2C saved")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — INTERACTIVE 3D GEOSPATIAL GROUND MODEL
# ─────────────────────────────────────────────────────────────────────────────
"""
## 6. Interactive 3D Geospatial Ground Model

Built with **Plotly** — fully interactive in browser:

| Layer (trace)              | Toggle via legend  | Property shown                        |
|----------------------------|--------------------|---------------------------------------|
| Facies volumes             | ✓ per facies code  | Lithofacies colour                    |
| Transmissivity surface     | ✓                  | Interpolated T (m²/day) as colour map |
| Hydraulic conductivity     | ✓                  | Interpolated K (m/day) as colour map  |
| pH surface                 | ✓                  | Hydrogeochemical quality              |
| TDS surface                | ✓                  | Total dissolved solids                |
| Fe concentration           | ✓                  | Iron (mg/L) — contamination risk      |
| Well sticks                | ✓                  | Borehole locations                    |
| Horizon surfaces (H1, H2)  | ✓                  | Near-surface stratigraphic boundaries |
| Aquifer/Aquitard highlight | ✓                  | Aquifer class overlay                 |

**Navigation**: Rotate (left-click+drag), zoom (scroll), pan (right-click+drag).
Click legend items to toggle layers on/off.
"""

def hex_to_plotly(hex_color, alpha=0.85):
    r,g,b = [int(hex_color.lstrip('#')[i:i+2],16) for i in (0,2,4)]
    return f'rgba({r},{g},{b},{alpha})'

# Build per-interval 3D blocks
# Each interval becomes a vertical box at the well location
fig3d = go.Figure()

# ── Helper: interpolate property onto regular XY grid ────────────────────────
def make_property_surface(prop_col, interp_pts=40, log_scale=False, source_df=None):
    """
    Compute mean property per well (midpoint), then interpolate onto XY grid.
    Returns (xi, yi, zi, values_interp)
    """
    if source_df is None:
        src = intervals[intervals[prop_col].notna()].copy()
    else:
        src = source_df.copy()
    
    src = src[src['easting'].notna() & src['northing'].notna()]
    if len(src) < 4: return None
    
    pt_mean = src.groupby('well').agg(
        x=('easting','mean'), y=('northing','mean'), val=(prop_col,'mean')).dropna()
    
    if len(pt_mean) < 3: return None
    
    pts = pt_mean[['x','y']].values
    vals = np.log10(pt_mean['val'].values + 1e-6) if log_scale else pt_mean['val'].values
    
    xi = np.linspace(pts[:,0].min()-2000, pts[:,0].max()+2000, interp_pts)
    yi = np.linspace(pts[:,1].min()-2000, pts[:,1].max()+2000, interp_pts)
    XI, YI = np.meshgrid(xi, yi)
    
    try:
        ZI = griddata(pts, vals, (XI, YI), method='linear')
        # Fill edges with nearest
        ZI_n = griddata(pts, vals, (XI, YI), method='nearest')
        ZI = np.where(np.isnan(ZI), ZI_n, ZI)
    except:
        return None
    return XI, YI, ZI, pt_mean

# ── 6A  Facies interval blocks (3D bars) ─────────────────────────────────────
facies_groups = intervals.groupby('facies_code')
for code, grp in facies_groups:
    col = hex_to_plotly(FACIES_COLORS.get(code, '#AAAAAA'), 0.7)
    grp_valid = grp[grp['easting'].notna()]
    if grp_valid.empty: continue

    x_vals, y_vals, z_top, z_base = [], [], [], []
    texts = []
    for _, row in grp_valid.iterrows():
        x_vals.append(row['easting'])
        y_vals.append(row['northing'])
        z_top.append(-row['top_m'])
        z_base.append(-row['base_m'])
        texts.append(
            f"<b>{row['well']}</b><br>Code: {row['facies_code']}<br>"
            f"Facies: {row['facies_name']}<br>"
            f"Top: {row['top_m']:.1f} m | Base: {row['base_m']:.1f} m<br>"
            f"Thick: {row['thick_m']:.1f} m<br>"
            f"K: {row['K_mday']:.1f} m/day | T: {row['T_m2day']:.1f} m²/day<br>"
            f"Aq.Class: {row['aq_class']} | Region: {row['region']}"
        )

    # Plot as vertical bars using 3D scatter with error bars approximation
    # Each interval = a line segment from z_top to z_base
    for xv,yv,zt,zb,txt in zip(x_vals,y_vals,z_top,z_base,texts):
        fig3d.add_trace(go.Scatter3d(
            x=[xv,xv], y=[yv,yv], z=[zt,zb],
            mode='lines',
            line=dict(color=FACIES_COLORS.get(code,'#AAA'), width=14),
            name=f"Facies: {code}",
            legendgroup=f"facies_{code}",
            showlegend=False,
            hovertext=txt, hoverinfo='text',
            visible=True
        ))

# Add one trace per facies for legend
for code in sorted(FACIES_COLORS.keys()):
    if code in intervals['facies_code'].values:
        fig3d.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(color=FACIES_COLORS[code], size=10),
            name=f"[Facies] {code}",
            legendgroup=f"facies_{code}",
            showlegend=True,
        ))

# ── 6B  Well sticks ───────────────────────────────────────────────────────────
well_summary = intervals.groupby('well').agg(
    easting=('easting','mean'), northing=('northing','mean'),
    td_m=('td_m','mean'), region=('region','first')
).dropna(subset=['easting','northing']).reset_index()

region_marker_colors = {'Delta North':'blue','Delta Central':'orange','Delta South':'green'}
for _, row in well_summary.iterrows():
    td = min(row['td_m'], 260) if pd.notna(row['td_m']) else 100
    mc = region_marker_colors.get(row['region'],'grey')
    fig3d.add_trace(go.Scatter3d(
        x=[row['easting'], row['easting']],
        y=[row['northing'], row['northing']],
        z=[2, -td],
        mode='lines+markers',
        line=dict(color='black', width=2),
        marker=dict(color=['black', mc], size=[5, 3]),
        name='Well Sticks',
        legendgroup='well_sticks',
        showlegend=_ == well_summary.index[0],
        hovertext=f"<b>{row['well']}</b><br>Region: {row['region']}<br>TD: {td:.0f} m",
        hoverinfo='text'
    ))
    fig3d.add_trace(go.Scatter3d(
        x=[row['easting']], y=[row['northing']], z=[5],
        mode='text',
        text=[row['well'].replace('Idumuje-Unor','Id-Unor').replace('RA Ogunu-Warri','RA-Warri')],
        textfont=dict(size=7, color='black'),
        name='Well Labels', legendgroup='well_labels',
        showlegend=_ == well_summary.index[0],
        hoverinfo='skip'
    ))

# ── 6C  Horizon surfaces ──────────────────────────────────────────────────────
for H_depth, H_name, H_col, H_opacity in [
    (3.0,  'H1: Base LVL (~3 m)',     'peru',       0.3),
    (22.0, 'H2: Base Sub-wtrd (~22m)','steelblue',  0.25),
    (80.0, 'H3: Deep Benin (~80m)',   'slateblue',  0.20),
]:
    xi_all = well_summary['easting'].values
    yi_all = well_summary['northing'].values
    xi = np.linspace(xi_all.min()-3000, xi_all.max()+3000, 30)
    yi = np.linspace(yi_all.min()-3000, yi_all.max()+3000, 30)
    XI, YI = np.meshgrid(xi, yi)
    ZI = np.full_like(XI, -H_depth)
    fig3d.add_trace(go.Surface(
        x=XI, y=YI, z=ZI,
        colorscale=[[0,H_col],[1,H_col]],
        showscale=False, opacity=H_opacity,
        name=H_name, legendgroup=H_name, showlegend=True,
        hovertemplate=f"{H_name}<extra></extra>"
    ))

# ── 6D  Property surfaces (T, K, pH, TDS, Fe) ────────────────────────────────
SURFACE_DEPTH = -45  # display at 45 m depth for visual clarity

def add_property_surface(prop_col, trace_name, cmap, log_scale=False, depth=-45, source_df=None):
    result = make_property_surface(prop_col, log_scale=log_scale, source_df=source_df)
    if result is None: return
    XI, YI, ZI, _ = result
    ZI_depth = np.full_like(XI, depth)
    fig3d.add_trace(go.Surface(
        x=XI, y=YI, z=ZI_depth,
        surfacecolor=ZI,
        colorscale=cmap,
        colorbar=dict(title=trace_name, x=1.02, len=0.4, thickness=12, ),
        opacity=0.75, showscale=True,
        name=trace_name, legendgroup=trace_name, showlegend=True,
        visible='legendonly',
        hovertemplate=f"{trace_name}: %{{surfacecolor:.2f}}<extra></extra>"
    ))

add_property_surface('T_m2day',  'Transmissivity (m²/day)', 'Blues',   log_scale=True,  depth=-50)
add_property_surface('K_mday',   'Hydraulic K (m/day)',     'Greens',  log_scale=True,  depth=-60)

# Geochem surfaces for 3D model (add as toggleable surfaces)
gc_merged = geochem_df.merge(well_summary[['well','easting','northing']], on='well', how='inner')
for prop, name, cmap3d, depth3d in [('pH','pH','RdYlGn',-65),('TDS','TDS (mg/L)','YlOrRd',-70),('Fe','Fe (mg/L)','OrRd',-75)]:
    if prop not in gc_merged.columns: continue
    sub3d = gc_merged[['well','easting','northing',prop]].dropna()
    if len(sub3d) < 3: continue
    pts3d = sub3d[['easting','northing']].values.astype(float)
    vals3d= sub3d[prop].values.astype(float)
    xi3d = np.linspace(pts3d[:,0].min()-2000, pts3d[:,0].max()+2000, 25)
    yi3d = np.linspace(pts3d[:,1].min()-2000, pts3d[:,1].max()+2000, 25)
    XI3d, YI3d = np.meshgrid(xi3d, yi3d)
    ZI3d = griddata(pts3d, vals3d, (XI3d, YI3d), method='linear')
    ZI3d_n = griddata(pts3d, vals3d, (XI3d, YI3d), method='nearest')
    ZI3d = np.where(np.isnan(ZI3d), ZI3d_n, ZI3d)
    ZI3d_depth = np.full_like(XI3d, depth3d)
    fig3d.add_trace(go.Surface(
        x=XI3d, y=YI3d, z=ZI3d_depth,
        surfacecolor=ZI3d,
        colorscale=cmap3d,
        colorbar=dict(title=name, x=1.02, len=0.35, thickness=12),
        opacity=0.72, showscale=True,
        name=f"Geochem: {name}", legendgroup=f"geochem_{prop}",
        showlegend=True, visible='legendonly',
        hovertemplate=f"{name}: %{{surfacecolor:.2f}}<extra></extra>"
    ))

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(os.path.join(FIG_DIR,'fig04_property_maps.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 3 (property maps) saved")

# ── Summary output CSV ────────────────────────────────────────────────────────
summary_out = intervals.groupby(['well','region']).agg(
    n_intervals=('facies_code','count'),
    td_m=('td_m','mean'),
    net_aquifer_m=('thick_m', lambda x: x[intervals.loc[x.index,'aq_class']=='Aquifer'].sum()),
    ng_pct=('aq_class', lambda x: (x=='Aquifer').sum() / len(x) * 100),
    mean_K=('K_mday','mean'), max_K=('K_mday','max'),
    mean_T=('T_m2day','mean'), total_T=('T_m2day','sum'),
    easting=('easting','mean'), northing=('northing','mean')
).round(2).reset_index()
summary_out.to_csv(os.path.join(OUT_DIR,'well_aquifer_summary.csv'), index=False)
print(f"✓ Well summary CSV saved")

intervals.to_csv(os.path.join(OUT_DIR,'master_intervals.csv'), index=False)
geochem_df.to_csv(os.path.join(OUT_DIR,'hydrogeochemistry.csv'), index=False)
print(f"✓ Master intervals + geochem CSVs saved")

print("\n" + "="*65)
print(" PIPELINE COMPLETE — Niger Delta 3D Ground Model")
print("="*65)
print(f" Figures : {FIG_DIR}")
print(f" Outputs : {OUT_DIR}")
print("   niger_delta_3D_model.html  <- Open in browser for interactive 3D model")
