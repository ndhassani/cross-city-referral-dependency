"""
02_referral_dependency_analysis.py
==================================
All tables, figures and sensitivity analyses of the paper
"Cross-City Referral Dependency in Saudi Arabia: A Diagnosis-Level, City-to-City Analysis of
Referral Volume and Destination".

Input    data/processed/referrals_analysis.csv    (written by 01_data_preparation.py)
         data/processed/referrals_quarterly.csv   (quarter-level file; for Tables S36-S38, Figure S18)
Output   reports/analysis/tables/*.csv  +  reports/analysis/all_tables.xlsx  (one sheet per table)
         reports/analysis/figures/*.png (300 dpi)
         reports/analysis/key_numbers.json  (every number quoted in the text)

Sections
  1. Descriptive statistics           Table 1, Tables S1-S5, Figures S2-S3
  2. Cross-city dependency index      Table 2, Tables S6-S8, S33-S34, Figures 2-3
  3. Multivariable model (main)       Tables 3-4, S12-S13, Figure 1
  4. Sensitivity analyses             Tables S9-S11; quarter-level and data-quality: Tables S36-S38, Figure S18
  5. Hubs                             Table 5, Tables S21, S35, Figures S8-S10
  6. Exploratory cluster analyses     Table 6, Tables S14-S20, S22-S31, Figures 4-5, S4-S7, S11-S17
  7. Maps                             Figure 6, Table S32

Run:     python 02_referral_dependency_analysis.py
         options: --input PATH  --quarterly-input PATH  --out-dir PATH  --min-same-city 100
                  --n-permutations 100  --n-bootstrap 200  --skip-clustering
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from icd10am_chapters import CHAPTER_ORDER, CHAPTER_SHORT

warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42

AGE_GROUPS = ["00-17", "18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
MODEL_CHAPTERS = [c for c in CHAPTER_ORDER if c != "OTHER"]       # 21 chapters + supplementary codes
HUBS = ["Riyadh", "Jeddah", "Makkah", "Asir", "Madinah"]          # destination block of Analysis B
QUARTER_LABEL = lambda q: f"{q[5:]} {q[:4]}"                      # "2025-Q4" -> "Q4 2025"

# Coordinates used for mapping (Table S32): regions at their administrative capital; approximate.
COORDS = {
    "Al-Ahsa": (25.3833, 49.5867, "Al-Hofuf (main city of Al-Ahsa; unit includes the former Eastern directorate)"),
    "Al-Baha": (20.0129, 41.4677, "Al-Baha city (regional capital)"),
    "Al-Jouf": (29.9697, 40.2064, "Sakaka (regional capital)"),
    "Asir": (18.2164, 42.5053, "Abha (regional capital)"),
    "Bisha": (19.9977, 42.6052, "City"),
    "Hafr Al-Batin": (28.4328, 45.9708, "City"),
    "Hail": (27.5114, 41.7208, "Hail city (regional capital)"),
    "Jazan": (16.8892, 42.5511, "Jazan city (regional capital)"),
    "Jeddah": (21.4858, 39.1925, "City"),
    "Madinah": (24.5247, 39.5692, "City"),
    "Makkah": (21.3891, 39.8579, "City"),
    "Najran": (17.5656, 44.2289, "Najran city (regional capital)"),
    "Northern Borders": (30.9753, 41.0381, "Arar (regional capital)"),
    "Qassim": (26.3260, 43.9750, "Buraydah (regional capital)"),
    "Qunfudah": (19.1264, 41.0789, "City (Al-Qunfudhah)"),
    "Qurayyat": (31.3318, 37.3428, "City (Al-Qurayyat)"),
    "Riyadh": (24.7136, 46.6753, "City"),
    "Tabuk": (28.3838, 36.5550, "Tabuk city (regional capital)"),
    "Taif": (21.2703, 40.4158, "City"),
}

# Label offsets (points) so that neighbouring city names on the maps do not overlap
LABEL_OFFSET = {"Jeddah": (-38, 4), "Makkah": (5, 5), "Taif": (5, -9), "Al-Baha": (-36, 3), "Bisha": (5, 3),
                "Qurayyat": (4, 4), "Northern Borders": (4, 4)}

# Colours (colour-blind safe categorical palette; blue/orange diverging)
CAT = ["#2a78d6", "#e8873a", "#3aa57a", "#c4479b", "#8a6bd1", "#b5892c", "#4fa3c7", "#d6574a"]
BLUE, ORANGE, GREY = "#2a78d6", "#e8873a", "#9a9a9a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})

TABLES, KEY = {}, {}


# ==============================================================================================
# Helpers
# ==============================================================================================
def chapter_label(code):
    return {"SUPP": "Supplementary codes", "OTHER": "Other (special/unclassified)"}.get(
        code, f"{code} {CHAPTER_SHORT.get(code, code)}")


def wilson(x, n):
    lo, hi = proportion_confint(np.asarray(x), np.asarray(n), method="wilson")
    return lo * 100, hi * 100


def cramers_v_bc(table):
    """Chi-square test and bias-corrected Cramer's V (Bergsma 2013) for a contingency table."""
    t = np.asarray(table, dtype=float)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    chi2, p, df, expected = stats.chi2_contingency(t, correction=False)
    n = t.sum(); r, k = t.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    v = np.sqrt(phi2 / min(kc - 1, rc - 1)) if min(kc - 1, rc - 1) > 0 else np.nan
    return {"n": int(n), "chi2": chi2, "df": int(df), "p": p, "V": v, "pct_expected_lt5": (expected < 5).mean() * 100}


def fmt_p(p):
    if p == 0 or p < 1e-300:
        return "<1e-300"
    return f"{p:.2g}" if p < 0.001 else f"{p:.3f}"


def hhi(shares_pct):
    s = np.asarray(shares_pct) / 100
    return float((s ** 2).sum() * 10000)


def di_table(df, by):
    """Cross-city dependency (cross-city referrals / all referrals) per group, with Wilson CI."""
    g = df.groupby(by).agg(n=("REFERRALS", "sum"))
    g["cross"] = df[df.CROSS_CITY == 1].groupby(by).REFERRALS.sum().reindex(g.index, fill_value=0)
    g["same"] = g.n - g.cross
    g["DI"] = g.cross / g.n * 100
    g["lo"], g["hi"] = wilson(g.cross, g.n)
    return g


def save_table(name, df, index=False):
    TABLES[name] = df.reset_index() if index else df
    return df


def savefig(fig, out, name):
    fig.savefig(out / f"{name}.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ==============================================================================================
# 1. Descriptive statistics
# ==============================================================================================
def descriptives(a, out):
    tot = a.REFERRALS.sum()
    rows = {}
    for t, g in a.groupby("REFERRAL_TYPE"):
        rows[t] = {
            "Referrals, n (%)": f"{g.REFERRALS.sum():,} ({g.REFERRALS.sum() / tot * 100:.1f})",
            "Code-cell records, n": f"{len(g):,}",
            "Distinct ICD-10 codes, n": f"{g.DISEASE_CODE.nunique():,}",
            "Referring cities, n": g.REFERRING_CITY.nunique(),
            "Receiving cities, n": g.RECEIVING_CITY.nunique(),
            "Referrals per record, median (IQR)": f"{g.REFERRALS.median():.0f} ({g.REFERRALS.quantile(.25):.0f}"
                                                  f"–{g.REFERRALS.quantile(.75):.0f})",
            "Repeated records, %": f"{g.REPEATED.mean() * 100:.1f}",
        }
    t1 = pd.DataFrame(rows)[["Same-city", "Cross-city"]].rename_axis("Characteristic")
    save_table("Table1_volume_by_type", t1, index=True)
    cross = a[a.CROSS_CITY == 1].REFERRALS.sum()
    KEY.update(total_referrals=int(tot), cross_city_referrals=int(cross), cross_city_share_pct=cross / tot * 100,
               median_per_record_same=float(a[a.CROSS_CITY == 0].REFERRALS.median()),
               median_per_record_cross=float(a[a.CROSS_CITY == 1].REFERRALS.median()),
               repeated_same_pct=a[a.CROSS_CITY == 0].REPEATED.mean() * 100,
               repeated_cross_pct=a[a.CROSS_CITY == 1].REPEATED.mean() * 100)

    # Table S1 - integrity checks
    grp = ["REFERRAL_TYPE", "REFERRING_CITY", "RECEIVING_CITY", "AGE_GROUP"]
    gsize = a.groupby(grp).DISEASE_CODE.transform("size")
    gshare = a.groupby(grp)["SHARE_IN_GROUP_%"].transform("sum")
    s1 = pd.DataFrame([
        ("Rows", f"{len(a):,}", ""),
        ("Duplicate rows (same code in the same cell)", int(a.duplicated(grp + ["DISEASE_CODE"]).sum()), "Expected 0"),
        ("REPEATED equals (N_OCCURRENCES > 1)", f"{(a.REPEATED == (a.N_OCCURRENCES > 1)).mean() * 100:.1f}%",
         "Confirms derived flag"),
        ("N_OCCURRENCES <= N_QUARTERS x N_GENDERS", f"{(a.N_OCCURRENCES <= a.N_QUARTERS * a.N_GENDERS).mean() * 100:.1f}%",
         "Internal consistency"),
        ("Same-city rows with referring = receiving city",
         f"{(a.loc[a.CROSS_CITY == 0, 'REFERRING_CITY'] == a.loc[a.CROSS_CITY == 0, 'RECEIVING_CITY']).mean() * 100:.1f}%", ""),
        ("Cross-city rows with referring != receiving city",
         f"{(a.loc[a.CROSS_CITY == 1, 'REFERRING_CITY'] != a.loc[a.CROSS_CITY == 1, 'RECEIVING_CITY']).mean() * 100:.1f}%", ""),
        ("Groups where rows = CODES_IN_GROUP", f"{(gsize == a.CODES_IN_GROUP).mean() * 100:.1f}%", "Each group is complete"),
        ("Groups with SHARE_IN_GROUP_% summing to 100 +/- 1", f"{((gshare - 100).abs() <= 1).mean() * 100:.1f}%", "Rounding only"),
        ("Missing values", int(a.isna().sum().sum()), ""),
    ], columns=["Check", "Result", "Comment"])
    save_table("TableS1_integrity_checks", s1)

    s2 = pd.DataFrame([
        ("Rows (code x cell records)", f"{len(a):,}"),
        ("Distinct ICD-10 codes", f"{a.DISEASE_CODE.nunique():,}"),
        ("ICD-10 chapters", f"{a.ICD_CHAPTER.nunique()} (incl. merged \"Other\")"),
        ("Referring cities", a.REFERRING_CITY.nunique()),
        ("Receiving cities", a.RECEIVING_CITY.nunique()),
        ("Age groups", a.AGE_GROUP.nunique()),
        ("Total referrals (sum of REFERRALS)", f"{tot:,}"),
    ], columns=["Item", "Value"])
    save_table("TableS2_dataset_overview", s2)

    # Tables S3 (age) and S4 (chapter)
    def by_type(col, order, label):
        t = a.pivot_table(index=col, columns="REFERRAL_TYPE", values="REFERRALS", aggfunc="sum",
                          fill_value=0).reindex(order, fill_value=0)
        res = pd.DataFrame({label: [chapter_label(i) if col == "ICD_CHAPTER" else i for i in t.index],
                            "Same-city, n": t["Same-city"].values,
                            "Same-city, % of column": (t["Same-city"] / t["Same-city"].sum() * 100).round(1).values,
                            "Cross-city, n": t["Cross-city"].values,
                            "Cross-city, % of column": (t["Cross-city"] / t["Cross-city"].sum() * 100).round(1).values})
        n = t.sum(axis=1).values; x = t["Cross-city"].values
        lo, hi = wilson(x, n)
        res["Cross-city share, %"] = np.round(x / n * 100, 2)
        res["95% CI"] = [f"{l:.2f}–{h:.2f}" for l, h in zip(lo, hi)]
        return res
    s3 = save_table("TableS3_age_group", by_type("AGE_GROUP", AGE_GROUPS, "Age group"))
    s4 = save_table("TableS4_chapter", by_type("ICD_CHAPTER", CHAPTER_ORDER, "ICD-10 chapter"))

    # Table S5 - chi-square tests with bias-corrected Cramer's V
    tests = [("Referral type x Age group", a, "REFERRAL_TYPE", "AGE_GROUP"),
             ("Referral type x ICD-10 chapter", a, "REFERRAL_TYPE", "ICD_CHAPTER"),
             ("Referral type x Referring city", a, "REFERRAL_TYPE", "REFERRING_CITY"),
             ("Age group x ICD-10 chapter", a, "AGE_GROUP", "ICD_CHAPTER"),
             ("Referring city x ICD-10 chapter (same-city)", a[a.CROSS_CITY == 0], "REFERRING_CITY", "ICD_CHAPTER"),
             ("Referring city x ICD-10 chapter (cross-city)", a[a.CROSS_CITY == 1], "REFERRING_CITY", "ICD_CHAPTER"),
             ("Referring city x Receiving city (cross-city)", a[a.CROSS_CITY == 1], "REFERRING_CITY", "RECEIVING_CITY")]
    s5 = []
    for name, d, r, c in tests:
        res = cramers_v_bc(d.pivot_table(index=r, columns=c, values="REFERRALS", aggfunc="sum", fill_value=0))
        s5.append({"Association": name, "N (referrals)": f"{res['n']:,}", "Chi-square": round(res["chi2"], 1),
                   "df": res["df"], "p": fmt_p(res["p"]), "Cramer's V (bias-corrected)": round(res["V"], 3),
                   "Cells with expected < 5, %": round(res["pct_expected_lt5"], 1)})
    save_table("TableS5_chi_square", pd.DataFrame(s5))
    KEY["cramers_v_type_city_all"] = s5[2]["Cramer's V (bias-corrected)"]

    # crude shares quoted in the text
    ch = di_table(a, "ICD_CHAPTER"); ag = di_table(a, "AGE_GROUP")
    KEY["crude_chapter_share_pct"] = {chapter_label(k): round(v, 1) for k, v in ch.DI.items()}
    KEY["crude_age_share_pct"] = ag.DI.round(1).to_dict()

    # Figure S2 - age group
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(AGE_GROUPS)); w = 0.4
    ax[0].bar(x - w / 2, s3["Same-city, n"] / 1000, w, color=BLUE, label="Same-city")
    ax[0].bar(x + w / 2, s3["Cross-city, n"] / 1000, w, color=ORANGE, label="Cross-city")
    ax[0].set_xticks(x, AGE_GROUPS); ax[0].set_ylabel("Referrals (thousands)"); ax[0].legend(frameon=False)
    ax[0].set_title("(a) Referral volume by age group", loc="left")
    agd = ag.reindex(AGE_GROUPS)
    ax[1].errorbar(x, agd.DI, yerr=[agd.DI - agd.lo, agd.hi - agd.DI], fmt="o", color=ORANGE, capsize=3)
    ax[1].axhline(KEY["cross_city_share_pct"], ls="--", color=GREY, lw=1)
    ax[1].set_xticks(x, AGE_GROUPS); ax[1].set_ylabel("Cross-city share, % (95% CI)")
    ax[1].set_title("(b) Share referred to another city", loc="left")
    savefig(fig, out, "FigureS2_age_group")

    # Figure S3 - chapters
    chd = ch.drop("OTHER", errors="ignore").sort_values("DI")
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(chd))
    ax.barh(y, chd.DI, color=ORANGE, height=0.7)
    ax.errorbar(chd.DI, y, xerr=[chd.DI - chd.lo, chd.hi - chd.DI], fmt="none", color="#333", capsize=2, lw=0.8)
    ax.axvline(KEY["cross_city_share_pct"], ls="--", color=GREY, lw=1)
    ax.set_yticks(y, [chapter_label(i) for i in chd.index]); ax.set_xlabel("Cross-city share, % (95% Wilson CI)")
    for yi, (v, n) in enumerate(zip(chd.DI, chd.n)):
        ax.text(v + 1, yi, f"n={n:,}", va="center", fontsize=7, color="#555")
    savefig(fig, out, "FigureS3_chapter_cross_city_share")


# ==============================================================================================
# 2. Cross-city dependency index
# ==============================================================================================
def dependency(a, out, min_same):
    tot_x = a[a.CROSS_CITY == 1].REFERRALS.sum(); tot = a.REFERRALS.sum(); p0 = tot_x / tot
    d = di_table(a, "REFERRING_CITY")
    d["p_binom"] = [stats.binomtest(int(x), int(n), p0).pvalue for x, n in zip(d.cross, d.n)]
    d["q_BH"] = multipletests(d.p_binom, method="fdr_bh")[1]
    xc = a[a.CROSS_CITY == 1]
    dest = xc.groupby(["REFERRING_CITY", "RECEIVING_CITY"]).REFERRALS.sum()
    d["n_dest"] = xc.groupby("REFERRING_CITY").RECEIVING_CITY.nunique().reindex(d.index, fill_value=0)
    main = dest.groupby(level=0).idxmax().map(lambda t: t[1])
    d["main_dest"] = main.reindex(d.index)
    d["main_share"] = (dest.groupby(level=0).max() / d.cross).reindex(d.index) * 100
    d["coverage_adequate"] = d.same >= min_same
    d = d.sort_values(["DI", "n"], ascending=[False, False])
    t2 = pd.DataFrame({
        "Referring city": [c + ("" if ok else " †") for c, ok in zip(d.index, d.coverage_adequate)],
        "Same-city, n": d.same.map("{:,}".format), "Cross-city, n": d.cross.map("{:,}".format),
        "DI, % (95% CI)": [f"{v:.1f} ({l:.1f}–{h:.1f})" for v, l, h in zip(d.DI, d.lo, d.hi)],
        "Destinations, n": d.n_dest,
        "Main destination (share)": [f"{m} ({s:.0f}%)" if isinstance(m, str) else "–" for m, s in zip(d.main_dest, d.main_share)],
        "q (BH), vs national": d.q_BH.map(fmt_p)})
    save_table("Table2_dependency_index", t2)
    ok_cities = d.index[d.coverage_adequate].tolist()
    KEY.update(coverage_adequate_cities=ok_cities, n_coverage_adequate=len(ok_cities),
               coverage_adequate_referrals=int(d.loc[ok_cities, "n"].sum()),
               coverage_adequate_pct=d.loc[ok_cities, "n"].sum() / tot * 100,
               cities_no_same_city=d.index[d.same == 0].tolist(),
               cities_lt_min_same={c: int(s) for c, s in d.same.items() if 0 < s < min_same},
               di_by_city=d.DI.round(1).to_dict())

    # Figure 2
    dd = d.sort_values("DI")
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(dd))
    col = np.where(dd.q_BH >= 0.05, GREY, np.where(dd.DI > p0 * 100, ORANGE, BLUE))
    ax.hlines(y, dd.lo, dd.hi, color=col, lw=2)
    ax.scatter(dd.DI, y, color=col, zorder=3, s=28, edgecolor="white")
    ax.axvline(p0 * 100, ls="--", color=GREY, lw=1)
    ax.text(p0 * 100 + 1, len(dd) - 0.4, f"National {p0 * 100:.1f}%", fontsize=8, color="#555")
    ax.set_yticks(y, [f"{c}{'' if ok else ' †'} (n={n:,})" for c, ok, n in zip(dd.index, dd.coverage_adequate, dd.n)])
    ax.set_xlabel("Cross-city dependency index, % (95% Wilson CI)"); ax.set_xlim(-2, 102)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, label="Above national (BH q < 0.05)"),
                       Line2D([], [], marker="o", ls="", color=BLUE, label="Below national (BH q < 0.05)"),
                       Line2D([], [], marker="o", ls="", color=GREY, label="Not significant")],
              frameon=False, loc="lower right", fontsize=8)
    savefig(fig, out, "Figure2_dependency_index_by_city")

    # City x chapter DI (Tables S33-S34, Figure 3) and heterogeneity within chapters (Table S6)
    cc = di_table(a, ["REFERRING_CITY", "ICD_CHAPTER"])
    order = d.index.tolist()
    s33 = cc.DI.unstack().reindex(index=order, columns=CHAPTER_ORDER).round(1)
    s34 = cc.n.unstack().reindex(index=order, columns=CHAPTER_ORDER)
    s33.columns = s34.columns = [CHAPTER_SHORT[c] for c in CHAPTER_ORDER]
    save_table("TableS33_DI_city_by_chapter", s33.rename_axis("City"), index=True)
    save_table("TableS34_referrals_city_by_chapter", s34.fillna(0).astype(int).rename_axis("City"), index=True)

    fig, ax = plt.subplots(figsize=(11, 6.2))
    h = s33.iloc[:, :-1]                                                # without "Other"
    n = s34.iloc[:, :-1]
    sns.heatmap(h, cmap=LinearSegmentedColormap.from_list("b", ["#f2f6fb", "#9ec5f4", "#2a78d6", "#0d366b"]),
                vmin=0, vmax=100, ax=ax, linewidths=0.4, linecolor="white", annot=True, fmt=".0f",
                annot_kws={"fontsize": 6.5}, cbar_kws={"label": "Cross-city dependency index, %"})
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            if 0 < (n.iat[i, j] if pd.notna(n.iat[i, j]) else 0) < 20:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=False, ec="#d6574a", lw=1.1))
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_yticklabels([c + ("" if c in ok_cities else " †") for c in h.index], rotation=0)
    savefig(fig, out, "Figure3_DI_city_by_chapter")

    s6 = []
    for ch in MODEL_CHAPTERS:
        g = a[a.ICD_CHAPTER == ch]
        tab = g.pivot_table(index="REFERRING_CITY", columns="CROSS_CITY", values="REFERRALS", aggfunc="sum", fill_value=0)
        res = cramers_v_bc(tab)
        gok = g[g.REFERRING_CITY.isin(ok_cities)]
        res_ok = cramers_v_bc(gok.pivot_table(index="REFERRING_CITY", columns="CROSS_CITY", values="REFERRALS",
                                              aggfunc="sum", fill_value=0))
        cdi = cc.loc[(slice(None), ch), "DI"]
        s6.append({"ICD-10 chapter": chapter_label(ch), "Referrals, n": int(g.REFERRALS.sum()),
                   "Cross-city share, %": round(g.loc[g.CROSS_CITY == 1, "REFERRALS"].sum() / g.REFERRALS.sum() * 100, 1),
                   "Range across cities, %": f"{cdi.min():.0f}–{cdi.max():.0f}", "Chi-square": round(res["chi2"], 1),
                   "df": res["df"], "p_raw": res["p"], "Cramer's V": round(res["V"], 3),
                   "Cramer's V, coverage-adequate cities": round(res_ok["V"], 3)})
    s6 = pd.DataFrame(s6)
    s6["p"] = s6.p_raw.map(fmt_p); s6["q (BH-FDR)"] = multipletests(s6.p_raw, method="fdr_bh")[1]
    s6["q (BH-FDR)"] = s6["q (BH-FDR)"].map(fmt_p)
    save_table("TableS6_heterogeneity_within_chapter", s6.drop(columns="p_raw").sort_values("Cramer's V", ascending=False))

    # Tables S7-S8 - codes
    code = a.groupby("DISEASE_CODE").agg(chapter=("ICD_CHAPTER", "first"), total=("REFERRALS", "sum"))
    code["cross"] = xc.groupby("DISEASE_CODE").REFERRALS.sum().reindex(code.index, fill_value=0)
    code["ref_cities"] = xc.groupby("DISEASE_CODE").REFERRING_CITY.nunique().reindex(code.index, fill_value=0)
    code["recv_cities"] = xc.groupby("DISEASE_CODE").RECEIVING_CITY.nunique().reindex(code.index, fill_value=0)
    code["share"] = code.cross / code.total * 100

    def code_table(c):
        return pd.DataFrame({"ICD-10 code": c.index, "Chapter": c.chapter.map(chapter_label).values,
                             "Total referrals": c.total.values, "Cross-city referrals": c.cross.values,
                             "Referring cities sending cross-city": c.ref_cities.values,
                             "Receiving cities (cross-city)": c.recv_cities.values, "Cross-city share, %": c.share.round(1).values})
    save_table("TableS7_top20_codes_cross_city_volume", code_table(code.sort_values("cross", ascending=False).head(20)))
    save_table("TableS8_top20_codes_cross_city_share",
               code_table(code[code.total >= 50].sort_values(["share", "cross"], ascending=False).head(20)))
    return d


# ==============================================================================================
# 3. Multivariable model (quasi-binomial)
# ==============================================================================================
def cells(df, by):
    g = df.groupby(by).agg(n=("REFERRALS", "sum")).reset_index()
    x = df[df.CROSS_CITY == 1].groupby(by).REFERRALS.sum().rename("x").reset_index()
    g = g.merge(x, on=by, how="left").fillna({"x": 0})
    g["p"] = g.x / g.n
    return g


def model(a, out, ok_cities):
    d = a[a.REFERRING_CITY.isin(ok_cities) & (a.ICD_CHAPTER != "OTHER")]
    c = cells(d, ["REFERRING_CITY", "ICD_CHAPTER", "AGE_GROUP"])
    terms = {"Referring city": 'C(REFERRING_CITY, Treatment("Riyadh"))',
             "ICD-10 chapter": 'C(ICD_CHAPTER, Treatment("IX"))',
             "Age group": 'C(AGE_GROUP, Treatment("18-29"))'}
    fit = lambda f, data: smf.glm(f, data, family=sm.families.Binomial(), var_weights=data.n).fit()
    full_f = "p ~ " + " + ".join(terms.values())
    m = fit(full_f, c)
    phi = m.pearson_chi2 / m.df_resid
    expl = m.null_deviance - m.deviance
    rows = []
    for name, term in terms.items():
        m2 = fit("p ~ " + " + ".join(t for t in terms.values() if t != term), c)
        dD, ddf = m2.deviance - m.deviance, int(round(m2.df_resid - m.df_resid))
        F = (dD / ddf) / phi
        rows.append({"Term": name, "df": ddf, "Deviance explained": round(dD), "share": dD,
                     "F": round(F, 1), "p_raw": stats.f.sf(F, ddf, m.df_resid)})
    for r in rows:                                     # share of the deviance explained by the full model
        r["Share of explained deviance, %"] = round(r.pop("share") / expl * 100, 1)
    # city x chapter interaction: deviance of the main-effects model fitted to city x chapter cells
    cc = cells(d, ["REFERRING_CITY", "ICD_CHAPTER"])
    mi = fit("p ~ C(REFERRING_CITY) + C(ICD_CHAPTER)", cc)
    m0 = fit("p ~ 1", cc)
    Fi = (mi.deviance / mi.df_resid) / phi
    rows.append({"Term": "City x chapter interaction", "df": int(mi.df_resid), "Deviance explained": round(mi.deviance),
                 "F": round(Fi, 1), "p_raw": stats.f.sf(Fi, mi.df_resid, m.df_resid),
                 "Share of explained deviance, %": round(mi.deviance / m0.deviance * 100, 1)})
    t3 = pd.DataFrame(rows)
    t3["p"] = t3.p_raw.map(fmt_p)
    t3 = t3[["Term", "df", "Deviance explained", "Share of explained deviance, %", "F", "p"]]
    save_table("Table3_deviance_contributions", t3)
    KEY.update(model_cells=len(c), model_dispersion=phi, model_null_deviance=m.null_deviance,
               model_residual_deviance=m.deviance, model_explained_deviance=expl,
               interaction_F=Fi, interaction_df=int(mi.df_resid), interaction_p=float(stats.f.sf(Fi, mi.df_resid, m.df_resid)))

    # odds ratios with quasi-binomial confidence intervals
    mq = smf.glm(full_f, c, family=sm.families.Binomial(), var_weights=c.n).fit(scale="X2")
    ors = pd.DataFrame({"OR": np.exp(mq.params), "lo": np.exp(mq.conf_int()[0]), "hi": np.exp(mq.conf_int()[1]),
                        "p": mq.pvalues})
    def part(term, ref, label):
        r = ors[ors.index.str.startswith(term)].copy()
        r.index = r.index.str.extract(r"\[T\.(.*)\]")[0].values
        r.loc[ref] = [1.0, np.nan, np.nan, np.nan]
        r["Factor"] = label
        return r
    ch = part(terms["ICD-10 chapter"], "IX", "ICD-10 chapter").sort_values("OR", ascending=False)
    ag = part(terms["Age group"], "18-29", "Age group").reindex(["18-29"] + [g for g in AGE_GROUPS if g != "18-29"])
    ci = part(terms["Referring city"], "Riyadh", "Referring city").sort_values("OR", ascending=False)
    fmt = lambda r: "1 (reference)" if pd.isna(r.lo) else f"{r.OR:.2f} ({r.lo:.2f}–{r.hi:.2f})"
    s12 = pd.concat([ch.loc[["IX"]], ch.drop("IX"), ag])
    s12 = pd.DataFrame({"Factor": s12.Factor, "Level": [chapter_label(i) if f == "ICD-10 chapter" else i
                                                        for i, f in zip(s12.index, s12.Factor)],
                        "Adjusted OR (95% CI)": [fmt(r) for _, r in s12.iterrows()],
                        "p": [("" if pd.isna(p) else fmt_p(p)) for p in s12.p]})
    save_table("TableS12_OR_chapter_age", s12)
    ci = pd.concat([ci.loc[["Riyadh"]], ci.drop("Riyadh")])
    save_table("TableS13_OR_city", pd.DataFrame({"Level": ci.index, "Adjusted OR (95% CI)": [fmt(r) for _, r in ci.iterrows()],
                                                 "p": [("" if pd.isna(p) else fmt_p(p)) for p in ci.p]}))
    t4 = ch.drop("IX")
    save_table("Table4_adjusted_OR_chapter", pd.DataFrame(
        {"ICD-10 chapter": [CHAPTER_SHORT[i] for i in list(t4.index) + ["IX"]],
         "Adjusted OR (95% CI)": [fmt(r) for _, r in t4.iterrows()] + ["1 (reference)"]}))
    KEY["adjusted_OR_chapter"] = {CHAPTER_SHORT[i]: [round(r.OR, 2), round(r.lo, 2), round(r.hi, 2)]
                                  for i, r in t4.iterrows()}

    # Figure 1 - forest plot
    f1 = ch.sort_values("OR")
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(f1))
    sig = (f1.lo > 1) | (f1.hi < 1)
    colr = np.where(f1.index == "IX", "#333", np.where(~sig, GREY, np.where(f1.OR > 1, ORANGE, BLUE)))
    ax.hlines(y, f1.lo, f1.hi, color=colr, lw=1.8)
    ax.scatter(f1.OR, y, color=colr, s=30, zorder=3, edgecolor="white")
    ax.axvline(1, color="#333", lw=0.8)
    ax.set_xscale("log"); ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20], ["0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.set_yticks(y, [chapter_label(i) + (" (reference)" if i == "IX" else "") for i in f1.index])
    for yi, (i, r) in enumerate(f1.iterrows()):
        if i != "IX":
            ax.text(40, yi, f"{r.OR:.2f} ({r.lo:.2f}–{r.hi:.2f})", va="center", fontsize=7.5)
    ax.set_xlim(0.12, 160); ax.set_xlabel("Adjusted odds ratio of cross-city referral (95% CI, log scale)")
    savefig(fig, out, "Figure1_adjusted_OR_chapter")


# ==============================================================================================
# 4. Sensitivity analyses
# ==============================================================================================
def headline(x, ref_chapter_share=None):
    tot = x.REFERRALS.sum(); xc = x[x.CROSS_CITY == 1]
    inflow = xc.groupby("RECEIVING_CITY").REFERRALS.sum().sort_values(ascending=False)
    ch = di_table(x[x.ICD_CHAPTER != "OTHER"], "ICD_CHAPTER").DI
    ag = di_table(x, "AGE_GROUP").DI
    res = {"n_cities": x.REFERRING_CITY.nunique(), "n": int(tot), "share": xc.REFERRALS.sum() / tot * 100,
           "top3": inflow.head(3).sum() / inflow.sum() * 100, "top3_names": inflow.index[:3].tolist(),
           "hhi": hhi(inflow / inflow.sum() * 100), "chapters": ch, "top_chapter": ch.idxmax(), "top_chapter_share": ch.max(),
           "top_age": ag.idxmax(), "top_age_share": ag.max()}
    for nm, col in [("V_city", "REFERRING_CITY"), ("V_chapter", "ICD_CHAPTER"), ("V_age", "AGE_GROUP")]:
        res[nm] = cramers_v_bc(x.pivot_table(index="REFERRAL_TYPE", columns=col, values="REFERRALS",
                                             aggfunc="sum", fill_value=0))["V"]
    if ref_chapter_share is not None:
        com = ch.index.intersection(ref_chapter_share.index)
        res["rho_chapter"] = stats.spearmanr(ch[com], ref_chapter_share[com]).statistic
    return res


def sensitivity(a, d_city, ok_cities):
    full = headline(a)
    ok = headline(a[a.REFERRING_CITY.isin(ok_cities)], full["chapters"])
    full["rho_chapter"] = 1.0
    rows = [("Referring cities, n", full["n_cities"], ok["n_cities"]),
            ("Referrals, n", f"{full['n']:,}", f"{ok['n']:,}"),
            ("Cross-city share, %", f"{full['share']:.1f}", f"{ok['share']:.1f}"),
            ("Cramer's V, referral type x referring city", f"{full['V_city']:.2f}", f"{ok['V_city']:.2f}"),
            ("Cramer's V, referral type x ICD-10 chapter", f"{full['V_chapter']:.2f}", f"{ok['V_chapter']:.2f}"),
            ("Cramer's V, referral type x age group", f"{full['V_age']:.2f}", f"{ok['V_age']:.2f}"),
            ("Chapter with highest cross-city share", f"{CHAPTER_SHORT[full['top_chapter']]} ({full['top_chapter_share']:.1f}%)",
             f"{CHAPTER_SHORT[ok['top_chapter']]} ({ok['top_chapter_share']:.1f}%)"),
            ("Spearman rho, chapter ranking vs all cities", "1.00", f"{ok['rho_chapter']:.2f}"),
            ("Age group with highest cross-city share", f"{full['top_age']} ({full['top_age_share']:.1f}%)",
             f"{ok['top_age']} ({ok['top_age_share']:.1f}%)"),
            ("Top-3 hubs (share of cross-city inflow)", f"{', '.join(full['top3_names'])} ({full['top3']:.1f}%)",
             f"{', '.join(ok['top3_names'])} ({ok['top3']:.1f}%)"),
            ("Inflow HHI", f"{full['hhi']:,.0f}", f"{ok['hhi']:,.0f}")]
    save_table("TableS9_coverage_sensitivity", pd.DataFrame(rows, columns=["Measure", "All referring cities",
                                                                           f"{len(ok_cities)} coverage-adequate cities"]))
    KEY.update(coverage_adequate_cross_share_pct=ok["share"], coverage_adequate_rho_chapter=ok["rho_chapter"],
               coverage_adequate_top3_pct=ok["top3"], V_city_all=full["V_city"], V_city_adequate=ok["V_city"],
               top3_hub_share_pct=full["top3"], inflow_hhi=full["hhi"])

    # Table S10 - regional nesting
    xc = a[a.CROSS_CITY == 1]
    region = a.groupby("REFERRING_CITY").REFERRING_REGION.first()
    t = pd.DataFrame({"Region": region.str.replace(" Region", "", regex=False),
                      "cross": xc.groupby("REFERRING_CITY").REFERRALS.sum(),
                      "crossing": xc[xc.CROSS_REGION == 1].groupby("REFERRING_CITY").REFERRALS.sum(),
                      "n": a.groupby("REFERRING_CITY").REFERRALS.sum()}).fillna(0)
    t["Within own region, %"] = ((t.cross - t.crossing) / t.cross * 100).round(1)
    t["Crossing a regional boundary, %"] = (t.crossing / t.cross * 100).round(1)
    t["Inter-regional dependency, % of all referrals"] = (t.crossing / t.n * 100).round(1)
    t = t.sort_values("Inter-regional dependency, % of all referrals", ascending=False)
    t["Cross-city referrals, n"] = t.cross.astype(int)
    save_table("TableS10_regional_nesting", t[["Region", "Cross-city referrals, n", "Within own region, %",
                                               "Crossing a regional boundary, %",
                                               "Inter-regional dependency, % of all referrals"]].rename_axis("Referring city"),
               index=True)
    KEY["cross_region_share_of_cross_city_pct"] = xc[xc.CROSS_REGION == 1].REFERRALS.sum() / xc.REFERRALS.sum() * 100

    # Table S11 - supplementary codes by city
    n = a.groupby("REFERRING_CITY").REFERRALS.sum()
    u = a[a.ICD_CHAPTER == "SUPP"].groupby("REFERRING_CITY").REFERRALS.sum().reindex(n.index, fill_value=0)
    s11 = pd.DataFrame({"Referrals, n": n, "Supplementary codes, n": u, "Supplementary codes, %": (u / n * 100).round(1)})
    save_table("TableS11_supplementary_codes_by_city", s11.sort_values("Supplementary codes, %", ascending=False)
               .rename_axis("Referring city"), index=True)


def quarterly_sensitivity(q, out):
    """Tables S36-S38 and Figure S18 from the quarter-level file."""
    quarters = sorted(q.QUARTER.unique())
    di = lambda g: g.loc[g.CROSS_CITY == 1, "REFERRALS"].sum() / g.REFERRALS.sum() * 100 if g.REFERRALS.sum() else np.nan
    pooled = q.groupby("REFERRING_CITY").apply(di, include_groups=False)
    order = pooled.sort_values(ascending=False).index
    nref = q.groupby("REFERRING_CITY").REFERRALS.sum()
    flagged = set(map(tuple, q.loc[q.COVERAGE_FLAG == 1, ["REFERRING_CITY", "QUARTER"]].drop_duplicates().values))
    heap = sorted(q.loc[q.HEAPING_CITY == 1, "REFERRING_CITY"].unique())

    qdi = q.groupby(["REFERRING_CITY", "QUARTER"]).apply(di, include_groups=False).unstack().reindex(columns=quarters)
    rows = []
    for c in order:
        rows.append([c, f"{int(nref[c]):,}"] + ["–" if pd.isna(qdi.loc[c, x]) else
                                               f"{qdi.loc[c, x]:.1f}" + ("‡" if (c, x) in flagged else "")
                                               for x in quarters] + [f"{pooled[c]:.1f}"])
    nat = [di(q[q.QUARTER == x]) for x in quarters]
    rows.append(["All cities", f"{int(q.REFERRALS.sum()):,}"] + [f"{v:.1f}" for v in nat] + [f"{di(q):.1f}"])
    rhos = []
    for x in quarters:
        s = qdi[x].dropna()
        rhos.append(stats.spearmanr(s, pooled[s.index]).statistic)
    rows.append(["Spearman rho with pooled city DI", ""] + [f"{r:.2f}" for r in rhos] + ["–"])
    save_table("TableS36_DI_by_quarter", pd.DataFrame(rows, columns=["Referring city", "Referrals, n"] +
                                                      [QUARTER_LABEL(x) for x in quarters] + ["Pooled"]))
    KEY.update(quarterly_rho_min=min(rhos), quarterly_rho_max=max(rhos),
               quarterly_cross_share_pct=dict(zip(quarters, [round(v, 1) for v in nat])))

    gdi = q.groupby(["REFERRING_CITY", "GENDER"]).apply(di, include_groups=False).unstack()
    gn = q.groupby(["REFERRING_CITY", "GENDER"]).REFERRALS.sum().unstack()
    rows = [[c, f"{int(gn.loc[c, 'M']):,}", f"{gdi.loc[c, 'M']:.1f}", f"{int(gn.loc[c, 'F']):,}", f"{gdi.loc[c, 'F']:.1f}",
             f"{gdi.loc[c, 'M'] - gdi.loc[c, 'F']:+.1f}"] for c in order]
    m, f = di(q[q.GENDER == "M"]), di(q[q.GENDER == "F"])
    rows.append(["All cities", f"{int(q[q.GENDER == 'M'].REFERRALS.sum()):,}", f"{m:.1f}",
                 f"{int(q[q.GENDER == 'F'].REFERRALS.sum()):,}", f"{f:.1f}", f"{m - f:+.1f}"])
    save_table("TableS37_DI_by_gender", pd.DataFrame(rows, columns=["Referring city", "Male, n", "Male DI, %", "Female, n",
                                                                    "Female DI, %", "Difference (M - F), pp"]))
    KEY.update(di_male_pct=m, di_female_pct=f, rho_gender=stats.spearmanr(gdi.loc[order, "M"], gdi.loc[order, "F"]).statistic)

    full_ch = q[q.ICD_CHAPTER != "OTHER"].groupby("ICD_CHAPTER").apply(di, include_groups=False)
    def head(x):
        xc = x[x.CROSS_CITY == 1]; hubs = xc.groupby("RECEIVING_CITY").REFERRALS.sum().sort_values(ascending=False)
        cdi = x.groupby("REFERRING_CITY").apply(di, include_groups=False)
        ch = x[x.ICD_CHAPTER != "OTHER"].groupby("ICD_CHAPTER").apply(di, include_groups=False)
        com = ch.index.intersection(full_ch.index)
        return {"n": x.REFERRALS.sum(), "share": di(x), "rho_city": stats.spearmanr(cdi, pooled[cdi.index]).statistic,
                "rho_ch": stats.spearmanr(ch[com], full_ch[com]).statistic,
                "top3": hubs.head(3).sum() / hubs.sum() * 100, "hubs": ", ".join(hubs.index[:3])}
    fq = q.COVERAGE_FLAG == 1; hc = q.HEAPING_CITY == 1
    fl_txt = "; ".join(f"{c} {QUARTER_LABEL(x)}" for c, x in sorted(flagged))
    scen = [("All data (reference)", "None", q),
            ("Excluding count-heaping cities", f"{len(heap)} cities: " + ", ".join(heap), q[~hc]),
            ("Excluding flagged city-quarters", f"{len(flagged)} city-quarters: {fl_txt}", q[~fq]),
            ("Excluding both", f"{len(heap)} cities and the remaining flagged city-quarters", q[~hc & ~fq])]
    rows = []
    for name, exc, x in scen:
        h = head(x); ref = name.startswith("All")
        rows.append([name, exc, f"{int(h['n']):,} ({h['n'] / q.REFERRALS.sum() * 100:.1f})", f"{h['share']:.1f}",
                     "1.00" if ref else f"{h['rho_city']:.2f}", "1.00" if ref else f"{h['rho_ch']:.2f}",
                     f"{h['top3']:.1f}", h["hubs"]])
        KEY[f"dq_{name}"] = {"share": h["share"], "rho_chapter": h["rho_ch"], "top3": h["top3"]}
    save_table("TableS38_data_quality_sensitivity", pd.DataFrame(rows, columns=[
        "Analysis", "Excluded", "Referrals retained, n (%)", "Cross-city share, %", "City DI ranking rho",
        "Chapter ranking rho", "Top-3 hub share, %", "Top-3 hubs"]))

    # Figure S18
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]; marks = ["o", "s", "^", "D", "v"]
    x = np.array([q[(q.QUARTER == z) & (q.CROSS_CITY == 1)].REFERRALS.sum() for z in quarters])
    n = np.array([q[q.QUARTER == z].REFERRALS.sum() for z in quarters]); p = x / n * 100
    lo, hi = wilson(x, n)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax[0].errorbar(range(len(quarters)), p, yerr=[p - lo, hi - p], fmt="-o", color=BLUE, lw=2, ms=7, capsize=3)
    for i, v in enumerate(p):
        ax[0].annotate(f"{v:.1f}%", (i, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8.5)
    ax[0].axhline(x.sum() / n.sum() * 100, ls="--", color=GREY, lw=1)
    ax[0].set_xticks(range(len(quarters)), [QUARTER_LABEL(z) for z in quarters]); ax[0].set_ylim(0, 13)
    ax[0].set_ylabel("Cross-city share of referrals, %"); ax[0].set_title("(a) National cross-city share by quarter", loc="left")
    for k, z in enumerate(quarters):
        s = qdi[z].dropna()
        ax[1].scatter(pooled[s.index], s, s=40, marker=marks[k % 5], color=ramp[k % 5], edgecolor="white",
                      label=f"{QUARTER_LABEL(z)} (rho = {rhos[k]:.2f})", zorder=3)
    ax[1].plot([0, 100], [0, 100], ls="--", color=GREY, lw=1)
    ax[1].set_xlabel("Pooled city DI, %"); ax[1].set_ylabel("Quarterly city DI, %")
    ax[1].set_title("(b) Quarterly vs pooled DI by referring city", loc="left"); ax[1].legend(frameon=False, fontsize=8)
    plt.tight_layout(); savefig(fig, out, "FigureS18_DI_stability_by_quarter")


# ==============================================================================================
# 5. Hubs
# ==============================================================================================
def hubs(a, out):
    xc = a[a.CROSS_CITY == 1]; tot = xc.REFERRALS.sum()
    od = xc.pivot_table(index="REFERRING_CITY", columns="RECEIVING_CITY", values="REFERRALS", aggfunc="sum", fill_value=0)
    save_table("TableS35_origin_destination_matrix", od.rename_axis("Referring city"), index=True)
    inflow = od.sum().sort_values(ascending=False)
    share = inflow / tot * 100
    rows = []
    for c in inflow.index:
        col = od[c]; g = xc[xc.RECEIVING_CITY == c].groupby("ICD_CHAPTER").REFERRALS.sum()
        rows.append({"Receiving city": c, "Inflow, n": int(inflow[c]), "Share of national inflow, %": round(share[c], 1),
                     "Origin cities, n": int((col > 0).sum()), "Main origin (share)": f"{col.idxmax()} ({col.max() / col.sum() * 100:.0f}%)",
                     "Leading chapter (share of inflow)": f"{CHAPTER_SHORT[g.idxmax()]} ({g.max() / g.sum() * 100:.0f}%)"})
    full = pd.DataFrame(rows)
    big = full[full["Share of national inflow, %"] >= 0.5]; small = full[full["Share of national inflow, %"] < 0.5]
    t5 = pd.concat([big, pd.DataFrame([{"Receiving city": f"Other {len(small)} cities", "Inflow, n": int(small["Inflow, n"].sum()),
                                        "Share of national inflow, %": round(small["Share of national inflow, %"].sum(), 1),
                                        "Origin cities, n": "–", "Main origin (share)": "–",
                                        "Leading chapter (share of inflow)": "–"}])]) if len(small) else big
    save_table("Table5_receiving_hubs", t5)
    KEY.update(n_receiving_lt_half_pct=len(small), inflow_by_city=inflow.to_dict())

    # Table S21 - destination concentration by referring city
    rows = []
    for c, r in od.iterrows():
        r = r[r > 0]
        rows.append({"Referring city": c, "Cross-city referrals, n": int(r.sum()), "Destination cities": len(r),
                     "Main destination (share)": f"{r.idxmax()} ({r.max() / r.sum() * 100:.0f}%)",
                     "Destination HHI": round(hhi(r / r.sum() * 100))})
    save_table("TableS21_destination_concentration", pd.DataFrame(rows).sort_values("Cross-city referrals, n", ascending=False))

    # Figure S8 - OD matrix
    order_o = od.sum(axis=1).sort_values(ascending=False).index; order_d = inflow.index
    fig, ax = plt.subplots(figsize=(9, 7))
    m = od.loc[order_o, order_d]
    sns.heatmap(np.log10(m.replace(0, np.nan)), cmap="Blues", ax=ax, linewidths=0.4, linecolor="white",
                annot=m.replace(0, np.nan), fmt=".0f", annot_kws={"fontsize": 6.5},
                cbar_kws={"label": "Cross-city referrals (log10)"})
    ax.set_xlabel("Receiving city"); ax.set_ylabel("Referring city")
    savefig(fig, out, "FigureS8_origin_destination_matrix")

    # Figure S9 - chapter composition of inflow, Figure S10 - inflow share
    top = inflow.index[:6]
    comp = xc[xc.RECEIVING_CITY.isin(top)].pivot_table(index="RECEIVING_CITY", columns="ICD_CHAPTER", values="REFERRALS",
                                                       aggfunc="sum", fill_value=0).reindex(index=top)
    comp = comp.div(comp.sum(axis=1), axis=0) * 100
    comp = comp.reindex(columns=[c for c in MODEL_CHAPTERS if c in comp.columns])
    comp.columns = [CHAPTER_SHORT[c] for c in comp.columns]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    sns.heatmap(comp, cmap="Oranges", annot=True, fmt=".0f", annot_kws={"fontsize": 7}, ax=ax, linewidths=0.4,
                cbar_kws={"label": "% of inflow"})
    ax.set_ylabel(""); ax.set_xlabel("")
    savefig(fig, out, "FigureS9_chapter_composition_of_inflow")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(range(len(share)), share.values[::-1], color=BLUE)
    ax.set_yticks(range(len(share)), share.index[::-1]); ax.set_xlabel("Share of national cross-city inflow, %")
    for i, v in enumerate(share.values[::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_title(f"HHI = {hhi(share):,.0f}; top-3 share = {share.head(3).sum():.1f}%", loc="left")
    savefig(fig, out, "FigureS10_inflow_share_by_receiving_city")
    return od


# ==============================================================================================
# 6. Exploratory cluster analyses
#    A: 15 cities with same-city referrals, local profile (chapter + age)
#    B: 19 referring cities, outbound profile (chapter + age + destination + dependency index)
#    C: 22 ICD-10 chapters, D: 100 highest-volume codes (age, cross-city rate, volume, repetition,
#       receiving cities; repetition and receiving cities adjusted for volume)
# ==============================================================================================
def shrunk_sqrt(counts, prior):
    """Profile shrunk towards the pooled profile (Dirichlet prior of prior x 0.5 x K pseudo-referrals),
    then Hellinger-transformed (square root), so each block has unit norm."""
    K = counts.shape[1]
    pi = counts.sum() / counts.values.sum()
    alpha = prior * 0.5 * K
    p = counts.add(alpha * pi, axis=1).div(counts.sum(axis=1) + alpha, axis=0)
    return np.sqrt(p), p * 100


def city_features(a, analysis, prior=1.0, cities=None):
    """Feature matrix, blocks and display profiles for Analyses A and B."""
    d = a[a.ICD_CHAPTER != "OTHER"]
    if cities is not None:
        d = d[d.REFERRING_CITY.isin(cities)]
    sub = d[d.CROSS_CITY == (0 if analysis == "A" else 1)].copy()
    ent = sorted(sub.loc[sub.REFERRALS > 0, "REFERRING_CITY"].unique())
    piv = lambda col, cats: sub.pivot_table(index="REFERRING_CITY", columns=col, values="REFERRALS", aggfunc="sum",
                                            fill_value=0).reindex(index=ent, columns=cats, fill_value=0)
    blocks = {"Chapter profile": (piv("ICD_CHAPTER", MODEL_CHAPTERS), [f"{CHAPTER_SHORT[c]}" for c in MODEL_CHAPTERS]),
              "Age profile": (piv("AGE_GROUP", AGE_GROUPS), [f"Age {g}" for g in AGE_GROUPS])}
    if analysis == "B":
        sub["DEST"] = np.where(sub.RECEIVING_CITY.isin(HUBS), sub.RECEIVING_CITY, "Other")
        blocks["Destination profile"] = (piv("DEST", HUBS + ["Other"]), [f"To {h}" for h in HUBS + ["other cities"]])
    Xs, disp, cols = [], [], {}
    for name, (cnt, labels) in blocks.items():
        h, shown = (np.sqrt(cnt.div(cnt.sum(axis=1), axis=0)), cnt.div(cnt.sum(axis=1), axis=0) * 100) \
            if prior == 0 else shrunk_sqrt(cnt, prior)
        h.columns = shown.columns = labels
        Xs.append(h); disp.append(shown); cols[name] = labels
    if analysis == "B":
        tot = d.groupby("REFERRING_CITY").REFERRALS.sum().reindex(ent)
        di = sub.groupby("REFERRING_CITY").REFERRALS.sum().reindex(ent) / tot
        Xs.append(di.rename("Dependency index").to_frame()); disp.append((di * 100).rename("Dependency index").to_frame())
        cols["Dependency index"] = ["Dependency index"]
    X = pd.concat(Xs, axis=1); D = pd.concat(disp, axis=1)
    vol = sub.groupby("REFERRING_CITY").REFERRALS.sum().reindex(ent)
    return X, cols, vol, D


def entity_features(a, analysis, adjust=True, codes=None):
    """Feature matrix for Analyses C (chapters) and D (codes); display values are unadjusted."""
    d = a[a.ICD_CHAPTER != "OTHER"]
    ent_col = "ICD_CHAPTER" if analysis == "C" else "DISEASE_CODE"
    if analysis == "D":
        d = d[d.DISEASE_CODE.isin(codes)]
    g = d.groupby(ent_col)
    vol = g.REFERRALS.sum()
    lv = np.log10(vol)
    cross = d[d.CROSS_CITY == 1].groupby(ent_col).REFERRALS.sum().reindex(vol.index, fill_value=0) / vol * 100
    rep = g.REPEATED.mean() * 100                                                   # share of records repeated
    recv = d[d.CROSS_CITY == 1].groupby(ent_col).RECEIVING_CITY.nunique().reindex(vol.index, fill_value=0).astype(float)
    age = d.pivot_table(index=ent_col, columns="AGE_GROUP", values="REFERRALS", aggfunc="sum",
                        fill_value=0).reindex(index=vol.index, columns=AGE_GROUPS, fill_value=0)
    age_sh = age.div(age.sum(axis=1), axis=0)
    z = lambda v: (v - v.mean()) / v.std(ddof=0) if v.std(ddof=0) > 0 else v * 0
    resid = lambda y: y - np.polyval(np.polyfit(lv, y, 1), lv)
    X = pd.DataFrame(index=vol.index)
    for g_ in AGE_GROUPS:
        X[f"Age {g_}"] = z(np.sqrt(age_sh[g_])) / np.sqrt(len(AGE_GROUPS))
    X["Cross-city rate (%)"] = z(cross)
    X["Volume (log10 referrals)"] = z(lv)
    X["Repetition rate (%)"] = z(resid(rep) if adjust else rep)
    X["Receiving cities (n)"] = z(resid(recv) if adjust else recv)
    D = pd.concat([age_sh * 100, cross.rename("Cross-city rate (%)"), lv.rename("Volume (log10 referrals)"),
                   rep.rename("Repetition rate (%)"), recv.rename("Receiving cities (n)")], axis=1)
    D.columns = list(X.columns)
    cols = {"Age profile": [f"Age {g_}" for g_ in AGE_GROUPS], "Cross-city rate (%)": ["Cross-city rate (%)"],
            "Volume (log10 referrals)": ["Volume (log10 referrals)"], "Repetition rate (%)": ["Repetition rate (%)"],
            "Receiving cities (n)": ["Receiving cities (n)"]}
    if analysis == "C":
        X.index = D.index = vol.index = [chapter_label(i) for i in vol.index]
    return X, cols, vol, D


def fit_labels(X, method, k, n_init=100, seed=SEED):
    if method == "K-Means":
        return KMeans(k, n_init=n_init, random_state=seed).fit(X).labels_
    return AgglomerativeClustering(k, linkage="ward").fit(X).labels_


def select(X, ks, n_init=100):
    """K-Means and Ward for each k; choose the highest silhouette among solutions without singleton clusters."""
    X = np.asarray(X, float); rows, best = [], None
    for k in ks:
        if k >= len(X):
            break
        for method in ["K-Means", "Ward"]:
            lab = fit_labels(X, method, k, n_init)
            wss = sum(((X[lab == j] - X[lab == j].mean(axis=0)) ** 2).sum() for j in range(k))
            r = {"Method": method, "k": k, "Silhouette": silhouette_score(X, lab), "Davies-Bouldin": davies_bouldin_score(X, lab),
                 "Calinski-Harabasz": calinski_harabasz_score(X, lab), "Within-cluster SS": wss,
                 "Smallest cluster": int(np.bincount(lab).min())}
            rows.append(r)
            if r["Smallest cluster"] >= 2 and (best is None or r["Silhouette"] > best[0]["Silhouette"] + 1e-12):
                best = (r, lab)
    tab = pd.DataFrame(rows)
    if best is None:                                                   # every solution has a singleton
        i = tab.Silhouette.idxmax(); best = (rows[i], fit_labels(X, rows[i]["Method"], rows[i]["k"], n_init))
    return tab, best[0], best[1]


def order_labels(lab, vol):
    """Number clusters 1..k by size (largest first), ties by volume."""
    s = pd.DataFrame({"lab": lab, "vol": np.asarray(vol)}).groupby("lab").agg(n=("vol", "size"), v=("vol", "sum"))
    order = s.sort_values(["n", "v"], ascending=False).index.tolist()
    return np.array([order.index(l) + 1 for l in lab])


def weighted(X, cols, block, w):
    X = X.copy()
    if w == 0:
        return X.drop(columns=cols[block])
    X[cols[block]] = X[cols[block]] * w
    return X


def bootstrap_records(a, analysis, rng, codes=None):
    """One parametric multinomial bootstrap replicate: referral counts resampled within each entity."""
    d = a[a.ICD_CHAPTER != "OTHER"].copy()
    if analysis in "AB":
        key = "REFERRING_CITY"
        if analysis == "A":
            d = d[d.CROSS_CITY == 0]
    else:
        key = "ICD_CHAPTER" if analysis == "C" else "DISEASE_CODE"
        if analysis == "D":
            d = d[d.DISEASE_CODE.isin(codes)]
    new = np.zeros(len(d), dtype=int)
    for _, idx in d.groupby(key).indices.items():
        w = d.REFERRALS.values[idx]
        new[idx] = rng.multinomial(w.sum(), w / w.sum())
    d["REFERRALS"] = new
    return d[d.REFERRALS > 0]


def features(a, analysis, codes=None, prior=1.0, adjust=True, cities=None):
    if analysis in "AB":
        return city_features(a, analysis, prior, cities)
    return entity_features(a, analysis, adjust, codes)


def cluster_analysis(a, analysis, title, ks, ok_cities, n_perm, n_boot, codes=None):
    X, cols, vol, D = features(a, analysis, codes)
    sel, best, lab = select(X.values, ks)
    lab = order_labels(lab, vol.values)
    method, k = best["Method"], best["k"]
    other = fit_labels(X.values, "Ward" if method == "K-Means" else "K-Means", k)
    res = {"analysis": analysis, "title": title, "X": X, "cols": cols, "vol": vol, "D": D, "labels": pd.Series(lab, index=X.index),
           "method": method, "k": k, "best": best, "selection": sel}

    # stability: bootstrap, leave-one-out, K-Means seeds, agreement between algorithms
    rng = np.random.default_rng(SEED)                     # own random stream per analysis (reproducible)
    boot_ari, co = [], np.zeros(len(X))
    same_main = lab[:, None] == lab[None, :]
    for _ in range(n_boot):
        bd = bootstrap_records(a, analysis, rng, codes)
        Xb = features(bd, analysis, codes)[0]
        Xb = Xb.reindex(X.index).fillna(0)
        lb = fit_labels(Xb.values, method, k, n_init=10)
        boot_ari.append(adjusted_rand_score(lab, lb))
        same_b = lb[:, None] == lb[None, :]
        co += ((same_main == same_b).sum(axis=1) - 1) / (len(X) - 1)
    stab = pd.Series(co / n_boot, index=X.index)
    loo = []
    for i in range(len(X)):
        keep = np.arange(len(X)) != i
        loo.append(adjusted_rand_score(lab[keep], fit_labels(X.values[keep], method, k)))
    seeds = [adjusted_rand_score(lab, KMeans(k, n_init=1, random_state=s).fit(X.values).labels_) for s in range(50)]
    boot_ari = np.array(boot_ari)
    res.update(stability=stab, boot_ari=boot_ari, loo=np.array(loo), seed_ari=np.array(seeds),
               ari_other_algorithm=adjusted_rand_score(lab, other))

    # null benchmark: every column permuted independently across entities, full selection re-run
    obs = select(X.values, ks, n_init=10)[1]["Silhouette"]
    null, rng = [], np.random.default_rng(SEED)
    for _ in range(n_perm):
        Xp = np.column_stack([rng.permutation(X.values[:, j]) for j in range(X.shape[1])])
        null.append(select(Xp, ks, n_init=10)[1]["Silhouette"])
    null = np.array(null)
    res.update(null=null, null_obs=obs, null_p=(1 + (null >= obs).sum()) / (1 + n_perm))

    # sensitivity to analytic choices
    sens = []
    def variant(name, Xv, ref_index=None):
        s, b, l = select(Xv.values, ks)
        base = lab if ref_index is None else res["labels"].reindex(ref_index).values
        sens.append({"Analysis": title, "Variant": name, "Selected": f"{b['Method']}, k = {b['k']}",
                     "Silhouette": round(b["Silhouette"], 3), "Smallest cluster": b["Smallest cluster"],
                     "ARI vs main": round(adjusted_rand_score(base, l), 2)})
    for block in cols:
        variant(f"{block}: dropped", weighted(X, cols, block, 0))
        variant(f"{block}: double weight", weighted(X, cols, block, np.sqrt(2)))
    if analysis in "CD":
        variant("Repetition and receiving cities not adjusted for volume", features(a, analysis, codes, adjust=False)[0])
    prior_rows = []
    if analysis in "AB":
        for p_name, p in [("none (raw proportions)", 0), ("1x (main)", 1), ("2x", 2), ("5x", 5)]:
            Xp = features(a, analysis, prior=p)[0]
            s, b, l = select(Xp.values, ks)
            prior_rows.append({"Analysis": title, "Prior strength": p_name, "Selected": f"{b['Method']}, k = {b['k']}",
                               "Silhouette": round(b["Silhouette"], 3), "Smallest cluster": b["Smallest cluster"],
                               "ARI vs main solution": round(adjusted_rand_score(lab, l), 2)})
    if analysis == "B":
        Xr = features(a, "B", cities=ok_cities)[0]
        variant(f"{len(ok_cities)} coverage-adequate cities only", Xr, ref_index=Xr.index)
    res.update(sensitivity=pd.DataFrame(sens), prior=pd.DataFrame(prior_rows))
    print(f"  Analysis {analysis}: {method}, k = {k}, silhouette {best['Silhouette']:.3f}; "
          f"null p = {res['null_p']:.3f}; bootstrap ARI median {np.median(boot_ari):.2f}")
    return res


def cluster_tables(r):
    A, X, D, lab, vol = r["analysis"], r["X"], r["D"], r["labels"], r["vol"]
    name = {"A": "S14", "B": "S17", "C": "S22", "D": "S25"}[A]
    sel = r["selection"].copy()
    for c in ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Within-cluster SS"]:
        sel[c] = sel[c].round(3)
    save_table(f"Table{name}_Analysis{A}_validity_indices", sel)
    memb = {"A": "S15", "B": "S18", "C": "S23", "D": "S26"}[A]
    m = pd.DataFrame({"Entity": X.index, "Cluster": [f"{A}{l}" for l in lab], "Volume (referrals)": vol.values,
                      "Stability": r["stability"].round(2).values}).sort_values(["Cluster", "Volume (referrals)"],
                                                                                 ascending=[True, False])
    save_table(f"Table{memb}_Analysis{A}_membership", m)
    # cluster summaries: group means of displayed values, distinguishing features by mean z-score
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    rows = []
    for c in sorted(lab.unique()):
        idx = lab.index[lab == c]
        zc = Z.loc[idx].mean().sort_values()
        hi_f = [f"{f} ({D.loc[idx, f].mean():.1f})" for f in zc.index[::-1][:3]]
        lo_f = [f"{f} ({D.loc[idx, f].mean():.1f})" for f in zc.index[:2]]
        rows.append({"Cluster": f"{A}{c}", "n": len(idx), "Members": ", ".join(idx[:25]) + (", ..." if len(idx) > 25 else ""),
                     "Referrals": int(vol[idx].sum()), "Highest features (mean)": "; ".join(hi_f),
                     "Lowest features (mean)": "; ".join(lo_f), "Stability": round(r["stability"][idx].mean(), 2)})
    summ = {"A": "S16", "B": "S18b", "C": "S24", "D": "S27"}[A]
    save_table(f"Table{summ}_Analysis{A}_cluster_summary", pd.DataFrame(rows))
    return pd.DataFrame(rows)


def table6(r, a):
    """Main-text Table 6: outbound typology of referring cities (Analysis B)."""
    lab, D = r["labels"], r["D"]
    xc = a[a.CROSS_CITY == 1].groupby("REFERRING_CITY").REFERRALS.sum()
    Z = (r["X"] - r["X"].mean()) / r["X"].std(ddof=0).replace(0, 1)
    rows = []
    for c in sorted(lab.unique()):
        idx = lab.index[lab == c]
        zc = Z.loc[idx].drop(columns=["Dependency index"]).mean().sort_values(ascending=False)
        feats = [f"{f} {D.loc[idx, f].mean():.1f}%" for f in zc.index[:3]]
        rows.append({"Group": f"B{c}", "Cities": ", ".join(idx), "Cross-city referrals, n": int(xc[idx].sum()),
                     "Mean DI, %": round(D.loc[idx, "Dependency index"].mean(), 1),
                     "To Riyadh, % (group mean)": round(D.loc[idx, "To Riyadh"].mean(), 1),
                     "To Jeddah, % (group mean)": round(D.loc[idx, "To Jeddah"].mean(), 1),
                     "Distinguishing features (group mean)": "; ".join(feats)})
    save_table("Table6_outbound_typology", pd.DataFrame(rows))


def compare_partitions(rB, rA, a, od):
    """Table S31 (Analysis B vs simpler partitions) and Table S20 (A vs B)."""
    lab = rB["labels"]; cities = lab.index
    main_dest = od.loc[cities].idxmax(axis=1)
    group = main_dest.map(lambda c: "Riyadh" if c == "Riyadh" else "Jeddah/Makkah" if c in ("Jeddah", "Makkah") else "Other hub")
    region = a.groupby("REFERRING_CITY").REFERRING_REGION.first().reindex(cities).str.replace(" Region", "", regex=False)
    G = nx.Graph()
    for o, row in od.iterrows():
        for d_, w in row.items():
            if w > 0:
                G.add_edge(o, d_, weight=G[o][d_]["weight"] + w if G.has_edge(o, d_) else w)
    comms = nx.community.louvain_communities(G, weight="weight", seed=SEED)
    comms = sorted(comms, key=lambda s: -sum(od.loc[od.index.intersection(list(s))].sum(axis=1)))
    cm = {c: f"N{i + 1}" for i, s in enumerate(comms) for c in s}
    mod = nx.community.modularity(G, comms, weight="weight")
    net = cities.map(cm)
    t = pd.DataFrame({"Referring city": cities, "Analysis B cluster": [f"B{l}" for l in lab], "Main destination": main_dest.values,
                      "Destination group": group.values, "Network community": net.values, "Region": region.values})
    ari = {"main destination": adjusted_rand_score(lab, main_dest), "destination group": adjusted_rand_score(lab, group),
           "network community": adjusted_rand_score(lab, net), "region": adjusted_rand_score(lab, region)}
    save_table("TableS31_AnalysisB_vs_simpler_partitions", t.sort_values(["Analysis B cluster", "Referring city"]))
    KEY.update(analysisB_ARI_partitions=ari, louvain_modularity=mod)
    both = rA["labels"].index.intersection(cities)
    t20 = pd.DataFrame({"City": both, "Analysis A cluster": [f"A{rA['labels'][c]}" for c in both],
                        "Analysis B cluster": [f"B{lab[c]}" for c in both],
                        "Dependency index, %": [round(rB["D"].loc[c, "Dependency index"], 1) for c in both]})
    save_table("TableS20_AnalysisA_vs_B", t20.sort_values(["Analysis A cluster", "Analysis B cluster"]))
    KEY.update(ARI_A_vs_B=adjusted_rand_score(rA["labels"][both], lab[both]),
               NMI_A_vs_B=normalized_mutual_info_score(rA["labels"][both], lab[both]))


def summary_tables(results):
    s28, s29, s30, s19 = [], [], [], []
    for r in results:
        b = r["best"]
        s28.append({"Analysis": r["title"], "Entities": len(r["X"]), "Features": r["X"].shape[1], "Selected method": r["method"],
                    "k": r["k"], "Silhouette": round(b["Silhouette"], 3), "Davies-Bouldin": round(b["Davies-Bouldin"], 3),
                    "ARI vs other algorithm (same k)": round(r["ari_other_algorithm"], 2),
                    "Bootstrap ARI, median (IQR)": f"{np.median(r['boot_ari']):.2f} ({np.percentile(r['boot_ari'], 25):.2f}"
                                                   f"–{np.percentile(r['boot_ari'], 75):.2f})",
                    "Replicates with ARI >= 0.80, %": round((r["boot_ari"] >= 0.8).mean() * 100, 1),
                    "Leave-one-out ARI, median (min)": f"{np.median(r['loo']):.2f} ({r['loo'].min():.2f})",
                    "K-Means seed ARI, median (min)": f"{np.median(r['seed_ari']):.2f} ({r['seed_ari'].min():.2f})",
                    "Mean co-assignment stability": round(r["stability"].mean(), 2)})
        s29.append({"Analysis": r["title"], "Entities": len(r["X"]), "Observed best silhouette": round(r["null_obs"], 3),
                    "Null mean": round(r["null"].mean(), 3), "Null 95th percentile": round(np.percentile(r["null"], 95), 3),
                    "Permutation p": round(r["null_p"], 3), "Structure beyond null?": "Yes" if r["null_p"] < 0.05 else "No"})
        s30.append(r["sensitivity"]); s19.append(r["prior"])
        KEY[f"cluster_{r['analysis']}"] = {"method": r["method"], "k": r["k"], "silhouette": b["Silhouette"],
                                           "null_p": r["null_p"], "null_p95": float(np.percentile(r["null"], 95))}
    save_table("TableS28_cluster_solutions_summary", pd.DataFrame(s28))
    save_table("TableS29_null_benchmark", pd.DataFrame(s29))
    save_table("TableS30_sensitivity_analytic_choices", pd.concat(s30, ignore_index=True))
    save_table("TableS19_prior_strength", pd.concat(s19, ignore_index=True))


def cluster_figures(r, out, fig_names):
    X, lab, A = r["X"], r["labels"], r["analysis"]
    sel = r["selection"]
    # selection plot
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, m in enumerate(["K-Means", "Ward"]):
        s = sel[sel.Method == m]
        ax[0].plot(s.k, s.Silhouette, "-o", color=CAT[i], label=m, ms=4)
        ax[1].plot(s.k, s["Davies-Bouldin"], "-o", color=CAT[i], label=m, ms=4)
    for a_ in ax:
        a_.axvline(r["k"], color=GREY, ls=":", lw=1); a_.set_xlabel("Number of clusters, k")
    ax[0].set_ylabel("Silhouette (higher = better)"); ax[1].set_ylabel("Davies-Bouldin (lower = better)")
    ax[0].legend(frameon=False)
    savefig(fig, out, fig_names["selection"])
    # dendrogram + PCA
    cols = {c: CAT[(c - 1) % len(CAT)] for c in sorted(lab.unique())}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    Z = linkage(X.values, method="ward")
    dendrogram(Z, labels=list(X.index), ax=ax[0], leaf_rotation=90, leaf_font_size=7 if len(X) < 40 else 4,
               color_threshold=0, above_threshold_color="#777777")
    for t in ax[0].get_xmajorticklabels():
        t.set_color(cols[lab[t.get_text()]])
    ax[0].axhline((Z[-r["k"], 2] + Z[-r["k"] + 1, 2]) / 2, color="#333", ls="--", lw=0.8)
    ax[0].set_title(f"(a) Ward dendrogram, cut at k = {r['k']}", loc="left"); ax[0].set_ylabel("Ward distance")
    pc = PCA(2, random_state=SEED).fit(X.values); P = pc.transform(X.values)
    for c in sorted(lab.unique()):
        m = (lab == c).values
        ax[1].scatter(P[m, 0], P[m, 1], color=cols[c], s=40, label=f"{A}{c}", edgecolor="white", zorder=3)
    if len(X) <= 25:
        for (x_, y_), t in zip(P, X.index):
            ax[1].annotate(t, (x_, y_), xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax[1].set_xlabel(f"PC1 ({pc.explained_variance_ratio_[0] * 100:.0f}%)")
    ax[1].set_ylabel(f"PC2 ({pc.explained_variance_ratio_[1] * 100:.0f}%)")
    ax[1].legend(frameon=False, fontsize=8); ax[1].set_title(f"(b) {r['method']} clusters, first two components", loc="left")
    savefig(fig, out, fig_names["dendrogram"])
    # profiles: group means (cell values) coloured by mean z-score
    D = r["D"]; Zs = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    gm = D.groupby(lab).mean().T; gz = Zs.groupby(lab).mean().T
    gm.columns = gz.columns = [f"{A}{c} (n={int((lab == c).sum())})" for c in gm.columns]
    fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(gm.columns), 0.26 * len(gm) + 1.2))
    sns.heatmap(gz, cmap="RdBu_r", center=0, vmin=-2, vmax=2, annot=gm.round(1), fmt="", annot_kws={"fontsize": 7}, ax=ax,
                cbar_kws={"label": "Mean z-score vs all entities"}, linewidths=0.3)
    starts = np.cumsum([0] + [len(v) for v in r["cols"].values()])[1:-1]
    for s_ in starts:
        ax.axhline(s_, color="#333", lw=1)
    ax.set_ylabel("")
    savefig(fig, out, fig_names["profiles"])


def null_figure(results, out):
    fig, ax = plt.subplots(1, len(results), figsize=(3.2 * len(results), 3))
    for a_, r in zip(np.atleast_1d(ax), results):
        a_.hist(r["null"], bins=20, color="#c9c9c9", edgecolor="white")
        a_.axvline(r["null_obs"], color=ORANGE, lw=2)
        a_.axvline(np.percentile(r["null"], 95), color="#333", ls="--", lw=1)
        a_.set_title(f"Analysis {r['analysis']} (p = {r['null_p']:.3f})", fontsize=9); a_.set_xlabel("Best silhouette")
    np.atleast_1d(ax)[0].set_ylabel("Permutations")
    savefig(fig, out, "FigureS17_null_benchmark")


# ==============================================================================================
# 7. Maps (Figure 6, Table S32)
# ==============================================================================================
def saudi_outline():
    """Optional country outline from Natural Earth (needs internet and GeoPandas); the map works without it."""
    try:
        import geopandas as gpd
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        w = gpd.read_file(url)
        return w[w["ADMIN"] == "Saudi Arabia"]
    except Exception:
        return None


def maps(a, out, rA, rB, od):
    save_table("TableS32_map_coordinates", pd.DataFrame([{"City / region": c, "Latitude": v[0], "Longitude": v[1],
                                                          "Location used": v[2]} for c, v in COORDS.items()]))
    outline = saudi_outline()
    same = a[a.CROSS_CITY == 0].groupby("REFERRING_CITY").REFERRALS.sum()
    allc = sorted(set(a.REFERRING_CITY) | set(a.RECEIVING_CITY))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    for a_ in ax:
        if outline is not None:
            outline.plot(ax=a_, color="#f3f3f3", edgecolor="#bbbbbb", lw=0.8)
        a_.set_xlim(34.5, 56); a_.set_ylim(15.5, 32.6); a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
        for s in a_.spines.values():
            s.set_visible(False)
    # (a) same-city volume coloured by Analysis A
    for c in allc:
        lat, lon, _ = COORDS[c]
        if c in same.index and same[c] > 0 and rA is not None and c in rA["labels"].index:
            colr = CAT[(rA["labels"][c] - 1) % len(CAT)]
            ax[0].scatter(lon, lat, s=max(same[c] / 300, 12), color=colr, alpha=0.75, edgecolor="#333", lw=0.5, zorder=3)
        else:
            ax[0].scatter(lon, lat, s=25, facecolor="none", edgecolor="#555", lw=1, zorder=3)
        ax[0].annotate(c, (lon, lat), xytext=LABEL_OFFSET.get(c, (4, 3)), textcoords="offset points", fontsize=7)
    if rA is not None:
        for cl in sorted(rA["labels"].unique()):
            ax[0].scatter([], [], color=CAT[(cl - 1) % len(CAT)], label=f"A{cl}", s=40)
        ax[0].scatter([], [], facecolor="none", edgecolor="#555", label="No same-city referrals", s=25)
        ax[0].legend(frameon=False, loc="lower left", fontsize=8)
    ax[0].set_title("(a) Same-city referral volume (circle area)", loc="left")
    # (b) cross-city flows coloured by the origin's Analysis B group
    mx = od.values.max()
    for o, row in od.iterrows():
        for d_, w in row.items():
            if w <= 0:
                continue
            (y0, x0, _), (y1, x1, _) = COORDS[o], COORDS[d_]
            colr = CAT[(rB["labels"][o] - 1) % len(CAT)] if rB is not None and o in rB["labels"].index else GREY
            ax[1].annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", mutation_scale=6, color=colr, alpha=0.6,
                                                                   lw=0.4 + 6 * np.sqrt(w / mx),
                                                                   connectionstyle="arc3,rad=0.15"))
    inflow = od.sum()
    for c in allc:
        lat, lon, _ = COORDS[c]
        ax[1].scatter(lon, lat, s=12, color="#333", zorder=4)
        if inflow.get(c, 0) > 0:
            ax[1].scatter(lon, lat, s=max(inflow[c] / 40, 15), facecolor="none", edgecolor="#333", lw=1, zorder=4)
        ax[1].annotate(c, (lon, lat), xytext=LABEL_OFFSET.get(c, (4, 3)), textcoords="offset points", fontsize=7,
                       bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7), zorder=5)
    if rB is not None:
        for cl in sorted(rB["labels"].unique()):
            ax[1].plot([], [], color=CAT[(cl - 1) % len(CAT)], lw=3, label=f"B{cl}")
        ax[1].legend(frameon=False, loc="lower left", fontsize=8, title="Origin group", title_fontsize=8)
    ax[1].set_title("(b) Cross-city flows (width ~ sqrt volume; arrow = direction); rings = inflow", loc="left")
    savefig(fig, out, "Figure6_maps")


# ==============================================================================================
# Main
# ==============================================================================================
def load(path):
    a = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False, dtype={"DISEASE_CODE": str, "ICD_CHAPTER": str})
    a["REFERRALS"] = a.REFERRALS.astype(int)
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/processed/referrals_analysis.csv")
    ap.add_argument("--quarterly-input", default="data/processed/referrals_quarterly.csv")
    ap.add_argument("--out-dir", default="reports/analysis")
    ap.add_argument("--min-same-city", type=int, default=100, help="coverage-adequate: at least this many same-city referrals")
    ap.add_argument("--n-permutations", type=int, default=100)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--skip-clustering", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir); fig_dir = out / "figures"; tab_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)

    a = load(args.input)
    print(f"Analysis dataset: {len(a):,} records, {a.REFERRALS.sum():,} referrals")
    print("1. Descriptive statistics"); descriptives(a, fig_dir)
    print("2. Cross-city dependency index"); d_city = dependency(a, fig_dir, args.min_same_city)
    ok = KEY["coverage_adequate_cities"]
    print("3. Multivariable model"); model(a, fig_dir, ok)
    print("4. Sensitivity analyses"); sensitivity(a, d_city, ok)
    if Path(args.quarterly_input).exists():
        quarterly_sensitivity(load(args.quarterly_input), fig_dir)
    else:
        print(f"   (quarter-level file {args.quarterly_input} not found - Tables S36-S38 skipped)")
    print("5. Hubs"); od = hubs(a, fig_dir)

    rA = rB = None
    if not args.skip_clustering:
        print("6. Exploratory cluster analyses (takes a few minutes)")
        top100 = a.groupby("DISEASE_CODE").REFERRALS.sum().nlargest(100).index
        specs = [("A", "Analysis A (same-city profiles)", range(2, 9), None),
                 ("B", "Analysis B (cross-city outbound profiles)", range(2, 9), None),
                 ("C", "Analysis C (ICD-10 chapters)", range(2, 9), None),
                 ("D", "Analysis D (top-100 codes)", range(2, 11), top100)]
        figs = {"A": ("FigureS4", "FigureS5", "FigureS6"), "B": ("FigureS7", "Figure4", "Figure5"),
                "C": ("FigureS11", "FigureS12", "FigureS13"), "D": ("FigureS14", "FigureS15", "FigureS16")}
        results = []
        for A, title, ks, codes in specs:
            r = cluster_analysis(a, A, title, ks, ok, args.n_permutations, args.n_bootstrap, codes)
            cluster_tables(r)
            f = figs[A]
            cluster_figures(r, fig_dir, {"selection": f"{f[0]}_Analysis{A}_selection",
                                         "dendrogram": f"{f[1]}_Analysis{A}_dendrogram_pca",
                                         "profiles": f"{f[2]}_Analysis{A}_profiles"})
            results.append(r)
        rA, rB = results[0], results[1]
        table6(rB, a)
        compare_partitions(rB, rA, a, od)
        summary_tables(results)
        null_figure(results, fig_dir)
    print("7. Maps"); maps(a, fig_dir, rA, rB, od)

    # save all tables
    with pd.ExcelWriter(out / "all_tables.xlsx") as xw:
        for name, t in sorted(TABLES.items(), key=lambda kv: kv[0]):
            t.to_csv(tab_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
            t.to_excel(xw, sheet_name=name.replace("Table", "")[:31], index=False)
    json.dump(KEY, open(out / "key_numbers.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nDone: {len(TABLES)} tables in {tab_dir}, figures in {fig_dir}, key numbers in {out / 'key_numbers.json'}")
    print(f"Headline: {KEY['total_referrals']:,} referrals, cross-city {KEY['cross_city_share_pct']:.1f}%, "
          f"top-3 hubs {KEY['top3_hub_share_pct']:.1f}%, HHI {KEY['inflow_hhi']:,.0f}")


if __name__ == "__main__":
    main()
