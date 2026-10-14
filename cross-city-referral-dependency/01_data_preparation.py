"""
01_data_preparation.py
======================
From the five quarterly "Referral of Medical Cases" Excel extracts to the analysis dataset.

Stages
  1. Loading                 read the five Excel files (all values as text), quarter from the file name
  2. Data-quality review     missing values, duplicates, spelling variants, unexpected categories,
                             invalid / extreme counts, malformed codes  (nothing is changed here)
  3. Cleaning and regions    harmonise city names (Eastern -> Al-Ahsa), remove exact duplicates and
                             invalid counts, city -> region lookup from Q4 2025, merge duplicate groups
  4. Diagnosis coding        ICD-10-AM chapter from the chapter code ranges (icd10am_chapters.py);
                             optional validation against a licensed ICD-10-AM tabular list
  5. Flags and datasets      count heaping, reporting-coverage changes, exclusion of Outside KSA,
                             analysis dataset aggregated over quarters and genders

Inputs   data/raw/Referral of Medical Cases <year> Q<q> .xlsx   (five files, one sheet each)
         --icd10am PATH  CSV of the licensed ICD-10-AM tabular list (optional; column "code")
Outputs  data/processed/referrals_clean.csv             cleaned quarter-level rows (incl. Outside KSA)
         data/processed/referrals_quarterly.csv         quarter-level analysis rows (Outside KSA excluded)
         data/processed/referrals_analysis.csv          aggregated over quarters and genders
         reports/preprocessing/*.xlsx, *.csv, methods_text.md, figure_S1_flow.png

Run:     python 01_data_preparation.py            (optional: --raw-dir PATH --out-dir PATH --icd10am PATH)
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils

from icd10am_chapters import CHAPTER_SHORT, icd10am_chapter

SEED = 42
np.random.seed(SEED)

# ----------------------------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------------------------
EXPECTED = ["GENDER", "PROVIDER_REGION", "REFERRING_PROVIDER_REGION", "DISEASE_CODE",
            "RECEIVING_PROVIDER_REGION", "AGE_GROUP", "TOTAL_REFERRALS"]
CITY_COLS = ["REFERRING_PROVIDER_REGION", "RECEIVING_PROVIDER_REGION"]
MISSING_TOKENS = {"", "NULL", "N/A", "NA", "NAN", "NONE", "-", "--", "?", "#N/A"}

# Standardisation dictionary (left: spelling found in the data, lower case; right: standard name).
# The Eastern Province directorate unit ("East", "Eastern", "Eastern Region" in the city columns) is first
# standardised to "Eastern" and then merged with Al-Ahsa (MERGE_UNITS), so that no city-level unit carries
# a region name.
CITY_FIXES = {
    "asir": "Asir", "assir": "Asir",
    "hafer albatin": "Hafr Al-Batin", "hafr albatin": "Hafr Al-Batin", "hafr al batin": "Hafr Al-Batin",
    "hafr al-batin": "Hafr Al-Batin",
    "hassa": "Al-Ahsa", "ihsaa": "Al-Ahsa", "al ahsa": "Al-Ahsa", "al-ahsa": "Al-Ahsa", "al hasa": "Al-Ahsa",
    "al-hasa": "Al-Ahsa", "ahsa": "Al-Ahsa", "hasa": "Al-Ahsa",
    "east": "Eastern", "eastern": "Eastern", "eastern region": "Eastern",
    "medina": "Madinah", "medinah": "Madinah", "madina": "Madinah", "al madinah": "Madinah", "al-madinah": "Madinah",
    "taef": "Taif", "taif": "Taif",
    "baha": "Al-Baha", "al baha": "Al-Baha", "al-baha": "Al-Baha", "albaha": "Al-Baha",
    "jouf": "Al-Jouf", "al jouf": "Al-Jouf", "al-jouf": "Al-Jouf", "aljouf": "Al-Jouf", "jawf": "Al-Jouf",
    "al jawf": "Al-Jouf",
    "qunfotha": "Qunfudah", "qunfudah": "Qunfudah", "qunfudhah": "Qunfudah", "al qunfudhah": "Qunfudah",
    "outside-ksa": "Outside KSA", "outside ksa": "Outside KSA",
    "mecca": "Makkah", "makka": "Makkah", "makkah": "Makkah",
    "jiddah": "Jeddah", "jedda": "Jeddah", "jeddah": "Jeddah",
    "gassim": "Qassim", "qaseem": "Qassim", "al qassim": "Qassim", "qassim": "Qassim",
    "jizan": "Jazan", "gizan": "Jazan", "jazan": "Jazan",
    "northern border": "Northern Borders", "northern borders": "Northern Borders",
    "qurayat": "Qurayyat", "qurayyat": "Qurayyat", "bisha": "Bisha", "riyadh": "Riyadh",
    "hail": "Hail", "tabuk": "Tabuk", "najran": "Najran",
}
MERGE_UNITS = {"Eastern": "Al-Ahsa"}

REGION_FIXES = {
    "centeral region": "Middle Region", "central region": "Middle Region", "middle region": "Middle Region",
    "eastern region": "Eastern Region", "western region": "Western Region",
    "northern region": "Northern Region", "southern region": "Southern Region",
}
REGIONS = ["Middle Region", "Eastern Region", "Western Region", "Northern Region", "Southern Region"]

# Administrative geography, used for cities absent from the Q4 2025 lookup and to check the lookup
GEOGRAPHY = {
    "Riyadh": "Middle Region", "Qassim": "Middle Region",
    "Makkah": "Western Region", "Jeddah": "Western Region", "Taif": "Western Region",
    "Madinah": "Western Region", "Qunfudah": "Western Region",
    "Al-Ahsa": "Eastern Region", "Eastern": "Eastern Region", "Hafr Al-Batin": "Eastern Region",
    "Asir": "Southern Region", "Bisha": "Southern Region", "Jazan": "Southern Region",
    "Najran": "Southern Region", "Al-Baha": "Southern Region",
    "Tabuk": "Northern Region", "Hail": "Northern Region", "Al-Jouf": "Northern Region",
    "Northern Borders": "Northern Region", "Qurayyat": "Northern Region",
    "Outside KSA": "Outside KSA",
}
MANUAL_REGION = {}            # your own decisions override everything, e.g. {"Qunfudah": "Western Region"}

AGE_GROUPS = ["00-17", "18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

# Data-quality flag rules (manuscript Section 2.2)
HEAP_RATIO = 4                # a value at least 4x as frequent as its neighbouring values within a city
HEAP_MIN_ROWS = 20            # ...and seen in at least 20 rows of that city
COVERAGE_LOW, COVERAGE_HIGH = 0.5, 2.0   # city-quarter total below half / above twice the city's median quarter


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def is_missing(s):
    return s.astype(str).str.strip().str.upper().isin(MISSING_TOKENS)


def norm(x):
    return re.sub(r"\s+", " ", str(x).strip()).lower()


def clean_names(s, fixes):
    """Trim spaces and apply the dictionary; names not in it keep their most common spelling."""
    trimmed = s.map(lambda x: re.sub(r"\s+", " ", str(x).strip()))
    missing = is_missing(trimmed)
    key = trimmed.map(norm)
    most_common = trimmed[~missing].groupby(key[~missing]).agg(lambda v: v.value_counts().idxmax())
    out = key.map(fixes).fillna(key.map(most_common))
    out[missing] = ""
    return out


def to_count(s):
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")


def standardize_col(c):
    return re.sub(r"\s+", "_", str(c).strip().upper())


def heaping_table(df, city_col="REFERRING_CITY", count_col="REFERRALS"):
    """For each city, the referral count value that is most over-represented relative to its neighbours."""
    rows = []
    for city, g in df.groupby(city_col):
        vc = g[count_col].value_counts()
        best_ratio, best_value, best_rows = 0.0, None, 0
        for v, n in vc.items():
            if v < 2 or n < HEAP_MIN_ROWS:
                continue
            ratio = n / max((vc.get(v - 1, 0) + vc.get(v + 1, 0)) / 2, 1)
            if ratio > best_ratio:
                best_ratio, best_value, best_rows = ratio, int(v), int(n)
        rows.append({"city": city, "heaped_value": best_value, "rows_with_value": best_rows,
                     "ratio_to_neighbours": round(best_ratio, 2), "count_heaping": best_ratio >= HEAP_RATIO})
    return pd.DataFrame(rows).sort_values("ratio_to_neighbours", ascending=False)


def coverage_table(df, quarters, city_col="REFERRING_CITY", count_col="REFERRALS"):
    """City-quarter totals relative to the city's median quarter."""
    cq = df.pivot_table(index=city_col, columns="QUARTER", values=count_col, aggfunc="sum",
                        fill_value=0).reindex(columns=quarters, fill_value=0)
    ratio = cq.div(cq.median(axis=1), axis=0)
    long = cq.stack().rename("referrals").to_frame().join(ratio.stack().rename("ratio_to_median"))
    long["coverage_flag"] = (long.ratio_to_median < COVERAGE_LOW) | (long.ratio_to_median > COVERAGE_HIGH)
    return long.reset_index().rename(columns={city_col: "city"})


# ----------------------------------------------------------------------------------------------
# Stage 1 - Loading
# ----------------------------------------------------------------------------------------------
def stage1_load(raw_dir):
    print("\n=== Stage 1: loading ===")
    pattern = re.compile(r"(\d{4})\s*Q\s*([1-4])", re.IGNORECASE)
    files = []
    for f in sorted(os.listdir(raw_dir)):
        if not f.lower().endswith((".xlsx", ".xls")) or f.startswith("~$"):
            continue
        m = pattern.search(f)
        if m is None:
            print("  WARNING - no year/quarter in file name, skipped:", f)
            continue
        files.append({"file": f, "year": int(m.group(1)), "q": int(m.group(2))})
    if not files:
        sys.exit(f"No Excel files found in {raw_dir}. Put the five quarterly files there (see README).")
    info = pd.DataFrame(files).sort_values(["year", "q"]).reset_index(drop=True)
    info["QUARTER"] = info.year.astype(str) + "-Q" + info.q.astype(str)
    info["TIME_INDEX"] = range(1, len(info) + 1)
    if not info.QUARTER.is_unique:
        sys.exit("Two files have the same quarter - check the file names.")

    parts, columns = [], {}
    for _, r in info.iterrows():
        x = pd.read_excel(Path(raw_dir) / r.file, sheet_name=0, dtype=str)   # text: no date conversion
        columns[r.QUARTER] = list(x.columns)
        x.columns = [standardize_col(c) for c in x.columns]
        for c in EXPECTED:
            if c not in x.columns:
                x[c] = ""                                   # e.g. PROVIDER_REGION before Q4 2025
        x = x[EXPECTED].fillna("")
        x["QUARTER"], x["TIME_INDEX"], x["SOURCE_FILE"] = r.QUARTER, r.TIME_INDEX, r.file
        parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    print(f"  {len(info)} files, {len(df):,} rows, {int(to_count(df.TOTAL_REFERRALS).sum()):,} referrals")
    col_table = pd.DataFrame({q: ["yes" if c in [standardize_col(k) for k in cols] else "" for c in EXPECTED]
                              for q, cols in columns.items()}, index=EXPECTED)
    return df, info, col_table


# ----------------------------------------------------------------------------------------------
# Stage 2 - Data-quality review (report only)
# ----------------------------------------------------------------------------------------------
def stage2_quality(df, quarters, report_dir):
    print("\n=== Stage 2: data-quality review (no changes) ===")
    ref = to_count(df.TOTAL_REFERRALS)
    issues, tables = [], {}

    def add(check, column, finding, mask, fix):
        mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
        issues.append({"check": check, "column": column, "finding": finding, "rows_affected": int(mask.sum()),
                       "referrals_affected": int(ref[mask].fillna(0).sum()), "proposed_fix": fix})

    # missing values (PROVIDER_REGION is expected to be empty before the last quarter)
    miss = pd.DataFrame({c: is_missing(df[c]) for c in EXPECTED})
    tables["missing_count"] = miss.groupby(df.QUARTER).sum().reindex(quarters).T.reset_index()
    for c in EXPECTED:
        m = miss[c] & (df.QUARTER == quarters[-1]) if c == "PROVIDER_REGION" else miss[c]
        if m.any():
            add("Missing values", c, f"{int(m.sum()):,} missing values", m,
                "Drop rows" if c == "TOTAL_REFERRALS" else "Review")
    if miss.PROVIDER_REGION.any():
        add("Missing values (expected)", "PROVIDER_REGION", "Recorded only in the last quarter",
            miss.PROVIDER_REGION & (df.QUARTER != quarters[-1]), "Fill from the city-to-region lookup (Stage 3)")

    # duplicates within a quarter
    key = [c for c in EXPECTED if c != "TOTAL_REFERRALS"] + ["QUARTER"]
    exact_extra = df.duplicated(EXPECTED + ["QUARTER"], keep="first")
    key_dup = df.duplicated(key, keep=False) & ~df.duplicated(EXPECTED + ["QUARTER"], keep=False)
    normed = df[key].apply(lambda s: s.map(norm))
    hidden = normed.duplicated(keep=False) & ~df.duplicated(key, keep=False)
    tables["duplicates"] = pd.DataFrame({"exact extra copies": exact_extra.groupby(df.QUARTER).sum(),
                                         "same group, different counts": key_dup.groupby(df.QUARTER).sum(),
                                         "hidden by spaces/case": hidden.groupby(df.QUARTER).sum()}
                                        ).reindex(quarters).fillna(0).astype(int).reset_index()
    if exact_extra.any():
        add("Duplicates", "all", "Exact duplicate rows", exact_extra, "Keep one copy")
    if key_dup.any():
        add("Duplicates", "group key", "Same group with different counts", key_dup, "Sum counts per group")
    if hidden.any():
        add("Duplicates", "group key", "Duplicates hidden by spaces/case", hidden, "Clean text, then sum")

    # spelling variants of city names (dictionary + fuzzy pairs)
    for c in CITY_COLS:
        raw = df[c][~is_missing(df[c])]
        std = clean_names(df[c], CITY_FIXES)[~is_missing(df[c])]
        changed = raw.map(lambda x: re.sub(r"\s+", " ", x.strip())) != std
        tables[f"name_changes_{c[:9].lower()}"] = (pd.DataFrame({"found": raw[changed], "standard": std[changed],
                                                                 "ref": ref[changed.index[changed]]})
                                                   .groupby(["found", "standard"]).ref.agg(["size", "sum"])
                                                   .rename(columns={"size": "rows", "sum": "referrals"}).reset_index())
        if changed.any():
            add("Spelling variants", c, f"{raw[changed].nunique()} spellings standardised",
                changed.reindex(df.index, fill_value=False), "Standardise with the city dictionary")
    names = sorted(set(df[CITY_COLS].stack()) - {""})
    sc = process.cdist(names, names, scorer=fuzz.token_sort_ratio, processor=utils.default_process,
                       score_cutoff=85)
    tables["fuzzy_city_pairs"] = pd.DataFrame([{"name_1": names[i], "name_2": names[j], "similarity": int(sc[i, j])}
                                               for i in range(len(names)) for j in range(i + 1, len(names))
                                               if sc[i, j] >= 85], columns=["name_1", "name_2", "similarity"])

    # unexpected categories
    g = df.GENDER.str.strip().str.upper()
    if (~g.isin({"F", "M"}) & ~is_missing(df.GENDER)).any():
        add("Unexpected value", "GENDER", "Values other than F/M", ~g.isin({"F", "M"}) & ~is_missing(df.GENDER),
            "Map or label Unknown")
    age = df.AGE_GROUP.str.strip()
    tables["age_groups"] = pd.crosstab(age, df.QUARTER).reindex(columns=quarters, fill_value=0).reset_index()
    if (~age.isin(AGE_GROUPS) & ~is_missing(df.AGE_GROUP)).any():
        add("Unexpected value", "AGE_GROUP", "Bands outside the 8 standard bands (e.g. Excel dates)",
            ~age.isin(AGE_GROUPS) & ~is_missing(df.AGE_GROUP), "Map to the standard bands")
    pr = clean_names(df.PROVIDER_REGION, REGION_FIXES)
    if (~pr.isin(REGIONS + [""])).any():
        add("Unexpected value", "PROVIDER_REGION", "Not one of the 5 regions", ~pr.isin(REGIONS + [""]), "Review")
    central = df.PROVIDER_REGION.map(norm).isin({"central region", "centeral region"})
    if central.any():
        add("Synonym", "PROVIDER_REGION", "'Central Region' used for 'Middle Region'", central, "Rename")
    for c in CITY_COLS:
        reg_name = df[c].map(norm).isin({"east", "eastern", "eastern region"})
        if reg_name.any():
            add("Region name in city column", c, "Eastern Province directorate unit", reg_name,
                "Merge with Al-Ahsa")

    # invalid and extreme counts
    present = ~is_missing(df.TOTAL_REFERRALS)
    for name, m, fix in [("not a number", present & ref.isna(), "Drop"), ("zero", ref == 0, "Drop"),
                         ("negative", ref < 0, "Drop"), ("decimal", ref.notna() & (ref % 1 != 0), "Inspect")]:
        if m.any():
            add("Invalid count", "TOTAL_REFERRALS", name, m, fix)
    p999 = ref.quantile(0.999)
    add("Extreme counts", "TOTAL_REFERRALS", f"Counts above the 99.9th percentile ({p999:,.0f})", ref > p999,
        "Keep if plausible (listed in the report)")
    tables["extreme_counts"] = df.loc[ref > p999].assign(REF=ref[ref > p999]).sort_values("REF", ascending=False)
    tables["count_distribution"] = ref.groupby(df.QUARTER).describe(percentiles=[.5, .9, .99, .999]
                                                                    ).reindex(quarters).reset_index()

    # malformed codes
    code = df.DISEASE_CODE.str.strip().str.upper()
    bad = ~code.str.fullmatch(r"[A-Z][0-9][0-9A-Z](\.?[0-9A-Z]{1,4})?") & ~is_missing(df.DISEASE_CODE)
    if bad.any():
        add("Format", "DISEASE_CODE", "Malformed ICD-10 codes", bad, "Review")
    if (is_missing(df.DISEASE_CODE)).any():
        add("Missing values", "DISEASE_CODE", "Missing code", is_missing(df.DISEASE_CODE), "Review")
    tables["malformed_codes"] = (pd.DataFrame({"code": df.DISEASE_CODE[bad], "ref": ref[bad]})
                                 .groupby("code").ref.agg(["size", "sum"]).reset_index())

    issues = pd.DataFrame(issues, columns=["check", "column", "finding", "rows_affected", "referrals_affected",
                                           "proposed_fix"])
    issues.insert(0, "issue", range(1, len(issues) + 1))
    with pd.ExcelWriter(report_dir / "stage2_data_quality_report.xlsx") as xw:
        issues.to_excel(xw, sheet_name="ISSUES_SUMMARY", index=False)
        for k, t in tables.items():
            t.to_excel(xw, sheet_name=k[:31], index=False)
    print(issues[["check", "column", "finding", "rows_affected", "referrals_affected"]].to_string(index=False))
    return issues


# ----------------------------------------------------------------------------------------------
# Stage 3 - Cleaning and region assignment
# ----------------------------------------------------------------------------------------------
def stage3_clean(df, quarters, report_dir):
    print("\n=== Stage 3: cleaning and region assignment ===")
    log = []
    n_rows_raw, ref_raw = len(df), int(to_count(df.TOTAL_REFERRALS).sum())

    # harmonise names (the Eastern unit is still separate here)
    df["REF"] = to_count(df.TOTAL_REFERRALS)
    for c in CITY_COLS:
        df[c] = clean_names(df[c], CITY_FIXES)
    df["PROVIDER_REGION"] = clean_names(df.PROVIDER_REGION, REGION_FIXES)
    df["GENDER"] = df.GENDER.str.strip().str.upper()
    df["AGE_GROUP"] = df.AGE_GROUP.str.strip()
    df["DISEASE_CODE"] = df.DISEASE_CODE.str.strip().str.upper()
    df["TOTAL_REFERRALS"] = df.TOTAL_REFERRALS.str.strip()

    # exact duplicate rows (export errors): keep one copy
    dup = df.duplicated(EXPECTED + ["QUARTER"], keep="first")
    log.append({"step": "Remove exact duplicate rows", "rows_removed": int(dup.sum()),
                "referrals_removed": int(df.loc[dup, "REF"].fillna(0).sum())})
    df = df[~dup].copy()

    # rows with missing, non-numeric, zero or negative counts
    bad = df.REF.isna() | (df.REF <= 0)
    log.append({"step": "Remove rows with missing, non-numeric, zero or negative counts",
                "rows_removed": int(bad.sum()), "referrals_removed": int(df.loc[bad, "REF"].fillna(0).clip(lower=0).sum())})
    df = df[~bad].copy()
    df["REF"] = df.REF.astype(int)

    # merge the Eastern directorate unit with Al-Ahsa
    for c in CITY_COLS:
        df[c] = df[c].replace(MERGE_UNITS)

    # city -> region lookup from the last quarter (majority region of each city's referrals)
    ref_q = quarters[-1]
    q4 = df[(df.QUARTER == ref_q) & (df.PROVIDER_REGION != "")]
    cr = q4.pivot_table(index="REFERRING_PROVIDER_REGION", columns="PROVIDER_REGION", values="REF",
                        aggfunc="sum", fill_value=0)
    lookup = pd.DataFrame({"majority_region": cr.idxmax(axis=1), "referrals": cr.sum(axis=1),
                           "majority_share_%": (cr.max(axis=1) / cr.sum(axis=1) * 100).round(2),
                           "n_regions": (cr > 0).sum(axis=1)})
    LOOKUP = lookup.majority_region.to_dict()
    cities = sorted((set(df.REFERRING_PROVIDER_REGION) | set(df.RECEIVING_PROVIDER_REGION)) - {""})
    city_region = []
    for c in cities:
        lk, geo, man = LOOKUP.get(c), GEOGRAPHY.get(c), MANUAL_REGION.get(c)
        final = man or lk or geo
        city_region.append({"city": c, "lookup_Q4": lk, "geography": geo, "manual": man, "final_region": final,
                            "source": "manual" if man else "lookup_Q4" if lk else "geography" if geo else "UNMAPPED",
                            "lookup_agrees_with_geography": (lk == geo) if lk and geo else None})
    city_region = pd.DataFrame(city_region)
    unmapped = city_region.loc[city_region.final_region.isna(), "city"].tolist()
    if unmapped:
        sys.exit(f"Cities without a region: {unmapped}. Add them to GEOGRAPHY or MANUAL_REGION.")
    FINAL = dict(zip(city_region.city, city_region.final_region))
    SOURCE = dict(zip(city_region.city, city_region.source))

    # lookup accuracy in the last quarter
    pred = q4.REFERRING_PROVIDER_REGION.map(FINAL)
    acc = q4.loc[pred == q4.PROVIDER_REGION, "REF"].sum() / q4.REF.sum() * 100
    n_multi = int((lookup.n_regions > 1).sum())
    print(f"  Lookup accuracy in {ref_q}: {acc:.2f}% of referrals; cities linked to >1 region: {n_multi}")

    orig = df.PROVIDER_REGION != ""
    df["REFERRING_REGION_SOURCE"] = np.where(orig, "original", df.REFERRING_PROVIDER_REGION.map(SOURCE))
    df.loc[~orig, "PROVIDER_REGION"] = df.loc[~orig, "REFERRING_PROVIDER_REGION"].map(FINAL)

    # merge rows describing the same group; the referral total must not change
    df = df.rename(columns={"REFERRING_PROVIDER_REGION": "REFERRING_CITY", "RECEIVING_PROVIDER_REGION": "RECEIVING_CITY",
                            "PROVIDER_REGION": "REFERRING_REGION"})
    df["RECEIVING_REGION"] = df.RECEIVING_CITY.map(FINAL)
    key = ["QUARTER", "TIME_INDEX", "GENDER", "AGE_GROUP", "DISEASE_CODE", "REFERRING_CITY", "REFERRING_REGION",
           "RECEIVING_CITY", "RECEIVING_REGION"]
    total_before = df.REF.sum()
    n_before = len(df)
    clean = (df.groupby(key, as_index=False).agg(REFERRALS=("REF", "sum"),
                                                 REFERRING_REGION_SOURCE=("REFERRING_REGION_SOURCE", "first")))
    assert clean.REFERRALS.sum() == total_before, "Referral total changed while merging groups"
    log.append({"step": "Merge rows describing the same group (counts summed)", "rows_removed": n_before - len(clean),
                "referrals_removed": 0})
    log = pd.DataFrame(log)
    print(log.to_string(index=False))
    print(f"  Rows {n_rows_raw:,} -> {len(clean):,}; referrals {ref_raw:,} -> {int(clean.REFERRALS.sum()):,}")

    with pd.ExcelWriter(report_dir / "stage3_mapping_report.xlsx") as xw:
        log.to_excel(xw, sheet_name="cleaning_log", index=False)
        lookup.reset_index().rename(columns={"REFERRING_PROVIDER_REGION": "city"}).to_excel(
            xw, sheet_name="lookup_last_quarter", index=False)
        city_region.to_excel(xw, sheet_name="city_region_final", index=False)
        pd.DataFrame(sorted(CITY_FIXES.items()), columns=["found_as", "standard_name"]).to_excel(
            xw, sheet_name="city_dictionary", index=False)
    stats = {"rows_raw": n_rows_raw, "referrals_raw": ref_raw,
             "dup_rows": int(log.rows_removed[0]), "dup_referrals": int(log.referrals_removed[0]),
             "invalid_rows": int(log.rows_removed[1]), "invalid_referrals": int(log.referrals_removed[1]),
             "rows_clean": len(clean), "referrals_clean": int(clean.REFERRALS.sum()),
             "lookup_accuracy_pct": round(float(acc), 2), "cities_multi_region": n_multi,
             "cities_geography_only": city_region.loc[city_region.source == "geography", "city"].tolist()}
    return clean, stats


# ----------------------------------------------------------------------------------------------
# Stage 4 - Diagnosis coding (ICD-10-AM chapter ranges)
# ----------------------------------------------------------------------------------------------
def stage4_icd(clean, tabular_list, report_dir):
    print("\n=== Stage 4: ICD-10-AM chapters ===")
    codes = pd.Series(sorted(clean.DISEASE_CODE.unique()))
    res = codes.map(icd10am_chapter)
    invalid = codes[res.isna()].tolist()
    if invalid:
        print(f"  WARNING - {len(invalid)} codes with an invalid format (kept as 'OTHER'):", invalid[:20])
    chap = pd.DataFrame({"DISEASE_CODE": codes,
                         "ICD_CATEGORY": codes.str.replace(".", "", regex=False).str[:3],
                         "ICD_CHAPTER": [r[0] if r else "OTHER" for r in res],
                         "ICD_CHAPTER_NAME": [r[1] if r else "Other (special/unclassified)" for r in res]})
    chap["ICD_CHAPTER_SHORT"] = chap.ICD_CHAPTER.map(CHAPTER_SHORT)
    clean = clean.merge(chap, on="DISEASE_CODE", how="left")

    ksa = clean[~clean[["REFERRING_CITY", "RECEIVING_CITY"]].eq("Outside KSA").any(axis=1)]   # shares as in the paper
    tot = ksa.REFERRALS.sum()
    summary = (ksa.groupby(["ICD_CHAPTER", "ICD_CHAPTER_SHORT"]).agg(codes=("DISEASE_CODE", "nunique"),
                                                                       referrals=("REFERRALS", "sum"))
               .reset_index())
    summary["pct_referrals"] = (summary.referrals / tot * 100).round(3)
    out = {"supp_pct": float(summary.loc[summary.ICD_CHAPTER == "SUPP", "referrals"].sum() / tot * 100),
           "other_pct": float(summary.loc[summary.ICD_CHAPTER == "OTHER", "referrals"].sum() / tot * 100),
           "other_codes": sorted(ksa.loc[ksa.ICD_CHAPTER == "OTHER", "ICD_CATEGORY"].unique().tolist())}

    # optional: validation against the licensed ICD-10-AM tabular list (not distributed with this code)
    tab = Path(tabular_list) if tabular_list else None
    if tab is not None and tab.exists():
        listed = set(pd.read_csv(tab, dtype=str)["code"].str.strip().str.upper().str.replace(".", "", regex=False))
        flat = ksa.DISEASE_CODE.str.replace(".", "", regex=False)
        exact = flat.isin(listed)
        parent = ~exact & (flat.str[:4].isin(listed) | flat.str[:3].isin(listed))
        out["tabular_listed_pct"] = float(ksa.loc[exact, "REFERRALS"].sum() / tot * 100)
        out["tabular_parent_pct"] = float(ksa.loc[parent, "REFERRALS"].sum() / tot * 100)
        print(f"  Tabular list: listed code {out['tabular_listed_pct']:.2f}%, listed parent {out['tabular_parent_pct']:.2f}%")
    else:
        print("  (No ICD-10-AM tabular list supplied (--icd10am) - code-level validation skipped.)")
    summary.to_csv(report_dir / "stage4_icd_chapters.csv", index=False)
    chap.to_csv(report_dir / "stage4_code_to_chapter.csv", index=False)
    print(f"  {len(codes):,} codes; supplementary U78-U88 {out['supp_pct']:.2f}%; unclassified {out['other_pct']:.3f}%"
          f" ({', '.join(out['other_codes'])})")
    return clean, out


# ----------------------------------------------------------------------------------------------
# Stage 5 - Data-quality flags and analysis datasets
# ----------------------------------------------------------------------------------------------
def stage5_datasets(clean, quarters, report_dir, data_dir):
    print("\n=== Stage 5: flags and analysis datasets ===")
    clean = clean.sort_values(["TIME_INDEX", "REFERRING_CITY", "RECEIVING_CITY", "GENDER", "AGE_GROUP",
                               "DISEASE_CODE"]).reset_index(drop=True)
    clean.to_csv(data_dir / "referrals_clean.csv", index=False, encoding="utf-8-sig")

    outside = clean[["REFERRING_CITY", "RECEIVING_CITY"]].eq("Outside KSA").any(axis=1)
    out_stats = {"outside_rows": int(outside.sum()), "outside_referrals": int(clean.loc[outside, "REFERRALS"].sum())}
    q = clean[~outside].copy()
    q["CROSS_CITY"] = (q.REFERRING_CITY != q.RECEIVING_CITY).astype(int)
    q["CROSS_REGION"] = (q.REFERRING_REGION != q.RECEIVING_REGION).astype(int)

    heap = heaping_table(q)
    cov = coverage_table(q, quarters)
    heap_cities = sorted(heap.loc[heap.count_heaping, "city"])
    flagged = set(map(tuple, cov.loc[cov.coverage_flag, ["city", "QUARTER"]].values))
    q["HEAPING_CITY"] = q.REFERRING_CITY.isin(heap_cities).astype(int)
    q["COVERAGE_FLAG"] = [int((c, x) in flagged) for c, x in zip(q.REFERRING_CITY, q.QUARTER)]
    q.to_csv(data_dir / "referrals_quarterly.csv", index=False, encoding="utf-8-sig")
    heap.to_csv(report_dir / "stage5_count_heaping.csv", index=False)
    cov.to_csv(report_dir / "stage5_reporting_coverage.csv", index=False)
    print(f"  Count-heaping cities ({len(heap_cities)}): {', '.join(heap_cities)}")
    print(f"  City-quarters with coverage changes ({len(flagged)}): "
          + "; ".join(f"{c} {x}" for c, x in sorted(flagged)))

    # analysis dataset: aggregated over quarters and genders
    key = ["REFERRING_CITY", "REFERRING_REGION", "RECEIVING_CITY", "RECEIVING_REGION", "AGE_GROUP", "DISEASE_CODE",
           "ICD_CATEGORY", "ICD_CHAPTER", "ICD_CHAPTER_SHORT"]
    # N_OCCURRENCES: number of quarter x gender cells (5 x 2) in which the code appeared in this cell
    a = q.groupby(key, as_index=False).agg(REFERRALS=("REFERRALS", "sum"), N_OCCURRENCES=("REFERRALS", "size"),
                                           N_QUARTERS=("QUARTER", "nunique"), N_GENDERS=("GENDER", "nunique"))
    a.insert(0, "REFERRAL_TYPE", np.where(a.REFERRING_CITY == a.RECEIVING_CITY, "Same-city", "Cross-city"))
    a["CROSS_CITY"] = (a.REFERRAL_TYPE == "Cross-city").astype(int)
    a["CROSS_REGION"] = (a.REFERRING_REGION != a.RECEIVING_REGION).astype(int)
    a["REPEATED"] = (a.N_OCCURRENCES > 1).astype(int)
    grp = ["REFERRAL_TYPE", "REFERRING_CITY", "RECEIVING_CITY", "AGE_GROUP"]       # one cell = one group of codes
    a["CODES_IN_GROUP"] = a.groupby(grp).DISEASE_CODE.transform("size")
    a["SHARE_IN_GROUP_%"] = (a.REFERRALS / a.groupby(grp).REFERRALS.transform("sum") * 100).round(2)
    a = a.sort_values(grp + ["REFERRALS"], ascending=[True] * 4 + [False]).reset_index(drop=True)
    assert a.REFERRALS.sum() == q.REFERRALS.sum()
    a.to_csv(data_dir / "referrals_analysis.csv", index=False, encoding="utf-8-sig")
    st = {**out_stats, "analysis_rows_quarterly": len(q), "analysis_referrals": int(q.REFERRALS.sum()),
          "analysis_records": len(a), "codes": int(a.DISEASE_CODE.nunique()),
          "referring_cities": int(a.REFERRING_CITY.nunique()), "receiving_cities": int(a.RECEIVING_CITY.nunique()),
          "heaping_cities": heap_cities, "flagged_city_quarters": [f"{c} {x}" for c, x in sorted(flagged)]}
    print(f"  Outside KSA excluded: {st['outside_rows']} rows, {st['outside_referrals']:,} referrals")
    print(f"  Analysis dataset: {st['analysis_records']:,} records, {st['analysis_referrals']:,} referrals, "
          f"{st['codes']:,} codes, {st['referring_cities']} referring / {st['receiving_cities']} receiving cities")
    return st


def flow_figure(s, path):
    """Figure S1: flow of rows and referrals from the raw extracts to the analysis dataset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(18, 10.9), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1800); ax.set_ylim(1090, 0); ax.axis("off")

    def box(x0, y0, x1, y1, text, main=True, fs=22):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=14",
                                    fc="#EEF3F8" if main else "#FCF3E8", ec="#3A6EA5" if main else "#C27C2C", lw=2.6))
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.35)
    box(205, 25, 1005, 135, f"Five quarterly extracts, {s['first_quarter']} – {s['last_quarter']}\n"
                            f"{s['rows_raw']:,} rows; {s['referrals_raw']:,} referrals")
    box(1077, 250, 1772, 372, f"Exact duplicate rows removed: {s['dup_rows']:,} rows, {s['dup_referrals']:,} referrals\n"
                              f"Rows with invalid counts removed\n(missing, non-numeric, zero, negative): "
                              f"{s['invalid_rows']:,}", False, 18.5)
    box(65, 468, 1145, 617, "Cleaned records (names harmonised, Eastern merged with\nAl-Ahsa, regions assigned, "
                            f"duplicate groups merged)\n{s['rows_clean']:,} rows; {s['referrals_clean']:,} referrals")
    box(1235, 730, 1615, 818, f"Outside KSA excluded:\n{s['outside_rows']:,} rows; {s['outside_referrals']:,} referrals",
        False, 18.5)
    box(78, 932, 1133, 1080, "Analysis dataset (aggregated over quarters and genders)\n"
                             f"{s['analysis_records']:,} code-level records; {s['analysis_referrals']:,} referrals\n"
                             f"{s['referring_cities']} referring and {s['receiving_cities']} receiving cities; "
                             f"{s['codes']:,} ICD-10 codes")
    k = dict(arrowstyle="->", lw=2.6, color="black", mutation_scale=26)
    ax.annotate("", (605, 468), (605, 145), arrowprops=k); ax.annotate("", (605, 932), (605, 607), arrowprops=k)
    k2 = dict(arrowstyle="->", lw=2.6, color="#C27C2C", mutation_scale=26)
    ax.annotate("", (1100, 311), (620, 311), arrowprops=k2); ax.annotate("", (1100, 774), (620, 774), arrowprops=k2)
    fig.savefig(path, dpi=100, facecolor="white"); plt.close(fig)


def methods_text(s):
    return f"""# Values for the Methods section (generated by 01_data_preparation.py)

- Raw extracts: {s['n_files']} files ({s['first_quarter']} to {s['last_quarter']}), {s['rows_raw']:,} rows, {s['referrals_raw']:,} referrals.
- Exact duplicate rows removed: {s['dup_rows']:,} rows ({s['dup_referrals']:,} referrals); rows with missing, non-numeric, zero or negative counts removed: {s['invalid_rows']:,}.
- Cleaned dataset: {s['rows_clean']:,} rows, {s['referrals_clean']:,} referrals.
- City-to-region lookup (built from {s['last_quarter']}) reproduced the recorded region for {s['lookup_accuracy_pct']:.2f}% of referrals in {s['last_quarter']}; {s['cities_multi_region']} cities were linked to more than one region. Cities assigned from administrative geography: {', '.join(s['cities_geography_only']) or 'none'}.
- ICD-10-AM supplementary codes (U78-U88): {s['supp_pct']:.1f}% of referrals; codes outside the ICD-10-AM chapters ({', '.join(s['other_codes'])}): {s['other_pct']:.2f}%.
- Validation against the ICD-10-AM tabular list: {('%.1f%% of referrals carried a listed code and %.1f%% a listed parent code' % (s['tabular_listed_pct'], s['tabular_parent_pct'])) if 'tabular_listed_pct' in s else 'not run (licensed list not supplied)'}.
- Referrals involving facilities outside Saudi Arabia excluded: {s['outside_referrals']:,} referrals ({s['outside_rows']} rows).
- Analysis dataset: {s['analysis_records']:,} records, {s['analysis_referrals']:,} referrals, {s['codes']:,} codes, {s['referring_cities']} referring and {s['receiving_cities']} receiving cities.
- Count heaping (value >= {HEAP_RATIO}x as frequent as its neighbours within a city): {', '.join(s['heaping_cities'])}.
- Reporting-coverage changes (city-quarter total < {COVERAGE_LOW} or > {COVERAGE_HIGH} x the city's median quarter): {len(s['flagged_city_quarters'])} city-quarters ({'; '.join(s['flagged_city_quarters'])}).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw", help="folder with the five quarterly Excel files")
    ap.add_argument("--out-dir", default=".", help="project folder for data/processed and reports/")
    ap.add_argument("--icd10am", default=None,
                    help="optional CSV of the licensed ICD-10-AM tabular list (column 'code') to validate codes")
    args = ap.parse_args()
    data_dir = Path(args.out_dir) / "data" / "processed"
    report_dir = Path(args.out_dir) / "reports" / "preprocessing"
    data_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)

    raw, info, col_table = stage1_load(args.raw_dir)
    quarters = info.QUARTER.tolist()
    col_table.to_csv(report_dir / "stage1_columns_by_quarter.csv")
    stage2_quality(raw, quarters, report_dir)
    clean, s3 = stage3_clean(raw.copy(), quarters, report_dir)
    clean, s4 = stage4_icd(clean, args.icd10am, report_dir)
    s5 = stage5_datasets(clean, quarters, report_dir, data_dir)

    s = {"n_files": len(info), "first_quarter": quarters[0], "last_quarter": quarters[-1], **s3, **s4, **s5}
    json.dump(s, open(report_dir / "preprocessing_summary.json", "w"), indent=1, default=str)
    (report_dir / "methods_text.md").write_text(methods_text(s), encoding="utf-8")
    flow_figure(s, report_dir / "figure_S1_flow.png")
    print(f"\nDone. Datasets in {data_dir}, reports in {report_dir}.")


if __name__ == "__main__":
    main()
