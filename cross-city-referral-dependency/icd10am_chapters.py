"""
Assign ICD-10-AM codes to chapters from the chapter code ranges.

No external ICD library is used: each code is classified by its first letter
and two digits, following the ICD-10-AM chapter structure (the chapter ranges
are the same in every ICD-10-AM edition). Supplementary codes for chronic
conditions (U78-U88) form their own group; other U codes (e.g. the activity
codes U53 and U66) are "Other (special/unclassified)".

Usage:
    from icd10am_chapters import icd10am_chapter, CHAPTER_SHORT
    icd10am_chapter("C16.9")   # -> ("II", "Neoplasms")
    CHAPTER_SHORT["II"]        # -> "Neoplasms" (short label used in tables and figures)
"""

import re

# (first code, last code, chapter number, chapter name) - ICD-10-AM chapter ranges
CHAPTERS = [
    ("A00", "B99", "I", "Certain infectious and parasitic diseases"),
    ("C00", "D48", "II", "Neoplasms"),
    ("D50", "D89", "III", "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism"),
    ("E00", "E90", "IV", "Endocrine, nutritional and metabolic diseases"),
    ("F00", "F99", "V", "Mental and behavioural disorders"),
    ("G00", "G99", "VI", "Diseases of the nervous system"),
    ("H00", "H59", "VII", "Diseases of the eye and adnexa"),
    ("H60", "H95", "VIII", "Diseases of the ear and mastoid process"),
    ("I00", "I99", "IX", "Diseases of the circulatory system"),
    ("J00", "J99", "X", "Diseases of the respiratory system"),
    ("K00", "K93", "XI", "Diseases of the digestive system"),
    ("L00", "L99", "XII", "Diseases of the skin and subcutaneous tissue"),
    ("M00", "M99", "XIII", "Diseases of the musculoskeletal system and connective tissue"),
    ("N00", "N99", "XIV", "Diseases of the genitourinary system"),
    ("O00", "O99", "XV", "Pregnancy, childbirth and the puerperium"),
    ("P00", "P96", "XVI", "Certain conditions originating in the perinatal period"),
    ("Q00", "Q99", "XVII", "Congenital malformations, deformations and chromosomal abnormalities"),
    ("R00", "R99", "XVIII", "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified"),
    ("S00", "T98", "XIX", "Injury, poisoning and certain other consequences of external causes"),
    ("V00", "Y98", "XX", "External causes of morbidity and mortality"),
    ("Z00", "Z99", "XXI", "Factors influencing health status and contact with health services"),
]
SUPPLEMENTARY = ("U78", "U88", "SUPP", "Supplementary codes for chronic conditions")
OTHER = ("OTHER", "Other (special/unclassified)")

# Short labels used in the tables and figures of the paper
CHAPTER_SHORT = {
    "I": "Infectious", "II": "Neoplasms", "III": "Blood and immune", "IV": "Endocrine and metabolic",
    "V": "Mental and behavioural", "VI": "Nervous system", "VII": "Eye", "VIII": "Ear", "IX": "Circulatory",
    "X": "Respiratory", "XI": "Digestive", "XII": "Skin", "XIII": "Musculoskeletal", "XIV": "Genitourinary",
    "XV": "Pregnancy and childbirth", "XVI": "Perinatal", "XVII": "Congenital malformations",
    "XVIII": "Symptoms and signs", "XIX": "Injury and poisoning", "XX": "External causes",
    "XXI": "Health-status factors", "SUPP": "Supplementary codes", "OTHER": "Unclassified",
}
CHAPTER_ORDER = [c[2] for c in CHAPTERS] + ["SUPP", "OTHER"]

CODE_PATTERN = re.compile(r"^[A-Z]\d[0-9A-Z](\.?[0-9A-Z]{1,4})?$")


def icd10am_chapter(code):
    """Return (chapter number, chapter name) for an ICD-10-AM code, or None if the format is invalid."""
    c = str(code).strip().upper()
    if not CODE_PATTERN.match(c):
        return None
    category = c.replace(".", "")[:3]          # e.g. "C16.9" -> "C16"
    if SUPPLEMENTARY[0] <= category <= SUPPLEMENTARY[1]:
        return SUPPLEMENTARY[2], SUPPLEMENTARY[3]
    for first, last, number, name in CHAPTERS:
        if first <= category <= last:
            return number, name
    return OTHER


if __name__ == "__main__":
    for example in ["A41.9", "C16.9", "D57.0", "H66.9", "I64", "S72.0", "V99", "Z50.9", "U82.3", "U66.02"]:
        print(f"{example:8} -> {icd10am_chapter(example)}")
