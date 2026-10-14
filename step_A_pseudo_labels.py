"""
RSNA Knee — Improvement Module A: Multilingual Report → Pseudo-Label Extraction
WHY THIS MATTERS: We have 58 labeled studies. Reports cover 4,407 studies.
Extracting even noisy labels from reports = 75x more training data = biggest ROC-AUC jump.
OFFLINE: Regex only. No internet required.
Run this ONCE in Kaggle, save output as pseudo_labels.csv, then use in train_v3.
"""
import os
import re
import pandas as pd
import numpy as np

print("=" * 70)
print("RSNA Knee: Pseudo-Label Extraction from Multilingual Reports")
print("=" * 70)

comp_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'sample_submission.csv' in files:
        comp_dir = root
        break

train_df = pd.read_csv(os.path.join(comp_dir, 'train.csv'))
print(f"Loaded train.csv: {train_df.shape}")
print(f"Columns: {list(train_df.columns)}")

# Check which column holds the report text
report_col = None
for candidate in ['Report', 'report', 'ReportText', 'report_text', 'findings', 'Findings']:
    if candidate in train_df.columns:
        report_col = candidate
        print(f"Found report column: '{report_col}'")
        break

if report_col is None:
    # Show all columns to identify the text column
    print("\nCould not auto-detect report column. Showing first 3 rows:")
    print(train_df.head(3).to_string())
    raise ValueError("Update 'report_col' variable below with the correct column name!")

# -----------------------------------------------------------------------
# MULTILINGUAL KEYWORD PATTERNS PER FINDING
# Encoding: 1 = present, 0 = absent/normal, -1 = not mentioned (default)
# Strategy: match POSITIVE patterns → label=1
#           match NEGATIVE/INTACT patterns → label=0
#           no match → label=-1 (masked out during training)
# -----------------------------------------------------------------------

# NEGATION words in 12 languages (English, German, French, Spanish, Portuguese,
# Italian, Dutch, Japanese, Chinese, Korean, Russian, Arabic)
NEGATION = [
    # English
    r'\bno\b', r'\bnot\b', r'\bwithout\b', r'\bnormal\b', r'\bintact\b',
    r'\bundamaged\b', r'\bnegative\b', r'\bunremarkable\b', r'\bno evidence of\b',
    # German
    r'\bkein\b', r'\bkeine\b', r'\bohne\b', r'\bunauffällig\b', r'\bintakt\b',
    # French
    r'\bpas\b', r'\bsans\b', r'\bnormal\b', r'\bintact\b', r'\bnégatif\b',
    # Spanish
    r'\bno\b', r'\bsin\b', r'\bnormal\b', r'\bíntegro\b', r'\bintegro\b',
    # Portuguese
    r'\bsem\b', r'\bnormal\b', r'\bíntegro\b',
    # Italian
    r'\bnon\b', r'\bsenza\b', r'\bnormale\b', r'\bintegro\b',
    # Dutch
    r'\bgeen\b', r'\bniet\b', r'\bnormaal\b', r'\bintact\b',
    # Russian (transliterated)
    r'\bнет\b', r'\bбез\b', r'\bинтактна\b',
]

NEGATION_PATTERN = re.compile('|'.join(NEGATION), re.IGNORECASE)


def has_negation(text: str, keyword_match_start: int, window: int = 60) -> bool:
    """Check if any negation word appears within `window` chars BEFORE the keyword."""
    pre_window = text[max(0, keyword_match_start - window): keyword_match_start]
    return bool(NEGATION_PATTERN.search(pre_window))


# Finding-specific keyword rules: (positive_patterns, negative/normal_patterns)
FINDING_RULES = {
    'ACL': {
        'positive': [
            # English
            r'\bacl\s*(tear|rupture|injury|damage|laxity|sprain|disruption|instability)\b',
            r'\banterior\s+cruciate\s*(ligament)?\s*(tear|rupture|injury|damage)\b',
            r'\bacl\s*ist\s*(gerissen|rupturiert)\b',  # German
            r'\bligament(o|e)?\s+crois[eé]\s+ant[eé]rieur.*(rupture|lésion|déchirure)\b',  # French
            r'\bligamento\s+cruzado\s+anterior.*(rotura|ruptura|lesión)\b',  # Spanish
        ],
        'negative': [
            r'\bacl\s*(intact|normal|unremarkable|ohne befund)\b',
            r'\banteriores\s+kreuzband\s*(intakt|unauffällig)\b',
            r'\bacl\s*en\s*(buen estado|bonne état)\b',
        ]
    },
    'MCL': {
        'positive': [
            r'\bmcl\s*(tear|rupture|injury|sprain|damage)\b',
            r'\bmedial\s+collateral\s*(ligament)?\s*(tear|rupture|injury|sprain)\b',
            r'\binnenband\s*(riss|ruptur|verletzung)\b',  # German
            r'\bligament\s+collatéral\s+médial.*(rupture|lésion|déchirure)\b',  # French
            r'\bligamento\s+colateral\s+medial.*(rotura|ruptura|lesión)\b',  # Spanish
        ],
        'negative': [
            r'\bmcl\s*(intact|normal|unremarkable)\b',
            r'\bmedial\s+collateral.*(intact|normal)\b',
            r'\binnenband\s*(intakt|unauffällig)\b',
        ]
    },
    'Medial Meniscus': {
        'positive': [
            r'\bmedial\s+meniscus.*(tear|rupture|injury|lesion|damage|horn|cleavage|bucket)\b',
            r'\b(medial|inner)\s+menisk.*(riss|ruptur|läsion|verletzung)\b',  # German
            r'\bménisque\s+(médial|interne).*(rupture|lésion|déchirure|fissure)\b',  # French
            r'\bmenisco\s+(medial|interno).*(rotura|ruptura|lesión)\b',  # Spanish
            r'\bmeniscal\s+tear.*(medial|inner)\b',
            r'\b内側半月板.*(断裂|損傷)\b',  # Japanese
        ],
        'negative': [
            r'\bmedial\s+meniscus.*(intact|normal|unremarkable|preserved)\b',
            r'\bmedial\s+menisk.*(intakt|unauffällig)\b',
            r'\bménisque\s+(médial|interne).*(normal|intact)\b',
        ]
    },
    'Lateral Meniscus': {
        'positive': [
            r'\blateral\s+meniscus.*(tear|rupture|injury|lesion|damage|horn|cleavage|bucket)\b',
            r'\b(lateral|aussen|außen)\s*menisk.*(riss|ruptur|läsion|verletzung)\b',
            r'\bménisque\s+(latéral|externe).*(rupture|lésion|déchirure|fissure)\b',
            r'\bmenisco\s+(lateral|externo).*(rotura|ruptura|lesión)\b',
            r'\bmeniscal\s+tear.*(lateral|outer)\b',
            r'\b外側半月板.*(断裂|損傷)\b',
        ],
        'negative': [
            r'\blateral\s+meniscus.*(intact|normal|unremarkable|preserved)\b',
            r'\blateral\s+menisk.*(intakt|unauffällig)\b',
            r'\bménisque\s+(latéral|externe).*(normal|intact)\b',
        ]
    },
    'Medial OA': {
        'positive': [
            r'\bmedial\s+(compartment)?\s*(osteoarthritis|OA|arthrosis|arthrose)\b',
            r'\bmedial.*(knorpelschaden|arthrose|degeneration)\b',
            r'\barthr[oô]se\s+(médiale|du compartiment médial)\b',
            r'\bdegenerat\w+\s+(medial|inneren)\s+(compartment|kammer)\b',
            r'\bchondropathie?\s+mediale?\b',
        ],
        'negative': [
            r'\bmedial\s+(compartment)?.*(no|without|kein|sans|sin)\s*(OA|osteoarthritis|arthrosis)\b',
            r'\bmedial\s+(compartment|kammer)?.*(normal|intact|unauffällig)\b',
        ]
    },
    'Lateral OA': {
        'positive': [
            r'\blateral\s+(compartment)?\s*(osteoarthritis|OA|arthrosis|arthrose)\b',
            r'\blateral.*(knorpelschaden|arthrose|degeneration)\b',
            r'\barthr[oô]se\s+(latérale|du compartiment latéral)\b',
            r'\bdegenerat\w+\s+(lateral|äußer)\s+(compartment|kammer)\b',
        ],
        'negative': [
            r'\blateral\s+(compartment)?.*(no|without|kein|sans|sin)\s*(OA|osteoarthritis)\b',
        ]
    },
    'PF OA': {
        'positive': [
            r'\bpatellofemoral.*(osteoarthritis|OA|arthrosis|arthrose|chondromalacia)\b',
            r'\bpf\s+(OA|arthrosis|arthrose)\b',
            r'\bpatellar\s+(chondromalac|osteoarthr)\b',
            r'\barthr[oô]se\s+fémoro.?patellaire\b',
        ],
        'negative': [
            r'\bpatellofemoral.*(normal|intact|unauffällig|no OA)\b',
        ]
    },
    'Effusion': {
        'positive': [
            r'\b(joint\s+)?(effusion|erguss|épanchement|derrame|versamento)\b',
            r'\b(moderate|large|significant|increased|small)\s+(joint\s+)?effusion\b',
            r'\bfluid\s+(in|within)\s+(the\s+)?(joint|knee)\b',
            r'\b滲出液\b',  # Chinese
            r'\b관절삼출\b',  # Korean
        ],
        'negative': [
            r'\bno\s+(joint\s+)?effusion\b',
            r'\bkein\s+(Erguss|Gelenkserguss)\b',
            r'\bpas\s+d.?épanchement\b',
            r'\bsin\s+derrame\b',
            r'\beffusion\s*(absent|none)\b',
        ]
    },
    'Synovitis': {
        'positive': [
            r'\bsynovitis\b', r'\bsynovial\s+(thickening|hypertrophy|proliferation)\b',
            r'\bsynoviale\s+(Entzündung|Verdickung)\b',
            r'\bsynovite\b',  # French
            r'\bsinovitis\b',  # Spanish/Portuguese
        ],
        'negative': [
            r'\bno\s+synovitis\b',
            r'\bkeine\s+synovitis\b',
            r'\bsynovitis\s*(absent|none)\b',
        ]
    },
    "Baker's": {
        'positive': [
            r'\bbaker.?s?\s*(cyst|cyste|cisto|kyste|quiste)\b',
            r'\bpopliteal\s+(cyst|zyste)\b',
            r'\bバーカー嚢胞\b',
        ],
        'negative': [
            r'\bno\s+baker.?s?\s*cyst\b',
            r'\bkein\s+(baker|popliteal)\b',
            r'\bpopliteal\s+(no\s+)?cyst\s*absent\b',
        ]
    },
    'Contusion': {
        'positive': [
            r'\b(bone\s+)?(contusion|marrow\s+edema|bony\s+edema|bruise|Knochenödem|Knochenmarködem)\b',
            r'\bcontusion osseuse\b',
            r'\bcontusión\s+ósea\b',
        ],
        'negative': [
            r'\bno\s+(bone\s+)?(contusion|marrow\s+edema)\b',
            r'\bkein\s+knochenödem\b',
        ]
    },
    'Fracture': {
        'positive': [
            r'\b(fracture|fraktur|fractura|Fraktur|Knochenbruch|break|broken)\b',
            r'\b(tibial\s+plateau|femoral\s+condyle|patell).*(fracture|fraktur)\b',
            r'\b骨折\b',  # Chinese/Japanese
            r'\b골절\b',  # Korean
        ],
        'negative': [
            r'\bno\s+(acute\s+)?(fracture|fraktur)\b',
            r'\bkeine\s+(fraktur|Fraktur)\b',
            r'\bfracture\s*(absent|excluded|none)\b',
        ]
    },
}


def extract_label(text: str, positive_patterns: list, negative_patterns: list) -> int:
    """Returns 1, 0, or -1."""
    text_l = text.lower()
    for np_pattern in negative_patterns:
        if re.search(np_pattern, text_l, re.IGNORECASE):
            return 0
    for pp_pattern in positive_patterns:
        m = re.search(pp_pattern, text_l, re.IGNORECASE)
        if m:
            if not has_negation(text_l, m.start()):
                return 1
    return -1


# -----------------------------------------------------------------------
# EXTRACT PSEUDO LABELS FROM ALL REPORTS
# -----------------------------------------------------------------------
target_cols = list(FINDING_RULES.keys())
print(f"\n[EXTRACTION] Extracting labels from {len(train_df)} reports...")

results = []
for _, row in train_df.iterrows():
    study_id = row['StudyInstanceUID']
    report = str(row.get(report_col, ''))
    
    row_labels = {'StudyInstanceUID': study_id}
    for finding, patterns in FINDING_RULES.items():
        if pd.notna(row.get(finding)):
            # Gold-label exists: use it exactly
            row_labels[finding] = row[finding]
        else:
            # Extract from report
            row_labels[finding] = extract_label(report, patterns['positive'], patterns['negative'])
    results.append(row_labels)

pseudo_df = pd.DataFrame(results)
save_path = '/kaggle/working/pseudo_labels.csv'
pseudo_df.to_csv(save_path, index=False)

print("\n[SUMMARY] Label distribution after extraction:")
for col in target_cols:
    pos = (pseudo_df[col] == 1).sum()
    neg = (pseudo_df[col] == 0).sum()
    mask = (pseudo_df[col] == -1).sum()
    print(f"  {col:<22} positive={pos:4d} | negative={neg:4d} | not_mentioned={mask:4d}")

print(f"\n[SUCCESS] pseudo_labels.csv saved to: {save_path}")
print("Now run wave4_train_v3.py which loads this and trains on ALL studies!")
