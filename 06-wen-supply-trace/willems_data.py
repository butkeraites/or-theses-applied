"""Fetch and check the Willems (2008) supply-chain data.

The official data is the online appendix of Willems (2008), M&SOM 10(1):19-23,
doi:10.1287/msom.1070.0176, which I could not download (bot check). This uses a public
Google Sheets copy (linked from https://gist.github.com/sderaco/921848b968ffdd54864c80ee0ff28231)
and checks it against what the paper itself prints: Table 1 (chain -> company id, SIC code)
and the stated ranges (8 to 2,025 stages; 10 to 16,225 arcs). The data is NOT redistributed
here: it is downloaded on first run into data/ (git-ignored).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/1vD4UCxsincEQYQpAeCbEqnd3QFdKcuL4ukesbyu5RIc"
    "/export?format=xlsx"
)
DATA = Path(__file__).parent / "data"
XLSX = DATA / "willems2008_chains.xlsx"

# Willems (2008), Table 1: chain -> (company identifier, SIC code)
TABLE1 = {
    1: (1, 2869), 2: (2, 3674), 3: (3, 3577), 4: (4, 3944), 5: (5, 2099), 6: (6, 3421),
    7: (7, 3531), 8: (8, 3845), 9: (9, 2043), 10: (10, 5064), 11: (7, 3531), 12: (9, 2043),
    13: (3, 3674), 14: (11, 4731), 15: (12, 2841), 16: (8, 3845), 17: (13, 3577),
    18: (14, 3577), 19: (13, 3577), 20: (15, 3577), 21: (16, 2844), 22: (17, 2834),
    23: (18, 2851), 24: (19, 3546), 25: (20, 3523), 26: (21, 3724), 27: (22, 3845),
    28: (23, 3572), 29: (24, 3692), 30: (25, 4731), 31: (20, 3523), 32: (16, 2844),
    33: (12, 2844), 34: (26, 3661), 35: (22, 3845), 36: (27, 3523), 37: (28, 2869),
    38: (29, 3724),
}  # fmt: skip
STAGES_RANGE = (8, 2025)
ARCS_RANGE = (10, 16225)
# Figure 1 of the paper (chain 12) shows these stage names
FIG1_STAGES = ["Part_0017", "Part_0002", "Part_0003", "Part_0018", "Part_0004", "Manuf_0001",
               "Manuf_0002", "Manuf_0007", "Trans_0004", "Trans_0005", "Dist_0024",
               "Dist_0025", "Dist_0026", "Dist_0027", "Dist_0028", "Dist_0029"]


def load() -> dict[str, pd.DataFrame]:
    if not XLSX.exists():
        DATA.mkdir(exist_ok=True)
        print("downloading the public copy of the Willems (2008) data ...")
        urllib.request.urlretrieve(SHEET_URL, XLSX)
    return pd.read_excel(XLSX, sheet_name=None)


def check(x: dict[str, pd.DataFrame]) -> dict:
    c = x["chains"]
    s = x["stages in chains"]
    r = x["rels between stages"]
    mism = []
    for cid, (comp, sic) in TABLE1.items():
        row = c[c.chainId == cid]
        if row.empty:
            mism.append(f"chain {cid} missing")
            continue
        if int(row.companyId.iloc[0]) != comp or int(row.SIC_Code.iloc[0]) != sic:
            mism.append(f"chain {cid}: copy has company {int(row.companyId.iloc[0])}, "
                        f"SIC {int(row.SIC_Code.iloc[0])}; paper {comp}, {sic}")
    extra = sorted(set(c.chainId) - set(TABLE1))
    stages = s.drop_duplicates(["chainId", "Stage_Name"]).groupby("chainId").size()
    arcs = r.drop_duplicates().groupby("chainId").size()
    dup_rows = (s.groupby("chainId").size() - stages)
    fig1 = set(s[s.chainId == 12].Stage_Name)
    return {
        "chains_in_copy": int(len(c)),
        "table1_mismatches": mism,
        "extra_chains": [int(e) for e in extra],
        "stages_min_max": [int(stages.min()), int(stages.max())],
        "arcs_min_max": [int(arcs.min()), int(arcs.max())],
        "paper_stages_range": list(STAGES_RANGE),
        "paper_arcs_range": list(ARCS_RANGE),
        "ranges_match": [int(stages.min()), int(stages.max())] == list(STAGES_RANGE)
        and [int(arcs.min()), int(arcs.max())] == list(ARCS_RANGE),
        "chains_with_duplicate_stage_rows": {int(k): int(v) for k, v in dup_rows.items() if v},
        "chain12_fig1_stages_present": all(n in fig1 for n in FIG1_STAGES),
        "chain12_sic": str(c[c.chainId == 12].SIC_Description.iloc[0]),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(check(load()), indent=1))
