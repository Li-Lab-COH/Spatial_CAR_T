#!/usr/bin/env python3
# minimal: pick a row and emit VAR='value' lines; headers are matched case-insensitively.
import sys, csv, shlex

# Usage: manifest_min.py <csv> <row_idx> VAR1 VAR2 ...
# Example: manifest_min.py samples.csv 7 TGEN_ID LOUPE_ALIGNMENT CYT_IMAGE IMAGE SLIDE AREA FASTQS

if len(sys.argv) < 4:
    sys.stderr.write(
        "Usage: manifest_min.py <csv> <row_idx> VAR1 VAR2 ...\n"
    )
    sys.exit(2)

csv_path = sys.argv[1]
row_idx  = int(sys.argv[2])          # trust input; raises ValueError if bad
vars_req = sys.argv[3:]              # at least one by the len(argv) check

with open(csv_path, newline="", encoding="utf-8-sig") as f:
    rdr = csv.DictReader(f)

    # map lowercase header -> original header for case-insensitive lookups
    hdr_map = {(h or "").strip().lower(): h for h in rdr.fieldnames}

    # trust row_idx; will raise IndexError if out of range
    rows = list(rdr)
    target = rows[row_idx]

    for VAR in vars_req:
        key = VAR.lower()
        orig = hdr_map.get(key)
        val = "" if orig is None else (target.get(orig) or "")
        if key == "fastqs":
            # normalize list-like cells
            s = val.replace(";", ",")
            parts = [p.strip() for p in s.split(",") if p.strip()]
            val = ",".join(parts)
        # print shell-safe assignment; the variable value itself contains no quotes
        print(f"{VAR}={shlex.quote(val)}")