"""
NRB Macroeconomic PDF Downloader
Downloads all Current Macroeconomic and Financial Situation PDFs
from Nepal Rastra Bank website.

URL patterns observed across fiscal years:

FY 2023/24:
  - "...Two-Months-data-of-2023.pdf"       ← plural "Months", only start year, no .24
  - "...Three-Months-data-of-2023.24..."   ← plural, full FY label
  - "...Five-Months-data-of-2023.24..."    ← plural, full FY label
  - Annual: "...Annual-data-of-2023.24..."

FY 2025/26:
  - "...One-Month-data-of-2025.26..."      ← singular "Month"
  - "...Two-Month-data-of-2025.26..."      ← singular "Month"
  - "...Six-Month-data-of-2025.26..."      ← singular "Month"
  - Annual: "...Annual-data-of-2024.25..."

Strategy: try multiple filename variants per entry, use first 200-OK response.
"""

import time
import requests
from pathlib import Path

BASE_URL = "https://www.nrb.org.np/contents/uploads"

MONTH_WORDS = {
    1:  "One",
    2:  "Two",
    3:  "Three",
    4:  "Four",
    5:  "Five",
    6:  "Six",
    7:  "Seven",
    8:  "Eight",
    9:  "Nine",
    10: "Ten",
    11: "Eleven",
}

# FY month → (upload_calendar_month, year_offset_from_fy_start)
FY_MONTH_TO_UPLOAD = {
    1:  (9,  0),
    2:  (10, 0),
    3:  (11, 0),
    4:  (12, 0),
    5:  (1,  1),
    6:  (2,  1),
    7:  (3,  1),
    8:  (4,  1),
    9:  (5,  1),
    10: (6,  1),
    11: (7,  1),
    12: (8,  1),   # Annual
}


def fy_label_full(start_year: int) -> str:
    """e.g. 2025 → '2025.26'"""
    return f"{start_year}.{str(start_year + 1)[-2:]}"


def fy_label_short(start_year: int) -> str:
    """e.g. 2023 → '2023'  (some older URLs omit the end year)"""
    return str(start_year)


def build_candidate_urls(fy_start: int, fy_month: int) -> list[tuple[str, str]]:
    """
    Return list of (url, local_filename) candidates to try, most-likely first.
    We try all observed naming variants so at least one will hit.
    """
    upload_month, year_offset = FY_MONTH_TO_UPLOAD[fy_month]
    upload_year = fy_start + year_offset
    yy = f"{upload_year}/{upload_month:02d}"

    fy_full  = fy_label_full(fy_start)   # e.g. "2025.26"
    fy_short = fy_label_short(fy_start)  # e.g. "2025"

    candidates = []

    if fy_month == 12:
        # Annual report
        fname = (f"Current-Macroeconomic-and-Financial-Situation-English-"
                 f"Based-on-Annual-data-of-{fy_full}.pdf")
        local = f"NRB_Annual_{fy_full}.pdf"
        candidates.append((f"{BASE_URL}/{yy}/{fname}", local))

    else:
        word = MONTH_WORDS[fy_month]
        local = f"NRB_FY{fy_full}_Month{fy_month:02d}_{word}.pdf"

        # Variant A: singular "Month" + full FY label  (FY 2025/26 pattern)
        fA = (f"Current-Macroeconomic-and-Financial-Situation-English-"
              f"Based-on-{word}-Month-data-of-{fy_full}.pdf")
        # Variant B: plural "Months" + full FY label   (FY 2023/24 most entries)
        fB = (f"Current-Macroeconomic-and-Financial-Situation-English-"
              f"Based-on-{word}-Months-data-of-{fy_full}.pdf")
        # Variant C: plural "Months" + short year only (FY 2023/24 Two-Months edge case)
        fC = (f"Current-Macroeconomic-and-Financial-Situation-English-"
              f"Based-on-{word}-Months-data-of-{fy_short}.pdf")
        # Variant D: singular "Month" + short year only
        fD = (f"Current-Macroeconomic-and-Financial-Situation-English-"
              f"Based-on-{word}-Month-data-of-{fy_short}.pdf")

        for fname in [fA, fB, fC, fD]:
            candidates.append((f"{BASE_URL}/{yy}/{fname}", local))

    return candidates


def calendar_ym(fy_start: int, fy_month: int) -> tuple[int, int]:
    upload_month, year_offset = FY_MONTH_TO_UPLOAD[fy_month]
    return fy_start + year_offset, upload_month


def generate_entries(from_year: int, from_month: int,
                     to_year: int, to_month: int) -> list[dict]:
    entries = []
    for fy_start in range(2020, 2027):
        for fy_month in range(1, 13):
            cal_year, cal_month = calendar_ym(fy_start, fy_month)
            if (cal_year, cal_month) < (from_year, from_month):
                continue
            if (cal_year, cal_month) > (to_year, to_month):
                continue
            candidates = build_candidate_urls(fy_start, fy_month)
            entries.append({
                "candidates": candidates,
                "local_name": candidates[0][1],
                "cal_year": cal_year,
                "cal_month": cal_month,
                "fy": fy_label_full(fy_start),
                "fy_month": fy_month,
            })
    entries.sort(key=lambda e: (e["cal_year"], e["cal_month"]))
    return entries


def download_pdfs(output_dir: str = "nrb_pdfs",
                  from_ym: tuple = (2020, 8),
                  to_ym: tuple = (2026, 3),
                  delay: float = 1.5):
    """
    Download all NRB macroeconomic PDFs whose upload date is within the range.

    Args:
        output_dir: folder to save PDFs
        from_ym:    (year, month) start of upload date range inclusive
        to_ym:      (year, month) end of upload date range inclusive
        delay:      seconds between requests (be polite to server)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    entries = generate_entries(*from_ym, *to_ym)
    total = len(entries)
    print(f"Found {total} entries to attempt.\n")

    ok, skipped, failed = 0, 0, 0

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; NRB-PDF-Downloader/1.0)"
        )
    }

    for i, entry in enumerate(entries, 1):
        dest = out / entry["local_name"]
        label = (f"[{i:02d}/{total}] FY {entry['fy']} "
                 f"month {entry['fy_month']:02d} "
                 f"(upload {entry['cal_year']}/{entry['cal_month']:02d})")

        if dest.exists():
            print(f"  SKIP  {label} — already downloaded")
            skipped += 1
            continue

        print(f"  GET   {label}")

        hit = False
        for url, local_name in entry["candidates"]:
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                if resp.status_code == 200:
                    dest.write_bytes(resp.content)
                    size_kb = len(resp.content) / 1024
                    print(f"        ✓ {size_kb:.0f} KB → {dest.name}")
                    print(f"        URL: {url}")
                    ok += 1
                    hit = True
                    break
                # else try next variant silently
            except requests.RequestException as e:
                print(f"        ✗ Error on {url}: {e}")

        if not hit:
            print(f"        ✗ All variants returned non-200. URLs tried:")
            for url, _ in entry["candidates"]:
                print(f"          {url}")
            failed += 1

        if i < total:
            time.sleep(delay)

    print(f"\n{'='*60}")
    print(f"Done. Downloaded: {ok}  |  Skipped: {skipped}  |  Failed/Missing: {failed}")
    print(f"Files saved to: {out.resolve()}")


if __name__ == "__main__":
    download_pdfs(
        output_dir="nrb_pdfs",
        from_ym=(2020, 8),
        to_ym=(2026, 3),
        delay=1.5,
    )