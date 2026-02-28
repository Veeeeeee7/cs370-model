#!/usr/bin/env python3
from __future__ import annotations

import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

# -------------------- CONFIG --------------------
BUCKET = "public-datasets-lila"
PREFIX = "nacti-unzipped"

CSV_PATH = "stratified_sample.csv"   # your CSV with a 'filename' column
FILENAME_COL = "filename"
LIMIT = None  # e.g. 10 for testing; set to None to download all rows

OUTDIR = Path("/scratch/vmli3/cs370/data/")
LOG_FILE = Path("log.txt")

MAX_WORKERS = 24
RETRIES = 6
TIMEOUT_SEC = 60
# ------------------------------------------------

_log_lock = threading.Lock()


def log_line(line: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {line}\n")


def normalize_rel_path(p: str) -> str:
    """
    Accepts:
      - part0/sub063/....jpg
      - /part0/sub063/....jpg
      - gs://public-datasets-lila/nacti-unzipped/part0/sub063/....jpg
      - https://storage.googleapis.com/public-datasets-lila/nacti-unzipped/part0/sub063/....jpg

    Returns:
      - part0/sub063/....jpg
    """
    p = str(p).strip()

    # Strip known full prefixes if present
    gs_prefix = f"gs://{BUCKET}/{PREFIX}/"
    https_prefix = f"https://storage.googleapis.com/{BUCKET}/{PREFIX}/"

    if p.startswith(gs_prefix):
        p = p[len(gs_prefix):]
    elif p.startswith(https_prefix):
        p = p[len(https_prefix):]

    return p.lstrip("/")


def download_one(rel: str) -> tuple[str, str]:
    """
    Downloads one relative object path to OUTDIR/rel.
    Returns (rel, status_msg).
    """
    rel = normalize_rel_path(rel)
    url = f"https://storage.googleapis.com/{BUCKET}/{PREFIX}/{rel}"

    out_path = OUTDIR / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded (non-empty)
    if out_path.exists() and out_path.stat().st_size > 0:
        return rel, f"SKIP exists size={out_path.stat().st_size}"

    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            req = Request(url, headers={"User-Agent": "python-downloader"})
            with urlopen(req, timeout=TIMEOUT_SEC) as r:
                data = r.read()

            # Write atomically: download to .part then rename
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            with open(tmp_path, "wb") as f:
                f.write(data)
            tmp_path.replace(out_path)

            return rel, f"OK bytes={len(data)}"
        except (HTTPError, URLError, TimeoutError) as e:
            last_err = f"{type(e).__name__}: {e}"
            # exponential backoff capped at 30s
            time.sleep(min(2 ** attempt, 30))

    return rel, f"FAIL {last_err}"


def load_files_from_csv() -> list[str]:
    df = pd.read_csv(CSV_PATH)

    if FILENAME_COL not in df.columns:
        raise ValueError(
            f"CSV missing column '{FILENAME_COL}'. Columns found: {list(df.columns)}"
        )

    files = (
        df[FILENAME_COL]
        .dropna()
        .astype(str)
        .map(normalize_rel_path)
        .tolist()
    )

    if LIMIT is not None:
        files = files[: int(LIMIT)]

    return files


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    files = load_files_from_csv()
    total = len(files)

    log_line(
        f"START csv={CSV_PATH} col={FILENAME_COL} total={total} "
        f"outdir={OUTDIR} workers={MAX_WORKERS} retries={RETRIES}"
    )

    ok = skip = fail = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(download_one, rel): rel for rel in files}

        for i, fut in enumerate(as_completed(futures), start=1):
            rel, status = fut.result()

            if status.startswith("OK"):
                ok += 1
            elif status.startswith("SKIP"):
                skip += 1
            else:
                fail += 1

            line = f"{status} | {rel} | ({i}/{total})"
            print(line)
            log_line(line)

    log_line(f"DONE ok={ok} skip={skip} fail={fail} total={total}")


if __name__ == "__main__":
    main()