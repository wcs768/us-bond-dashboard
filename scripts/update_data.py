import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "dashboard.json"

HEADERS = {
    "User-Agent": "US-Bond-Yields-Dashboard/1.0"
}

FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=DGS2,DGS5,DGS10,DGS30,DFII5,DFII10,DFII30"
)

KW_PAGE_URL = (
    "https://www.federalreserve.gov/data/"
    "three-factor-nominal-term-structure-model.htm"
)

ACM_PAGE_URL = (
    "https://www.newyorkfed.org/research/data_indicators/"
    "term-premia-tabs"
)


def get(url, **kwargs):
    response = requests.get(url, headers=HEADERS, timeout=60, **kwargs)
    response.raise_for_status()
    return response


def as_number(value):
    try:
        if pd.isna(value):
            return None
        value = str(value).strip().replace(",", "")
        if value in ("", ".", "NA", "N/A", "nan"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_date(value):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.strftime("%Y-%m-%d")


def records_from_frame(frame, mapping):
    """
    mapping example:
    {
      "date": "observation_date",
      "y2": "DGS2",
      "y5": "DGS5",
      "y10": "DGS10",
      "y30": "DGS30"
    }
    """
    output = []

    for _, row in frame.iterrows():
        date = clean_date(row.get(mapping["date"]))

        if not date:
            continue

        item = {"date": date}

        for output_name, source_column in mapping.items():
            if output_name == "date":
                continue
            item[output_name] = as_number(row.get(source_column))

        output.append(item)

    return sorted(output, key=lambda x: x["date"])


def fetch_fred_data():
    print("Downloading nominal Treasury and TIPS data from FRED...")

    frame = pd.read_csv(io.StringIO(get(FRED_URL).text))
    frame.columns = [str(column).strip() for column in frame.columns]

    ust = records_from_frame(
        frame,
        {
            "date": "observation_date",
            "y2": "DGS2",
            "y5": "DGS5",
            "y10": "DGS10",
            "y30": "DGS30",
        },
    )

    tips = records_from_frame(
        frame,
        {
            "date": "observation_date",
            "y5": "DFII5",
            "y10": "DFII10",
            "y30": "DFII30",
        },
    )

    return ust, tips


def find_download_link(page_url, preferred_extensions):
    """
    Finds the first official CSV/XLSX download link on a source page.

    If the publisher changes its page design, define KW_DOWNLOAD_URL or
    ACM_DOWNLOAD_URL as a GitHub Actions repository variable.
    """
    page = get(page_url).text
    soup = BeautifulSoup(page, "html.parser")

    candidates = []

    for anchor in soup.select("a[href]"):
        href = urljoin(page_url, anchor.get("href"))
        text = anchor.get_text(" ", strip=True).lower()
        url_text = href.lower()

        if any(url_text.split("?")[0].endswith(ext) for ext in preferred_extensions):
            score = 0

            if "csv" in text:
                score += 20
            if "download" in text or "data" in text:
                score += 10
            if "term" in text or "yield" in text:
                score += 5

            candidates.append((score, href))

    if not candidates:
        raise RuntimeError(
            f"No downloadable {preferred_extensions} file was found on {page_url}."
        )

    return sorted(candidates, reverse=True)[0][1]


def normalize_column_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def choose_model_sheet(excel_file):
    """
    ACM files usually have a worksheet with a name containing 'fitted'.
    """
    fitted_sheets = [
        sheet for sheet in excel_file.sheet_names
        if "fitted" in sheet.lower() and "yield" in sheet.lower()
    ]

    if fitted_sheets:
        return fitted_sheets[0]

    generic_fitted = [
        sheet for sheet in excel_file.sheet_names
        if "fitted" in sheet.lower()
    ]

    return generic_fitted[0] if generic_fitted else excel_file.sheet_names[0]


def read_model_file(content, source_url):
    """
    Opens either CSV or Excel files. For Excel, it selects the worksheet
    that most likely contains fitted yields.
    """
    url_no_query = source_url.split("?")[0].lower()

    if url_no_query.endswith(".xlsx") or url_no_query.endswith(".xls"):
        excel = pd.ExcelFile(io.BytesIO(content))
        sheet = choose_model_sheet(excel)

        # Most official model files use a normal first-row header.
        frame = pd.read_excel(excel, sheet_name=sheet)
    else:
        text = content.decode("utf-8-sig", errors="replace")
        frame = pd.read_csv(io.StringIO(text))

    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def find_date_column(columns):
    normalised = {column: normalize_column_name(column) for column in columns}

    exact_candidates = ["date", "observationdate", "observation_date"]

    for candidate in exact_candidates:
        for original, cleaned in normalised.items():
            if cleaned == candidate:
                return original

    for original, cleaned in normalised.items():
        if "date" in cleaned:
            return original

    raise RuntimeError(f"No date column found. Available columns: {list(columns)}")


def find_tenor_column(columns, tenor):
    """
    Looks for fitted-yield fields for 2Y, 5Y, or 10Y.

    This function deliberately rejects term-premium fields. It prioritises:
    fittedyield2year, fittedyield2, yield2year, yield2, y2, etc.
    """
    scored = []

    for column in columns:
        cleaned = normalize_column_name(column)

        if "termpremium" in cleaned or cleaned.startswith("tp"):
            continue

        tenor_patterns = [
            f"fittedyield{tenor}year",
            f"fittedyield{tenor}",
            f"yield{tenor}year",
            f"yield{tenor}",
            f"y{tenor}",
            f"{tenor}year",
        ]

        for rank, pattern in enumerate(tenor_patterns):
            if cleaned == pattern:
                score = 100 - rank
                if "fitted" in cleaned:
                    score += 30
                scored.append((score, column))
                break

    if not scored:
        raise RuntimeError(
            f"Could not identify the {tenor}Y fitted-yield column. "
            f"Available columns: {list(columns)}"
        )

    return sorted(scored, reverse=True)[0][1]


def fetch_model_data(model_name, landing_page_url, env_url_name):
    """
    Downloads official Kim–Wright or ACM data, then extracts fitted
    zero-coupon 2Y, 5Y and 10Y yields.
    """
    configured_url = os.getenv(env_url_name, "").strip()

    if configured_url:
        download_url = configured_url
        print(f"Using configured {model_name} download URL.")
    else:
        print(f"Locating {model_name} download file...")
        download_url = find_download_link(
            landing_page_url,
            preferred_extensions=[".csv", ".xlsx", ".xls"]
        )

    print(f"Downloading {model_name}: {download_url}")

    content = get(download_url).content
    frame = read_model_file(content, download_url)

    date_column = find_date_column(frame.columns)
    y2_column = find_tenor_column(frame.columns, 2)
    y5_column = find_tenor_column(frame.columns, 5)
    y10_column = find_tenor_column(frame.columns, 10)

    print(
        f"{model_name} columns selected: "
        f"date={date_column}, 2Y={y2_column}, 5Y={y5_column}, 10Y={y10_column}"
    )

    return records_from_frame(
        frame,
        {
            "date": date_column,
            "y2": y2_column,
            "y5": y5_column,
            "y10": y10_column,
        },
    )


def fetch_cds_data():
    """
    Generic licensed-vendor API connector.

    Required GitHub Secrets:
      CDS_API_URL
      CDS_API_TOKEN

    Expected vendor response format:
      {
        "data": [
          {
            "date": "2026-08-27",
            "cds_2y": 20.1,
            "cds_5y": 27.4,
            "cds_10y": 35.2
          }
        ]
      }

    If your vendor uses a different response structure, alter only this
    function. Do not put credentials in index.html.
    """
    api_url = os.getenv("CDS_API_URL", "").strip()
    api_token = os.getenv("CDS_API_TOKEN", "").strip()

    if not api_url or not api_token:
        print("CDS API credentials not configured. CDS series will be empty.")
        return []

    print("Downloading U.S. sovereign CDS data from vendor API...")

    response = requests.get(
        api_url,
        headers={
            **HEADERS,
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
        },
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()
    raw_rows = payload.get("data", payload)

    if not isinstance(raw_rows, list):
        raise RuntimeError("CDS API response must contain a list of records.")

    output = []

    for row in raw_rows:
        date = clean_date(row.get("date"))

        if not date:
            continue

        output.append(
            {
                "date": date,
                "y2": as_number(row.get("cds_2y")),
                "y5": as_number(row.get("cds_5y")),
                "y10": as_number(row.get("cds_10y")),
            }
        )

    return sorted(output, key=lambda x: x["date"])


def main():
    DATA_DIR.mkdir(exist_ok=True)

    ust, tips = fetch_fred_data()

    kw = fetch_model_data(
        model_name="Kim–Wright",
        landing_page_url=KW_PAGE_URL,
        env_url_name="KW_DOWNLOAD_URL",
    )

    acm = fetch_model_data(
        model_name="ACM",
        landing_page_url=ACM_PAGE_URL,
        env_url_name="ACM_DOWNLOAD_URL",
    )

    cds = fetch_cds_data()

    dashboard = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "ust_tips": "Federal Reserve H.15 via FRED",
            "kim_wright": KW_PAGE_URL,
            "acm": ACM_PAGE_URL,
            "cds": "Licensed vendor API",
        },
        "series": {
            "ust": ust,
            "tips": tips,
            "cds": cds,
            "kw": kw,
            "acm": acm,
        },
    }

    OUTPUT_FILE.write_text(
        json.dumps(dashboard, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"Dashboard data written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
