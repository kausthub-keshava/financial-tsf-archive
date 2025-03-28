"""
Functions to pull and calculate the value and equal weighted CRSP indices.

 - Data for indices: https://wrds-www.wharton.upenn.edu/data-dictionary/crsp_a_indexes/
 - Data for raw stock data: https://wrds-www.wharton.upenn.edu/pages/get-data/center-research-security-prices-crsp/annual-update/stock-security-files/monthly-stock-file/
 - Why we can't perfectly replicate them: https://wrds-www.wharton.upenn.edu/pages/support/support-articles/crsp/index-and-deciles/constructing-value-weighted-return-series-matches-vwretd-crsp-monthly-value-weighted-returns-includes-distributions/
 - Methodology used: https://wrds-www.wharton.upenn.edu/documents/396/CRSP_US_Stock_Indices_Data_Descriptions.pdf
 - Useful link: https://www.tidy-finance.org/python/wrds-crsp-and-compustat.html

Thank you to Tobias Rodriguez del Pozo for his assistance in writing this
code.

Note: This code is based on the old CRSP SIZ format. Information
about the new CIZ format can be found here:

 - Transition FAQ: https://wrds-www.wharton.upenn.edu/pages/support/manuals-and-overviews/crsp/stocks-and-indices/crsp-stock-and-indexes-version-2/crsp-ciz-faq/
 - CRSP Metadata Guide: https://wrds-www.wharton.upenn.edu/documents/1941/CRSP_METADATA_GUIDE_STOCK_INDEXES_FLAT_FILE_FORMAT_2_0_CIZ_09232022v.pdf

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import wrds

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
WRDS_USERNAME = config("WRDS_USERNAME")
START_DATE = pd.Timestamp("1925-01-01")
END_DATE = pd.Timestamp("2024-01-01")


def pull_CRSP_monthly_file(
    start_date=START_DATE,
    end_date=END_DATE,
    wrds_username=WRDS_USERNAME,
):
    """
    Pulls monthly CRSP stock data from a specified start date to end date.

    SQL query to pull data, controls for delisting, and importantly
    follows the guidelines that CRSP uses for inclusion, with the exception
    of code 73, which is foreign companies -- without including this, the universe
    of securities is roughly half of what it should be.
    """
    # Convert start_date to datetime if it's a string
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    query = f"""
    SELECT
        date,
        msf.permno, msf.permco, shrcd, exchcd, comnam, shrcls,
        ret, retx, dlret, dlretx, dlstcd,
        prc, altprc, vol, shrout, cfacshr, cfacpr,
        naics, siccd
    FROM crsp.msf AS msf
    LEFT JOIN
        crsp.msenames as msenames
    ON
        msf.permno = msenames.permno AND
        msenames.namedt <= msf.date AND
        msf.date <= msenames.nameendt
    LEFT JOIN
        crsp.msedelist as msedelist
    ON
        msf.permno = msedelist.permno AND
        date_trunc('month', msf.date)::date =
        date_trunc('month', msedelist.dlstdt)::date
    WHERE
        msf.date BETWEEN '{start_date}' AND '{end_date}' AND
        msenames.shrcd IN (10, 11, 20, 21, 40, 41, 70, 71, 73)
    """
    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     df = db.raw_sql(
    #         query, date_cols=["date", "namedt", "nameendt", "dlstdt"]
    #     )
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["date", "namedt", "nameendt", "dlstdt"])
    db.close()

    df = df.loc[:, ~df.columns.duplicated()]
    df["shrout"] = df["shrout"] * 1000

    # Also, as an additional note, CRSP reports that "cfacshr" and "cfacpr" are
    # not always equal. This means that we cannot use `market_cap` = `prc` *
    # `shrout` alone. We need to use the cumulative adjustment factors to adjust
    # for corporate actions that affect the stock price, such as stock splits.
    # "cfacshr" and "cfacpr" are not always equal because of less common
    # distribution events, spinoffs, and rights. See here: [CRSP - Useful
    # Variables](https://vimeo.com/443061703)

    df["adj_shrout"] = df["shrout"] * df["cfacshr"]
    df["adj_prc"] = df["prc"].abs() / df["cfacpr"]
    df["market_cap"] = df["adj_prc"] * df["adj_shrout"]

    # Deal with delisting returns
    df = apply_delisting_returns(df)

    return df


def apply_delisting_returns(df):
    """
    Use instructions for handling delisting returns from: Chapter 7 of
    Bali, Engle, Murray --
    Empirical asset pricing-the cross section of stock returns (2016)

    First change dlret column.
    If dlret is NA and dlstcd is not NA, then:
    if dlstcd is 500, 520, 551-574, 580, or 584, then dlret = -0.3
    if dlret is NA but dlstcd is not one of the above, then dlret = -1
    """
    df["dlret"] = np.select(
        [
            df["dlstcd"].isin([500, 520, 580, 584] + list(range(551, 575)))
            & df["dlret"].isna(),
            df["dlret"].isna() & df["dlstcd"].notna() & df["dlstcd"] >= 200,
            True,
        ],
        [-0.3, -1, df["dlret"]],
        default=df["dlret"],
    )

    df["dlretx"] = np.select(
        [
            df["dlstcd"].isin([500, 520, 580, 584] + list(range(551, 575)))
            & df["dlretx"].isna(),
            df["dlretx"].isna() & df["dlstcd"].notna() & df["dlstcd"] >= 200,
            True,
        ],
        [-0.3, -1, df["dlretx"]],
        default=df["dlretx"],
    )

    # Replace the inplace operations with direct assignments
    df["ret"] = df["ret"].fillna(df["dlret"])
    df["retx"] = df["retx"].fillna(df["dlretx"])
    return df


def apply_delisting_returns_alt(df):
    df["dlret"] = df["dlret"].fillna(0)
    df["ret"] = df["ret"] + df["dlret"]
    df["ret"] = np.where(
        (df["ret"].isna()) & (df["dlret"] != 0), df["dlret"], df["ret"]
    )
    return df


def pull_CRSP_index_files(
    start_date=START_DATE,
    end_date=END_DATE,
    wrds_username=WRDS_USERNAME,
):
    """
    Pulls the CRSP index files from crsp_a_indexes.msix:
    (Monthly)NYSE/AMEX/NASDAQ Capitalization Deciles, Annual Rebalanced (msix)
    """
    # Pull index files
    query = f"""
        SELECT *
        FROM crsp_a_indexes.msix
        WHERE caldt BETWEEN '{start_date}' AND '{end_date}'
    """
    # with wrds.Connection(wrds_username=wrds_username) as db:
    #     df = db.raw_sql(query, date_cols=["month", "caldt"])
    db = wrds.Connection(wrds_username=wrds_username)
    df = db.raw_sql(query, date_cols=["caldt"])
    db.close()
    return df


def load_CRSP_monthly_file(data_dir=DATA_DIR):
    path = Path(data_dir) / "CRSP_MSF_INDEX_INPUTS.parquet"
    df = pd.read_parquet(path)
    return df


def load_CRSP_index_files(data_dir=DATA_DIR):
    path = Path(data_dir) / "CRSP_MSIX.parquet"
    df = pd.read_parquet(path)
    return df


def _demo():
    df_msf = load_CRSP_monthly_file(data_dir=DATA_DIR)
    df_msf.info()
    df_msix = load_CRSP_index_files(data_dir=DATA_DIR)
    df_msix.info()


if __name__ == "__main__":
    # Create subfolder
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    df_msf = pull_CRSP_monthly_file(start_date=START_DATE, end_date=END_DATE)
    df_msf.to_parquet(data_dir / "CRSP_MSF_INDEX_INPUTS.parquet")

    df_msix = pull_CRSP_index_files(start_date=START_DATE, end_date=END_DATE)
    df_msix.to_parquet(data_dir / "CRSP_MSIX.parquet")
