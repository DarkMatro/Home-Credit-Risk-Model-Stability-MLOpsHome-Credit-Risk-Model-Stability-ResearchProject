from glob import glob

import polars as pl
from polars import Expr


def scan_file(path: str, **kwargs) -> pl.LazyFrame:
    """
    Scan into a LazyFrame from a parquet file.

    Parameters
    ----------
    path: str
        Path to a file

    Returns
    -------
    lf: pl.LazyFrame
    """
    lf = pl.scan_parquet(path, low_memory=True)
    lf = lf.pipe(cast_dtypes)
    if 'depth' in kwargs and kwargs['depth'] in [1, 2]:
        lf = lf.group_by("case_id").agg(get_exprs(lf, kwargs['date_col']))
    lf = lf.select(sorted(lf.columns))
    return lf


def scan_files(path: str, **kwargs) -> pl.LazyFrame:
    """
    Scan into a LazyFrame from multiple parquet files.

    Parameters
    ----------
    path: str
        Path to files

    Returns
    -------
    lf: pl.LazyFrame
    """
    chunks = [scan_file(x, **kwargs) for x in glob(path)]
    lf = pl.concat(chunks, how="vertical_relaxed")
    lf = lf.unique(subset=["case_id"])
    return lf


def get_exprs(lf: pl.LazyFrame, date_col: str | None) -> list[Expr]:
    """
    Scan into a LazyFrame from multiple parquet files.

    Parameters
    ----------
    lf: pl.LazyFrame

    date_col: str
        Feature with historical data using for aggregate.

    Returns
    -------
    lf: pl.LazyFrame
    """
    string_cols = set([
        col for dtype, col in zip(lf.dtypes, lf.columns) if dtype == pl.String
    ])
    other_cols = set(lf.columns) ^ string_cols ^ {'case_id'}
    exprs = [pl.max(col) for col in other_cols]
    if date_col is None:
        str_exprs = [pl.max(col) for col in string_cols]
        return exprs + str_exprs
    str_exprs = [
        pl.col(col).sort_by(date_col).drop_nulls().drop_nans().last()
        for col in string_cols
    ]
    return exprs + str_exprs


def cast_dtypes(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Cast LazyFrame columns to the specified dtypes:
    D - Transform date
    M - Masking categories
    A - Transform amount
    P - Transform DPD (Days past due)

    T - Unspecified Transform
    L - Unspecified Transform

    Parameters
    ----------
    lf: pl.LazyFrame

    Returns
    -------
    lf: pl.LazyFrame
    """
    for col in lf.columns:
        if col[-1] == 'D':
            lf = lf.with_columns(pl.col(col).cast(pl.Date))
        elif col[-1] == 'M':
            lf = lf.with_columns(pl.col(col).cast(pl.String))
        elif col[-1] == 'A':
            lf = lf.with_columns(pl.col(col).cast(pl.Float64))
        elif col[-1] == 'P':
            lf = lf.with_columns(pl.col(col).cast(pl.Int32))
        elif col in ['case_id', 'num_group1', 'num_group2']:
            lf = lf.with_columns(pl.col(col).cast(pl.Float64))
        elif col == 'date_decision':
            lf = lf.with_columns(pl.col(col).cast(pl.Date))
    return lf


def handle_dates(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Process date types columns.
    Birthdate aggregates by min and counts as years.

    Parameters
    ----------
    lf: pl.LazyFrame

    Returns
    -------
    lf: pl.LazyFrame
    """
    birth_cols = [col for col in lf.columns if 'birth' in col]
    lf = lf.with_columns(pl.min_horizontal(birth_cols).alias('birth_date')).drop(birth_cols)
    cols = [col for col, dt in zip(lf.columns, lf.dtypes) if dt == pl.Date]
    lf = lf.with_columns(pl.col(cols) - pl.col("date_decision"))
    lf = lf.with_columns(pl.col(cols).dt.total_days())
    lf = lf.with_columns((pl.col('birth_date') / -365.).cast(pl.Int32).alias('age_years'))
    lf = lf.drop("date_decision", 'birth_date')
    return lf
