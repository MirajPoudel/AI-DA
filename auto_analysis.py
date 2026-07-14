"""
Automatic full-dataset analysis: dataset-wide stats, correlations, top
categories, and distributions — used to generate the "Full Dataset Analysis"
PDF the moment a file is uploaded (independent of the chat Q&A flow).
"""

import pandas as pd
import numpy as np
import plotly.express as px

CHART_TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Bold


def _label(col: str) -> str:
    return str(col).replace("_", " ").replace("-", " ").title()


def build_overview_stats(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }


def build_numeric_summary_table(df: pd.DataFrame, numeric_cols: list, max_cols: int = 10):
    """Descriptive statistics for up to `max_cols` numeric columns, one row per column."""
    cols = numeric_cols[:max_cols]
    if not cols:
        return None
    desc = df[cols].describe().T.round(2)
    desc.insert(0, "Column", desc.index)
    desc = desc.reset_index(drop=True)
    return desc


def build_correlation_heatmap(df: pd.DataFrame, numeric_cols: list):
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Between Numeric Fields",
        labels=dict(color="Correlation"),
    )
    fig.update_layout(template=CHART_TEMPLATE, font=dict(size=13))
    return fig


def build_top_categories(df: pd.DataFrame, col: str, top_n: int = 10):
    """Top-N most frequent values for a categorical column, as (table, chart)."""
    counts = df[col].astype(str).value_counts().head(top_n).reset_index()
    counts.columns = [col, "Count"]
    label = _label(col)
    fig = px.bar(
        counts, x=col, y="Count", color=col,
        color_discrete_sequence=COLOR_SEQ,
        title=f"Top {len(counts)} {label} by Count",
        labels={col: label, "Count": "Number of Records"},
    )
    fig.update_layout(template=CHART_TEMPLATE, font=dict(size=13), showlegend=False,
                       legend_title_text="")
    return counts, fig


def build_numeric_distribution(df: pd.DataFrame, col: str):
    label = _label(col)
    fig = px.histogram(
        df, x=col, nbins=30,
        color_discrete_sequence=[COLOR_SEQ[0]],
        title=f"Distribution of {label}",
        labels={col: label},
    )
    fig.update_layout(template=CHART_TEMPLATE, font=dict(size=13))
    return fig
