"""
Automatic full-dataset analysis: dataset-wide stats, correlations, top
categories, distributions, and template-based narrative text — used to
generate the "Full Dataset Analysis" PDF the moment a file is uploaded
(independent of the chat Q&A flow). All functions here work with zero LLM
calls so the report always renders instantly; `agents/narrative_agent.py`
can optionally upgrade the prose when an LLM key is available.
"""

import pandas as pd
import numpy as np
import plotly.express as px

CHART_TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Bold


def _label(col: str) -> str:
    return str(col).replace("_", " ").replace("-", " ").title()


# ── Facts (pure computation, no LLM) ────────────────────────────────────────

def compute_full_facts(df: pd.DataFrame) -> dict:
    """Compute every data fact needed to write the report narrative, build
    tables, and pick charts — all derived directly from the data."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    missing = df.isnull().sum()
    missing_cols = {c: int(v) for c, v in missing.items() if v > 0}
    duplicate_rows = int(df.duplicated().sum())

    zero_heavy = {}
    for c in numeric_cols:
        zero_frac = float((df[c] == 0).mean())
        if zero_frac > 0.15:
            zero_heavy[c] = round(zero_frac * 100, 1)

    top_corr_pair = None
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        corr_values = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_values, 0)
        if corr_values.size and corr_values.max() > 0:
            idx = np.unravel_index(corr_values.argmax(), corr_values.shape)
            c1, c2 = corr.index[idx[0]], corr.columns[idx[1]]
            signed = df[[c1, c2]].corr().iloc[0, 1]
            top_corr_pair = (c1, c2, round(float(signed), 2))

    category_leaders = {}
    for c in categorical_cols[:6]:
        vc = df[c].astype(str).value_counts()
        if not vc.empty:
            category_leaders[c] = [(str(k), int(v)) for k, v in vc.head(2).items()]

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "missing_cols": missing_cols,
        "duplicate_rows": duplicate_rows,
        "zero_heavy": zero_heavy,
        "top_corr_pair": top_corr_pair,
        "category_leaders": category_leaders,
    }


# ── Tables ───────────────────────────────────────────────────────────────────

def build_descriptive_stats_table(df: pd.DataFrame, numeric_cols: list, max_cols: int = 12):
    """Variable / Mean / Median / Std Dev / Min / Max / 25th / 75th percentile."""
    cols = numeric_cols[:max_cols]
    if not cols:
        return None
    desc = df[cols].describe().T.round(2)
    out = pd.DataFrame({
        "Variable": desc.index,
        "Mean": desc["mean"],
        "Median": desc["50%"],
        "Std Dev": desc["std"],
        "Min": desc["min"],
        "Max": desc["max"],
        "25th Pct": desc["25%"],
        "75th Pct": desc["75%"],
    }).reset_index(drop=True)
    return out


# ── Charts ───────────────────────────────────────────────────────────────────

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
    """Top-N most frequent values for a categorical column, as (table, chart).

    The chart axis uses short position labels ("1", "2", ...) instead of the
    full category text — the actual values only appear in the accompanying
    table (and in the hover tooltip), so long category names never clutter
    the chart itself."""
    counts = df[col].astype(str).value_counts().head(top_n).reset_index()
    counts.columns = [col, "Count"]
    counts.insert(0, "#", [str(i + 1) for i in range(len(counts))])

    label = _label(col)
    fig = px.bar(
        counts, x="#", y="Count",
        color_discrete_sequence=[COLOR_SEQ[0]],
        title=f"Top {len(counts)} {label} by Count (see table for names)",
        labels={"#": label, "Count": "Number of Records"},
        hover_data={col: True, "#": False},
    )
    fig.update_xaxes(type="category")
    fig.update_layout(template=CHART_TEMPLATE, font=dict(size=13), showlegend=False)
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


def build_scatter_regression(df: pd.DataFrame, x_col: str, y_col: str):
    """Scatter plot with a fitted trend line for the two most correlated columns."""
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 3:
        return None, None

    x = sub[x_col].to_numpy(dtype=float)
    y = sub[y_col].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = round(1 - ss_res / ss_tot, 2) if ss_tot > 0 else 0.0

    label_x, label_y = _label(x_col), _label(y_col)
    fig = px.scatter(
        sub, x=x_col, y=y_col, opacity=0.5,
        color_discrete_sequence=[COLOR_SEQ[1]],
        title=f"{label_x} vs. {label_y} (R2 = {r2})",
        labels={x_col: label_x, y_col: label_y},
    )
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = slope * line_x + intercept
    fig.add_scatter(x=line_x, y=line_y, mode="lines", name="Trend line",
                     line=dict(color="crimson", width=3))
    fig.update_layout(template=CHART_TEMPLATE, font=dict(size=13))
    return fig, r2


def build_avg_by_category(df: pd.DataFrame, cat_col: str, num_col: str, top_n: int = 15):
    """Average of `num_col` per `cat_col`, as (table, chart). Like
    `build_top_categories`, the chart uses short position labels and the
    full category names live only in the returned table / hover tooltip."""
    grouped = (
        df.groupby(cat_col)[num_col]
        .mean()
        .round(2)
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    grouped.insert(0, "#", [str(i + 1) for i in range(len(grouped))])

    label_cat, label_num = _label(cat_col), _label(num_col)
    fig = px.bar(
        grouped, x="#", y=num_col,
        color_discrete_sequence=[COLOR_SEQ[3]],
        title=f"Average {label_num} by {label_cat} (see table for names)",
        labels={"#": label_cat, num_col: f"Average {label_num}"},
        hover_data={cat_col: True, "#": False},
    )
    fig.update_xaxes(type="category")
    fig.update_layout(template=CHART_TEMPLATE, font=dict(size=13), showlegend=False)
    return grouped, fig


def build_boxplot(df: pd.DataFrame, col: str):
    label = _label(col)
    fig = px.box(
        df, y=col, points="outliers",
        color_discrete_sequence=[COLOR_SEQ[2]],
        title=f"Boxplot of {label} (Outlier Detection)",
        labels={col: label},
    )
    fig.update_layout(template=CHART_TEMPLATE, font=dict(size=13))
    return fig


# ── Template narrative (no LLM required — always available as a fallback) ───

def write_overview_narrative(df: pd.DataFrame, facts: dict) -> str:
    n_rows, n_cols = facts["rows"], facts["columns"]
    numeric_n, cat_n = len(facts["numeric_columns"]), len(facts["categorical_columns"])
    preview_cols = df.columns[:8].tolist()
    cols_preview = ", ".join(str(c) for c in preview_cols)
    more = f", and {n_cols - 8} more" if n_cols > 8 else ""
    return (
        f"This dataset contains {n_rows:,} records across {n_cols} columns "
        f"({numeric_n} numeric, {cat_n} categorical/text). Key fields include {cols_preview}{more}. "
        "The analysis below profiles data quality, summarizes each numeric field, and explores "
        "relationships between variables to surface the most notable patterns."
    )


def write_quality_narrative(facts: dict) -> str:
    parts = []
    if not facts["missing_cols"]:
        parts.append("No missing values were found in any column — the dataset is fully complete.")
    else:
        items = ", ".join(f"{c} ({v:,} missing)" for c, v in list(facts["missing_cols"].items())[:8])
        parts.append(f"The following columns contain missing values: {items}.")
        parts.append("Consider whether these gaps are structural (e.g. optional fields) or "
                      "require imputation before deeper analysis.")
    if facts["duplicate_rows"]:
        parts.append(f"{facts['duplicate_rows']:,} duplicate rows were detected and should be "
                      "reviewed before drawing conclusions.")
    else:
        parts.append("No duplicate rows were detected.")
    if facts["zero_heavy"]:
        cols = ", ".join(f"{c} ({p}% zeros)" for c, p in facts["zero_heavy"].items())
        parts.append(f"Columns with an unusually high share of zero values ({cols}) may indicate "
                      "missing or unreported data rather than true zeros.")
    return " ".join(parts)


def write_key_findings(facts: dict) -> list:
    findings = []
    if facts["top_corr_pair"]:
        c1, c2, r = facts["top_corr_pair"]
        direction = "positive" if r > 0 else "negative"
        findings.append(f"{_label(c1)} and {_label(c2)} show the strongest {direction} "
                         f"correlation in the dataset (r = {r}).")
    for c, leaders in facts["category_leaders"].items():
        if leaders:
            top_val, top_count = leaders[0]
            findings.append(f"The most common {_label(c)} is \"{top_val}\" ({top_count:,} records).")
    if facts["duplicate_rows"]:
        findings.append(f"{facts['duplicate_rows']:,} duplicate rows warrant review.")
    if facts["missing_cols"]:
        worst_col = max(facts["missing_cols"], key=facts["missing_cols"].get)
        findings.append(f"{_label(worst_col)} has the most missing data "
                         f"({facts['missing_cols'][worst_col]:,} records) among all fields.")
    return findings[:6]


def write_summary_narrative(facts: dict) -> str:
    completeness = "largely complete" if not facts["missing_cols"] else \
        "mostly complete with a few gaps worth investigating"
    return (
        f"Overall, this {facts['rows']:,}-row dataset is {completeness}. The exploratory analysis "
        "above highlights the key relationships between numeric fields and the most frequent "
        "categories, providing a foundation for deeper investigation into specific questions."
    )
