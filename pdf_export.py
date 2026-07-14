import tempfile
import os
from fpdf import FPDF
import pandas as pd

from auto_analysis import (
    build_overview_stats,
    build_numeric_summary_table,
    build_correlation_heatmap,
    build_top_categories,
    build_numeric_distribution,
)


class AnalysisPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(30, 30, 30)
        self.cell(0, 14, "AI Data Analyst Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(100, 130, 220)
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_line_width(0.2)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _split_insight(insight: str):
    """Split insight into a short direct answer and a longer description."""
    if not insight:
        return "", ""
    # Split on first sentence boundary
    for sep in (". ", ".\n", "\n\n"):
        idx = insight.find(sep)
        if idx != -1 and idx < 200:
            return insight[: idx + 1].strip(), insight[idx + len(sep):].strip()
    # Fallback: whole text as answer, no extra description
    return insight.strip(), ""


def _section_title(pdf: FPDF, text: str):
    if pdf.get_y() > 250:
        pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(50, 90, 180)
    pdf.cell(0, 9, f"  {text}", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)


def _add_chart_image(pdf: FPDF, fig, tmp_files: list, width: float = 188):
    img_bytes = fig.to_image(format="png", width=1000, height=520, scale=1.5)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(img_bytes)
    tmp.close()
    tmp_files.append(tmp.name)

    if pdf.get_y() > 210:
        pdf.add_page()
    pdf.image(tmp.name, x=10, w=width)
    pdf.ln(4)


def _draw_table(pdf: FPDF, df: pd.DataFrame, max_rows: int = 10):
    """Draw a bordered, styled table for a DataFrame."""
    df = df.head(max_rows)
    cols = list(df.columns)
    num_cols = len(cols)
    page_width = pdf.w - pdf.l_margin - pdf.r_margin  # usable width

    # Compute column widths proportionally (min 20, max 45)
    # Cap at 45 so the table stays compact rather than stretching full-page width
    col_w = min(45.0, max(20.0, page_width / num_cols))
    # If total width exceeds page, shrink equally
    total = col_w * num_cols
    if total > page_width:
        col_w = page_width / num_cols
    row_h = 8

    # Header row
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(50, 90, 180)
    pdf.set_text_color(255, 255, 255)
    pdf.set_draw_color(30, 60, 150)
    for col in cols:
        label = str(col)[:18]
        pdf.cell(col_w, row_h, label, border=1, fill=True, align="C")
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 9)
    pdf.set_draw_color(180, 180, 210)
    for r_idx, (_, row) in enumerate(df.iterrows()):
        if r_idx % 2 == 0:
            pdf.set_fill_color(245, 247, 255)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(30, 30, 30)
        for col in cols:
            val = str(row[col])
            if len(val) > 20:
                val = val[:18] + ".."
            pdf.cell(col_w, row_h, val, border=1, fill=True, align="C")
        pdf.ln()

    pdf.ln(3)


def generate_pdf(history: list, dataset_name: str = "Dataset") -> bytes:
    pdf = AnalysisPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Dataset info
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 6, f"Source: {dataset_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    tmp_files = []

    for i, entry in enumerate(history, 1):
        # ── Question banner ──────────────────────────────────────────
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(255, 255, 255)
        pdf.set_fill_color(50, 90, 180)
        pdf.cell(0, 9, f"  Q{i}: {entry['query']}", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # ── Direct answer + description (structured, falls back to old
        # single-"insight" sessions by heuristically splitting the text) ──
        direct_answer = (entry.get("answer") or "").strip()
        description = (entry.get("description") or "").strip()
        if not direct_answer and not description:
            direct_answer, description = _split_insight(entry.get("insight") or "")

        if direct_answer:
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(20, 20, 20)
            pdf.multi_cell(0, 8, direct_answer, align="J")
            pdf.ln(2)

        # ── Description (2-3 lines) ──────────────────────────────────
        if description:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(70, 70, 70)
            pdf.multi_cell(0, 6, description, align="J")
            pdf.ln(3)

        # ── Table (DataFrame result) ─────────────────────────────────
        result = entry.get("result")
        if isinstance(result, pd.DataFrame) and not result.empty:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(50, 90, 180)
            pdf.cell(0, 7, "Comparison Table (Top 10)", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            _draw_table(pdf, result, max_rows=10)
        elif result is not None:
            result_str = str(result).strip()
            if result_str:
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(50, 90, 180)
                pdf.cell(0, 6, "Result:", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(0, 6, result_str)
                pdf.ln(2)

        # ── Chart ────────────────────────────────────────────────────
        fig = entry.get("fig")
        if fig is not None:
            try:
                img_bytes = fig.to_image(format="png", width=1000, height=520, scale=1.5)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_bytes)
                tmp.close()
                tmp_files.append(tmp.name)

                if pdf.get_y() > 210:
                    pdf.add_page()

                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(50, 90, 180)
                pdf.cell(0, 7, "Chart", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)
                pdf.image(tmp.name, x=10, w=188)
                pdf.ln(4)
            except Exception as e:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(180, 60, 60)
                pdf.cell(0, 6, f"[Chart could not be rendered: {e}]", new_x="LMARGIN", new_y="NEXT")

        # ── Divider ──────────────────────────────────────────────────
        pdf.set_draw_color(180, 190, 230)
        pdf.set_line_width(0.4)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.set_line_width(0.2)
        pdf.ln(6)

    pdf_bytes = bytes(pdf.output())

    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    return pdf_bytes


def generate_full_analysis_pdf(df: pd.DataFrame, dataset_name: str = "Dataset") -> bytes:
    """Generate a full, automatic dataset analysis report (no LLM/chat needed):
    overview stats, numeric summary statistics, correlations, top-10 category
    breakdowns, and numeric distributions — all with nicely labeled charts."""
    pdf = AnalysisPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 6, f"Source: {dataset_name}  |  Full Dataset Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    tmp_files = []
    overview = build_overview_stats(df)
    numeric_cols = overview["numeric_columns"]
    categorical_cols = overview["categorical_columns"]

    def _chart_or_note(fig):
        try:
            _add_chart_image(pdf, fig, tmp_files)
        except Exception as e:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(180, 60, 60)
            pdf.cell(0, 6, f"[Chart could not be rendered: {e}]", new_x="LMARGIN", new_y="NEXT")

    # ── Overview ─────────────────────────────────────────────────────
    _section_title(pdf, "Dataset Overview")
    overview_df = pd.DataFrame({
        "Metric": ["Rows", "Columns", "Missing Values", "Duplicate Rows",
                   "Numeric Columns", "Categorical Columns"],
        "Value": [overview["rows"], overview["columns"], overview["missing_total"],
                  overview["duplicate_rows"], len(numeric_cols), len(categorical_cols)],
    })
    _draw_table(pdf, overview_df, max_rows=10)

    # ── Numeric summary statistics ───────────────────────────────────
    if numeric_cols:
        summary_df = build_numeric_summary_table(df, numeric_cols)
        if summary_df is not None:
            _section_title(pdf, "Numeric Summary Statistics")
            _draw_table(pdf, summary_df, max_rows=10)

    # ── Correlation heatmap ───────────────────────────────────────────
    if len(numeric_cols) > 1:
        fig = build_correlation_heatmap(df, numeric_cols)
        if fig is not None:
            _section_title(pdf, "Correlation Analysis")
            _chart_or_note(fig)

    # ── Top-10 category breakdowns (up to 5 categorical columns) ─────
    for col in categorical_cols[:5]:
        try:
            counts, fig = build_top_categories(df, col)
        except Exception:
            continue
        if counts is None or counts.empty:
            continue
        _section_title(pdf, f"Top 10 - {col.replace('_', ' ').title()}")
        _draw_table(pdf, counts, max_rows=10)
        _chart_or_note(fig)

    # ── Numeric distributions (up to 5 numeric columns) ───────────────
    for col in numeric_cols[:5]:
        fig = build_numeric_distribution(df, col)
        _section_title(pdf, f"Distribution - {col.replace('_', ' ').title()}")
        _chart_or_note(fig)

    pdf_bytes = bytes(pdf.output())

    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    return pdf_bytes
