import tempfile
import os
from fpdf import FPDF
import pandas as pd

from auto_analysis import (
    _label,
    _prepare_categorical,
    compute_full_facts,
    build_descriptive_stats_table,
    build_correlation_heatmap,
    build_top_categories,
    build_numeric_distribution,
    build_scatter_regression,
    build_avg_by_category,
    build_boxplot,
    write_overview_narrative,
    write_quality_narrative,
    write_key_findings,
    write_summary_narrative,
)
from agents.narrative_agent import generate_report_narrative


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


def _draw_bullets(pdf: FPDF, items: list):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(30, 30, 30)
    for item in items:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, f"-  {item}")
    pdf.ln(2)


def _paragraph(pdf: FPDF, text: str):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(0, 6, text, align="J")
    pdf.ln(3)


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


def generate_full_analysis_pdf(df: pd.DataFrame, dataset_name: str = "Dataset", llm=None) -> bytes:
    """Generate a full, narrative dataset analysis report in the style of a
    professional data-analysis writeup: dataset overview, data-quality
    assessment, descriptive statistics, distributions, correlation analysis
    (with a scatter + trend line for the strongest pair), category breakdowns,
    an average-by-category comparison, outlier boxplots, key findings, and a
    closing summary. Works instantly with template-written prose; if `llm` is
    provided, the narrative sections are upgraded with AI-written text."""
    pdf = AnalysisPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(130, 130, 130)
    pdf.cell(0, 6, f"Source: {dataset_name}  |  Full Dataset Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    tmp_files = []
    facts = compute_full_facts(df)
    numeric_cols = facts["numeric_columns"]
    categorical_cols = facts["categorical_columns"]

    # Try an LLM-written narrative; always fall back to the template text
    # below if no LLM is available or the call fails for any reason.
    narrative = None
    if llm is not None:
        try:
            narrative = generate_report_narrative(facts, llm)
        except Exception:
            narrative = None

    overview_text = (narrative or {}).get("overview") or write_overview_narrative(df, facts)
    quality_text = (narrative or {}).get("quality") or write_quality_narrative(facts)
    key_findings = (narrative or {}).get("key_findings") or write_key_findings(facts)
    summary_text = (narrative or {}).get("summary") or write_summary_narrative(facts)

    def _chart_or_note(fig):
        try:
            _add_chart_image(pdf, fig, tmp_files)
        except Exception as e:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(180, 60, 60)
            pdf.cell(0, 6, f"[Chart could not be rendered: {e}]", new_x="LMARGIN", new_y="NEXT")

    # ── Dataset Overview ───────────────────────────────────────────────
    _section_title(pdf, "Dataset Overview")
    _paragraph(pdf, overview_text)

    # ── Quality Assessment ───────────────────────────────────────────────
    _section_title(pdf, "Quality Assessment")
    _paragraph(pdf, quality_text)

    # ── Descriptive Statistics ───────────────────────────────────────────
    if numeric_cols:
        stats_df = build_descriptive_stats_table(df, numeric_cols)
        if stats_df is not None:
            _section_title(pdf, "Descriptive Statistics")
            _draw_table(pdf, stats_df, max_rows=12)

    # ── Distributions ───────────────────────────────────────────────────
    if numeric_cols:
        _section_title(pdf, "Distributions")
        for col in numeric_cols[:8]:
            fig = build_numeric_distribution(df, col)
            _chart_or_note(fig)

    # ── Correlation Analysis ─────────────────────────────────────────────
    if len(numeric_cols) > 1:
        _section_title(pdf, "Correlation Analysis")
        heat_fig = build_correlation_heatmap(df, numeric_cols)
        if heat_fig is not None:
            _chart_or_note(heat_fig)

        if facts["top_corr_pair"]:
            c1, c2, r = facts["top_corr_pair"]
            scatter_fig, r2 = build_scatter_regression(df, c1, c2)
            if scatter_fig is not None:
                _paragraph(
                    pdf,
                    f"{_label(c1)} and {_label(c2)} show the strongest relationship in the "
                    f"dataset (correlation r = {r}). The scatter plot below fits a trend line "
                    f"(R2 = {r2}).",
                )
                _chart_or_note(scatter_fig)

    # ── Category Breakdown (up to 4 categorical columns) ─────────────────
    for col in categorical_cols[:4]:
        try:
            counts, fig, was_exploded = build_top_categories(df, col)
        except Exception:
            continue
        if counts is None or counts.empty:
            continue
        _section_title(pdf, f"Top 10 - {_label(col)}")
        leaders = facts["category_leaders"].get(col)
        if leaders:
            lead_text = f'The most common {_label(col)} is "{leaders[0][0]}" ({leaders[0][1]:,} records)'
            if len(leaders) > 1:
                lead_text += f', followed by "{leaders[1][0]}" ({leaders[1][1]:,}).'
            else:
                lead_text += "."
            if was_exploded:
                lead_text += (
                    f" This column stores multiple values per record (e.g. several tags in "
                    f"one field), so each record is counted once under every individual "
                    f"{_label(col)} it lists, rather than once under the combined text."
                )
            _paragraph(pdf, lead_text)
        _draw_table(pdf, counts, max_rows=10)
        _chart_or_note(fig)

    # ── Average <numeric> by <category> ──────────────────────────────────
    if numeric_cols and categorical_cols:
        num_col, cat_col = numeric_cols[0], categorical_cols[0]
        prepared_cat, _ = _prepare_categorical(df, cat_col)
        if prepared_cat[cat_col].nunique() <= 40:
            _section_title(pdf, f"Average {_label(num_col)} by {_label(cat_col)}")
            avg_table, avg_fig, _ = build_avg_by_category(df, cat_col, num_col)
            _draw_table(pdf, avg_table, max_rows=15)
            _chart_or_note(avg_fig)

    # ── Outlier Detection (boxplots, up to 4 numeric columns) ────────────
    if numeric_cols:
        _section_title(pdf, "Outlier Detection")
        for col in numeric_cols[:4]:
            _chart_or_note(build_boxplot(df, col))

    # ── Key Findings ───────────────────────────────────────────────────
    if key_findings:
        _section_title(pdf, "Key Findings")
        _draw_bullets(pdf, key_findings)

    # ── Summary ──────────────────────────────────────────────────────────
    _section_title(pdf, "Summary")
    _paragraph(pdf, summary_text)

    pdf_bytes = bytes(pdf.output())

    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    return pdf_bytes
