import io
import tempfile
import os
from fpdf import FPDF
import pandas as pd


class AnalysisPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 30, 30)
        self.cell(0, 12, "AI Data Analyst Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def generate_pdf(history: list, dataset_name: str = "Dataset") -> bytes:
    pdf = AnalysisPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Dataset info line
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, f"Source: {dataset_name}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    tmp_files = []

    for i, entry in enumerate(history, 1):
        # Section header: question
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(40, 80, 160)
        pdf.set_fill_color(235, 241, 255)
        pdf.cell(0, 8, f"Q{i}: {entry['query']}", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        # Insight text
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 7, entry.get("insight") or "")
        pdf.ln(2)

        # Result value (if scalar/short)
        result = entry.get("result")
        if result is not None:
            result_str = str(result)
            if len(result_str) < 300:
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(60, 60, 60)
                pdf.cell(0, 6, "Result:", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Courier", "", 10)
                pdf.set_text_color(20, 100, 20)
                pdf.multi_cell(0, 6, result_str)
                pdf.ln(2)
            elif isinstance(result, pd.DataFrame):
                # Render first few rows as a simple text table
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(60, 60, 60)
                pdf.cell(0, 6, "Result (first 10 rows):", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Courier", "", 8)
                pdf.set_text_color(20, 20, 20)
                pdf.multi_cell(0, 5, result.head(10).to_string(index=False))
                pdf.ln(2)

        # Chart image
        fig = entry.get("fig")
        if fig is not None:
            try:
                img_bytes = fig.to_image(format="png", width=900, height=500, scale=1.5)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_bytes)
                tmp.close()
                tmp_files.append(tmp.name)

                # Check remaining page space; add page if needed
                if pdf.get_y() > 220:
                    pdf.add_page()

                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(60, 60, 60)
                pdf.cell(0, 6, "Chart:", new_x="LMARGIN", new_y="NEXT")
                pdf.image(tmp.name, x=10, w=185)
                pdf.ln(4)
            except Exception as e:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(180, 60, 60)
                pdf.cell(0, 6, f"[Chart could not be rendered: {e}]", new_x="LMARGIN", new_y="NEXT")

        # Divider between entries
        pdf.set_draw_color(220, 220, 220)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

    pdf_bytes = bytes(pdf.output())

    for f in tmp_files:
        try:
            os.unlink(f)
        except Exception:
            pass

    return pdf_bytes
