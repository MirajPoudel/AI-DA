import fitz

doc = fitz.open("attached_assets/analysis_report_(2)_1783783502918.pdf")
print(f"Pages: {doc.page_count}")
for i, page in enumerate(doc):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    out = f".agents/outputs/page_{i+1}.png"
    pix.save(out)
    print(f"Saved {out} ({pix.width}x{pix.height})")
