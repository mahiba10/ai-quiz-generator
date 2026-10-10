"""Convert the Markdown docs to print-ready HTML. Open the HTML in a browser, then Print -> Save as PDF.
Usage: pip install markdown   then   python docs/build_pdf_ready_html.py"""
from pathlib import Path
import markdown

CSS = """body{font-family:Georgia,serif;max-width:800px;margin:2rem auto;line-height:1.55;color:#222;padding:0 1rem}
h1{border-bottom:2px solid #4F46E5;padding-bottom:.3rem}h2{color:#4F46E5;margin-top:1.6rem}
table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:6px 10px;text-align:left}
code{background:#f3f4f8;padding:1px 4px}@media print{body{margin:0;max-width:none}h2{page-break-after:avoid}}"""
docs = Path(__file__).resolve().parent
for name in ("concept_note", "lean_canvas"):
    body = markdown.markdown((docs / f"{name}.md").read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    (docs / f"{name}_print.html").write_text(f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>", encoding="utf-8")
    print("Created", f"{name}_print.html")
