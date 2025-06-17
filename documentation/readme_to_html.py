import os
import shutil
import markdown
import re

DOC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOC_DIR, '..'))
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

OUTPUT_HTML_DIR = os.path.join(DOC_DIR, 'html')
os.makedirs(OUTPUT_HTML_DIR, exist_ok=True)

FIGURES_DST_DIR = os.path.join(OUTPUT_HTML_DIR, 'figures')
os.makedirs(FIGURES_DST_DIR, exist_ok=True)

figures_list = ['solo_L2_rpw-tds-surv-rswf-e_20200619_V05.png', 'processing.png', 'clusterization.png', 'histograms.png', 'example.png']

for fig in figures_list:
    oldpath = os.path.join(FIGURES_DIR, fig)
    newpath = os.path.join(FIGURES_DST_DIR, fig)
    shutil.copy2(oldpath, newpath)

MARKDOWN_FILE = os.path.join(PROJECT_ROOT, 'README.md')
OUTPUT_HTML_FILE = os.path.join(OUTPUT_HTML_DIR, 'readme_html.html')

path = os.path.join(OUTPUT_HTML_DIR, 'mclustering/libfeatures.html')
style_block = ""
with open(path, "r", encoding="utf-8") as f:
    html = f.read()
    style_match = re.search(r"<style.*?</style>", html, re.DOTALL)
    if style_match:
        style_block = style_match.group(0)
print(style_block)

with open(MARKDOWN_FILE, "r", encoding="utf-8") as f:
    text = f.read()

html_body = markdown.markdown(text, extensions=["fenced_code", "codehilite", "tables"])

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Readme_html</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 10em; }}
        img {{ max-width: 50%; }}
        pre code {{ background-color: #f5f5f5; padding: 0.5em; display: block; overflow-x: auto; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""

with open(OUTPUT_HTML_FILE, "w", encoding="utf-8") as f:
    f.write(full_html)
