import os
import re


DOC_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_HTML_DIR = os.path.join(DOC_DIR, 'html')
PROJECT_ROOT = os.path.abspath(os.path.join(DOC_DIR, '..'))

MLUSTERING_DIR = os.path.join(OUTPUT_HTML_DIR, 'mclustering')
mlustering_files_order = [
    'libfeatures.html',
    'libpreprocessing.html',
    'libclustering.html',
    'libservice.html'
]
MANALYSE_DIR = os.path.join(OUTPUT_HTML_DIR, 'manalyse')
manalyse_files_order = [
    'plot_funcs.html'
]

MINTERACTIVE_DIR = os.path.join(OUTPUT_HTML_DIR, 'minteractive')
minteractive_files_order = [
    'libinteractive.html',
    'libprocesspipeline.html'
]

def collect_modules(modules, files_order, DIR):
    for fname in files_order:
        full_path = os.path.join(DIR, fname)
        if os.path.exists(full_path):
            modules.append(full_path)
        else:
            print(f'Warning: {fname} not found in {DIR}')
            print(full_path)

module_files = []
for order_list, module_dir in zip([mlustering_files_order, minteractive_files_order, manalyse_files_order], [MLUSTERING_DIR, MINTERACTIVE_DIR, MANALYSE_DIR]):
    collect_modules(module_files, order_list, module_dir)

readme_file = os.path.join(OUTPUT_HTML_DIR, 'readme_html.html')

# insert readme
def clean_embedded_html(content: str) -> str:
    """Удаляет <html>, <head>, <body>, <style> и <!DOCTYPE> из вложенного HTML."""
    content = re.sub(r'<style.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'<head.*?</head>', '', content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r'</?html[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'</?body[^>]*>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<!DOCTYPE [^>]+>', '', content, flags=re.IGNORECASE)
    return content.strip()

def update_image_paths(html_text):
    """
    Находит все <img src="figures/..."> и заменяет на <img src="html/figures/...">
    """
    return re.sub(
        r'(<img\s+[^>]*src=")(figures/[^"]+)"',
        r'\1html/\2"',
        html_text
    )

with open(readme_file, 'r', encoding='utf-8') as f:
        content = f.read()
readme_content = clean_embedded_html(content)
section_readme = f"""
<section>
{update_image_paths(readme_content)}
</section>
"""

# generate index
toc_lines = [
    "<h1>Index</h1>",
    "<ul>"
]

module_names = ['Clustering module: features extraction', 'Clustering module: preprocessing', 'Clustering module: clustering', \
                'Clustering module: service functions', 'Interactive module: service functions', \
                  'Interactive module: pipeline', 'Visualization']

for name, file in zip(module_names, module_files):
    link = rel_path = os.path.relpath(file, DOC_DIR).replace(os.sep, "/")
    toc_lines.append(f'  <li><a href="{link}">{name}</a></li>')

toc_lines.append("</ul>")

# settings
style_block = """
<style>
    body {
      font-family: sans-serif;
      font-size: 18px;
      margin: 2em;
      background: #fdfdfd;
    }
    section {
      margin-bottom: 2em;
    }
    a:link, a:visited {
    color: #0066cc;
    text-decoration: none;
  }

  @media print {
    a:link, a:visited {
      color: #000000;  /* или другой цвет, например тёмно-синий */
      text-decoration: none;
    }
  }
  </style>
"""


html_head = f'''<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <title>Project Documentation</title>
    {style_block}
</head>
<body>
<h1>Project Documentation</h1>
'''
html_footer = "</body>\n</html>"


final_html = html_head + section_readme + ''.join(toc_lines) + html_footer

output_path = os.path.join(DOC_DIR, 'documentation_full.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(final_html)

