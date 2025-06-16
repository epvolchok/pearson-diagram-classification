import os
import subprocess
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

html_parts = []
style_block = ''
index_file = os.path.join(OUTPUT_HTML_DIR, 'index.html')
readme_file = os.path.join(OUTPUT_HTML_DIR, 'readme_html.html')

module_files.insert(0, readme_file)



style_block = """
<style>
  body {
    font-family: Arial, sans-serif;
    font-size: 20px;
    margin: 2em;
    background: #fdfdfd;
  }
  section {
    margin-bottom: 2em;
  }
  iframe {
    width: 100%;
    height: 1000px;
    border: 1px solid #aaa;
    border-radius: 6px;
    background: white;
  }
  h2 {
    margin-bottom: 0.5em;
    font-size: 1.5em;
  }
</style>
"""

script_resize = """
<script>
    function resizeIframe(iframe) {
      try {
        iframe.style.height = iframe.contentWindow.document.body.scrollHeight + 'px';
      } catch (e) {
        console.warn("⚠️ Не удалось получить высоту iframe:", e);
      }
    }
  </script>"""

html_head = f'''<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <title>Project Documentation</title>
    {style_block}
    {script_resize}
</head>
<body>
<h1>Project Documentation</h1>
'''
html_footer = "</body>\n</html>"

sections = []
for filename in module_files:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    section = f"""
<section>
<iframe src="{filename}" onload="resizeIframe(this)"></iframe>
</section>
        """
    sections.append(section)


final_html = html_head + "\n".join(sections) + html_footer

output_path = os.path.join(DOC_DIR, 'full_documentation.html')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(final_html)

