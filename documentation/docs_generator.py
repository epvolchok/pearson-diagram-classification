import os
import subprocess

DOC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOC_DIR, '..'))

MLUSTERING_DIR = os.path.join(PROJECT_ROOT, 'mclustering')
mlustering_files_order = [
    'libfeatures.py',
    'libpreprocessing.py',
    'libclustering.py',
    'libservice.py'
]

MANALYSE_DIR = os.path.join(PROJECT_ROOT, 'manalyse')
manalyse_files_order = [
    'plot_funcs.py'
]

MINTERACTIVE_DIR = os.path.join(PROJECT_ROOT, 'minteractive')
minteractive_files_order = [
    'libinteractive.py',
    'libprocesspipeline.py'
]

modules = []

def collect_modules(modules, files_order, DIR):
    for fname in files_order:
        full_path = os.path.join(DIR, fname)
        if os.path.exists(full_path):
            modules.append(full_path)
        else:
            print(f'Warning: {fname} not found in {DIR}')
            print(full_path)

for order_list, module_dir in zip([mlustering_files_order, minteractive_files_order, manalyse_files_order], [MLUSTERING_DIR, MINTERACTIVE_DIR, MANALYSE_DIR]):
    collect_modules(modules, order_list, module_dir)


OUTPUT_HTML_DIR = os.path.join(DOC_DIR, 'html')
os.makedirs(OUTPUT_HTML_DIR, exist_ok=True)


command = [
    'pdoc',
    '--docformat', 'numpy',
    '--output-dir', OUTPUT_HTML_DIR
] + modules

print('Generating documentation with pdoc...')
subprocess.run(command, check=True)
print(f'Documentation saved in: {OUTPUT_HTML_DIR}')


print('Generating index.html...')


module_names = []
for path in modules:
    rel = os.path.relpath(path, PROJECT_ROOT).replace(os.sep, '.')
    if rel.endswith('.py'):
        rel = rel[:-3]
    if rel.endswith('__init__'):
        rel = rel.rsplit('.', 1)[0]
    module_names.append(rel)

index_lines = [
    '<!DOCTYPE html>',
    '<html lang="en">',
    '<head><meta charset="UTF-8"><title>Documentation Index</title></head>',
    '<body>',
    '<h1>Project Documentation</h1>',
    '<ul>'
]

for name in module_names:
    href = f'{name}.html'
    index_lines.append(f'<li><a href="{href}">{name}</a></li>')

index_lines += [
    '</ul>',
    '</body>',
    '</html>'
]


index_path = os.path.join(OUTPUT_HTML_DIR, 'index.html')
with open(index_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(index_lines))

print(f' index.html has been created: {index_path}')
