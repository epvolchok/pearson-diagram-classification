#!/bin/bash

# is pdoc  installed?
if ! command -v pdoc &> /dev/null
then
    echo "pdoc is not installed!"
    exit 1
fi

# is pandoc installed?
if ! command -v pandoc &> /dev/null
then
    echo "pandoc is not installed!"
    exit 1
fi


OUTDIR="docsources"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR"/*.md

# all .py files
#PYTHON_FILES=$(find . -type f -name "*.py")
PYTHON_FILES=("main.py" "libfeatures.py" "libpreprocessing.py" "libclustering.py" "libservice.py" "libinteractive.py")

# generate documentation
for file in $PYTHON_FILES; do
    filename=$(basename "$file" .py)
    echo "Documentation for $file"
    pdoc "$file" --docformat numpydoc --output-dir "$OUTDIR" --format markdown
done


FINAL_MD="documentation.md"
FINAL_PDF="documentation.pdf"
> "$FINAL_MD"

# To add README.md
if [ -f "README.md" ]; then
    echo "README.md is adding"
    cat README.md >> "$FINAL_MD"
    echo -e "\n\n---\n\n" >> "$FINAL_MD"
else
    echo "Can not find README.md"
fi


for doc_file in "$OUTDIR"/*.md; do
    cat "$doc_file" >> "$FINAL_MD"
    echo -e "\n\n---\n\n" >> "$FINAL_MD"
done

# PDF generation with pandoc
echo "🖨️  Convert into PDF..."
pandoc "$FINAL_MD" -o "$FINAL_PDF" --toc --pdf-engine=xelatex

echo "Documentation has been collected:"
echo "- Markdown: $FINAL_MD"
echo "- PDF:      $FINAL_PDF"
