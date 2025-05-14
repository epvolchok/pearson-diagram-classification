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
rm -f "$OUTDIR"/*.html

# all .py files
PYTHON_FILES=$(find . -type f -name "*.py")
#PYTHON_FILES=("main.py" "libfeatures.py" "libpreprocessing.py" "libclustering.py" "libservice.py" "libinteractive.py")
#PYTHON_FILES=("libfeatures.py" "libservice.py")

# generate documentation
for file in $PYTHON_FILES; do
    filename=$(basename "$file" .py)
    echo "Documentation for $file "
    pdoc "$file" --docformat numpy -o "$OUTDIR"
done

FINAL_HTML="documentation.html"
FINAL_PDF="documentation.pdf"
TEMP_README_HTML="$OUTDIR/temp_readme.html"

rm -f "$FINAL_HTML" "$FINAL_PDF" "$TEMP_README_HTML"

if [ -f "README.md" ]; then
    echo "Convert README.md в HTML"
    pandoc README.md -o "$TEMP_README_HTML" --from markdown --to html
else
    echo "README.md not found"
fi
echo "Merge html files"

ORDER=(
  "README"
  "libfeatures"
  "libpreprocessing"
  "libclustering"
  "libinteractive"
  "libservice"
)

{
  echo "<!DOCTYPE html>"
  echo "<html><head><meta charset='utf-8'><title>Documentation</title></head><body>"
} > "$FINAL_HTML"

for name in "${ORDER[@]}"; do
  filepath="$OUTDIR/$name.html"
  if [[ -f "$filepath" ]]; then
    echo "Adding $filepath"
    echo "<section id=\"$name\">" >> "$FINAL_HTML"
    cat "$filepath" >> "$FINAL_HTML"
    echo "</section><hr>" >> "$FINAL_HTML"
  else
    echo "$filepath not found"
  fi
done

echo "</body></html>" >> "$FINAL_HTML"


echo "HTML-documentation has been created: $FINAL_HTML"

# Convert HTML to PDF
#echo "Convert HTML to PDF"
#pandoc "$FINAL_HTML" -o "$FINAL_PDF" --pdf-engine=xelatex

#echo "PDF-documentation has been created: $FINAL_PDF"


#rm -f "$TEMP_README_HTML"
