#!/bin/bash

OUTDIR="docsources"
FINAL_HTML="documentation.html"
TEMP_README_HTML="$OUTDIR/temp_readme.html"

rm -f "$FINAL_HTML" "$TEMP_README_HTML"

if [ -f "README.md" ]; then
    echo "Convert README.md в HTML"
    pandoc README.md -o "$TEMP_README_HTML" --from markdown --to html
else
    echo "README.md not found"
fi

echo "Merge html files"

ORDER=(
  "temp_readme"
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