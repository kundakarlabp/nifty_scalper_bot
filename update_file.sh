#!/usr/bin/env bash
# Usage: ./update_file.sh <file_path> "<search_text>" "<replace_text>"
# Example: ./update_file.sh Dockerfile 'CMD \[.*\]' 'CMD ["python3","-m","src.main","start"]'

set -euo pipefail
FILE="$1"; SEARCH="$2"; REPLACE="$3"

if [[ ! -f "$FILE" ]]; then
  echo "❌ File not found: $FILE"
  exit 1
fi

cp "$FILE" "$FILE.bak.$(date +%Y%m%d-%H%M%S)"   # make backup
sed -i "s|$SEARCH|$REPLACE|g" "$FILE"           # replace text
git add "$FILE"
git commit -m "auto: updated $FILE ($SEARCH → $REPLACE)" || true
echo "✅ Updated $FILE"
