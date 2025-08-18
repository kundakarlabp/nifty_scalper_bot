#!/usr/bin/env bash
# update_script.sh
# Usage: ./update_script.sh <file> <search_regex> <replace_text>
# Example: ./update_script.sh Dockerfile 'CMD \[.*\]' 'CMD ["python3","-m","src.main","start"]'
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <file> <search_regex> <replace_text>"
  exit 1
fi

FILE="$1"; SEARCH="$2"; REPLACE="$3"

if [[ ! -f "$FILE" ]]; then
  echo "❌ File not found: $FILE"
  exit 1
fi

# Create a backup, then replace in-place (GNU sed in Codespaces)
cp "$FILE" "$FILE.bak.$(date +%Y%m%d-%H%M%S)"
sed -i "s|$SEARCH|$REPLACE|g" "$FILE"

echo "✅ Updated: $FILE"
git add "$FILE"
