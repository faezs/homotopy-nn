#!/bin/bash
find src/Neural -name "*.agda" -type f ! -name "*.bak*" ! -name "*_EDIT.agda" | sort | while read file; do
  holes=$(grep -c '{!!}' "$file" 2>/dev/null || echo 0)
  posts=$(grep -cE '^\s*postulate\s' "$file" 2>/dev/null || echo 0)
  if [ "$holes" -gt 0 ] || [ "$posts" -gt 0 ]; then
    echo "$holes,$posts,$file"
  fi
done
