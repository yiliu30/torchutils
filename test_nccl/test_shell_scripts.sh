#!/usr/bin/env bash
set -euo pipefail

# Simple shell script test harness:
# - Syntax check with `bash -n`
# - Optional lint with `shellcheck` if available
# Exits non-zero on any failure.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")"/.. && pwd)}"
EXCLUDE_PATTERNS=(
  "*/.git/*"
  "*/build/*"
  "*/dist/*"
  "*/node_modules/*"
  "*/venv/*"
  "*/.venv/*"
)

matches_exclude() {
  local path="$1"
  for pat in "${EXCLUDE_PATTERNS[@]}"; do
    if [[ "$path" == $pat ]]; then
      return 0
    fi
  done
  return 1
}

fail_count=0
checked_count=0

mapfile -t files < <(find "$ROOT_DIR" -type f -name "*.sh")

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No shell scripts found under $ROOT_DIR"
  exit 0
fi

if command -v shellcheck >/dev/null 2>&1; then
  echo "shellcheck detected: running lint in addition to syntax checks"
else
  echo "shellcheck not found: running syntax checks only"
fi

for f in "${files[@]}"; do
  # Skip excluded patterns
  for pat in "${EXCLUDE_PATTERNS[@]}"; do
    if [[ "$f" == $pat ]]; then
      continue 2
    fi
  done

  ((checked_count++)) || true
  printf "[CHECK] %s\n" "$f"
  if ! bash -n "$f"; then
    echo "[ERROR] Syntax check failed: $f"
    ((fail_count++)) || true
    continue
  fi
  if command -v shellcheck >/dev/null 2>&1; then
    # Use -x to follow sourced files, set warning severity
    if ! shellcheck -x -S warning "$f"; then
      echo "[WARN] Lint issues reported by shellcheck: $f"
      # Do not count as hard failure; adjust policy here if desired
      :
    fi
  fi
done

echo "Checked: $checked_count script(s)."
if [[ "$fail_count" -gt 0 ]]; then
  echo "Failures: $fail_count syntax error(s)."
  exit 1
fi

echo "All shell scripts passed syntax checks."
