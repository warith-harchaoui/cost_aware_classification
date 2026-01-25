#!/usr/bin/env bash
#
# ==============================================================================
# LaTeX Compilation Script (with BibTeX support) + optional PDF open
# ==============================================================================
# Usage:
#   ./compile_latex.sh <filename_without_extension> [-c|--clean] [-v|--verbose] [-O|--open]
#
# Examples:
#   ./compile_latex.sh kdd_submission
#   ./compile_latex.sh kdd_submission --clean
#   ./compile_latex.sh path/to/kdd_submission -c -v
#   ./compile_latex.sh kdd_submission -O
#
# Notes:
# - Run this script with bash, NOT sh:
#     ./compile_latex.sh kdd_submission
#     # or
#     bash ./compile_latex.sh kdd_submission
# ==============================================================================

set -euo pipefail

# ------------------------------- Helpers --------------------------------------

usage() {
  cat <<'EOF'
Usage: ./compile_latex.sh <filename_without_extension> [-c|--clean] [-v|--verbose] [-O|--open]

Options:
  -c, --clean     Remove auxiliary files after compilation
  -v, --verbose   Show pdflatex/bibtex output (default is quiet)
  -O, --open      Open the resulting PDF with the default viewer (local use)
  -h, --help      Show this help message
EOF
}

die() {
  echo "❌ $*" >&2
  exit 1
}

info() {
  echo "ℹ️  $*" >&2
}

# Run a command. In quiet mode, capture output and print it only on failure.
run_cmd() {
  if [[ "${VERBOSE}" == "true" ]]; then
    "$@"
    return
  fi

  local tmp
  tmp="$(mktemp)"
  if ! "$@" >"${tmp}" 2>&1; then
    cat "${tmp}" >&2
    rm -f "${tmp}"
    return 1
  fi
  rm -f "${tmp}"
}

# Pick a timeout command if available (macOS: gtimeout via coreutils).
pick_timeout_cmd() {
  if command -v gtimeout >/dev/null 2>&1; then
    echo "gtimeout"
  elif command -v timeout >/dev/null 2>&1; then
    echo "timeout"
  else
    echo ""
  fi
}

# Print helpful diagnostics from the .log file (tail + first error lines).
print_log_diagnostics() {
  local log_file="$1"
  [[ -f "${log_file}" ]] || return 0

  echo "---- Tail of ${log_file} ----" >&2
  tail -n 80 "${log_file}" >&2 || true

  if grep -n "^!" "${log_file}" >/dev/null 2>&1; then
    echo "---- Error lines (prefixed by '!') ----" >&2
    grep -n "^!" "${log_file}" | head -n 20 >&2 || true
  fi
}

# Run pdflatex with safe, non-interactive settings.
run_pdflatex() {
  local tex_file="$1"
  local timeout_cmd="$2"
  local timeout_seconds="$3"

  if [[ -n "${timeout_cmd}" ]]; then
    if ! run_cmd "${timeout_cmd}" "${timeout_seconds}" "${PDFLATEX[@]}" "${tex_file}"; then
      echo "❌ pdflatex failed or timed out after ${timeout_seconds}s." >&2
      print_log_diagnostics "${BASE}.log"
      exit 1
    fi
  else
    if ! run_cmd "${PDFLATEX[@]}" "${tex_file}"; then
      echo "❌ pdflatex failed." >&2
      print_log_diagnostics "${BASE}.log"
      exit 1
    fi
  fi

  # Strict mode: fail if TeX logged an error (lines start with "!")
  if [[ -f "${BASE}.log" ]] && grep -q "^!" "${BASE}.log"; then
    echo "❌ LaTeX errors detected in ${BASE}.log." >&2
    print_log_diagnostics "${BASE}.log"
    exit 1
  fi
}

# Cross-platform "open PDF" helper (macOS / Linux / WSL).
open_pdf() {
  local pdf="$1"

  if [[ ! -f "${pdf}" ]]; then
    info "PDF not found at: ${pdf}"
    return 0
  fi

  if command -v open >/dev/null 2>&1; then
    open "${pdf}"                       # macOS
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${pdf}" >/dev/null 2>&1 & # Linux
  elif command -v wslview >/dev/null 2>&1; then
    wslview "${pdf}"                    # WSL
  else
    info "PDF generated at ${pdf} (no opener found)"
  fi
}

# ------------------------------ Arg parsing -----------------------------------

CLEAN_MODE="false"
VERBOSE="false"
OPEN_MODE="false"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

TEX_BASENAME="$1"
shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--clean)   CLEAN_MODE="true" ;;
    -v|--verbose) VERBOSE="true" ;;
    -O|--open)    OPEN_MODE="true" ;;
    -h|--help)    usage; exit 0 ;;
    *)            die "Unknown argument: $1 (use --help)" ;;
  esac
  shift
done

# ------------------------------ Main logic ------------------------------------

TEX_PATH="${TEX_BASENAME}.tex"
[[ -f "${TEX_PATH}" ]] || die "File not found: ${TEX_PATH}"

WORKDIR="$(dirname "${TEX_PATH}")"
BASE="$(basename "${TEX_BASENAME}")"

pushd "${WORKDIR}" >/dev/null

info "🚀 Starting compilation for ${BASE}.tex (dir: ${WORKDIR})"

# Non-interactive mode that avoids "press Enter repeatedly" behavior.
# -file-line-error improves log readability.
PDFLATEX=(pdflatex -interaction=batchmode -file-line-error)

TIMEOUT_CMD="$(pick_timeout_cmd)"
TIMEOUT_SECONDS=90  # adjust if you have heavy TikZ / large figures

info "📝 Step 1/4: pdflatex (initial pass)"
run_pdflatex "${BASE}.tex" "${TIMEOUT_CMD}" "${TIMEOUT_SECONDS}"

# Only run BibTeX if bibliography directives are present in the aux.
if [[ -f "${BASE}.aux" ]] && grep -qE '^(\\citation|\\bibdata|\\bibstyle)' "${BASE}.aux"; then
  info "📚 Step 2/4: bibtex"
  if ! run_cmd bibtex "${BASE}"; then
    echo "❌ bibtex failed. See ${WORKDIR}/${BASE}.blg" >&2
    if [[ -f "${BASE}.blg" ]]; then
      echo "---- Tail of ${BASE}.blg ----" >&2
      tail -n 80 "${BASE}.blg" >&2 || true
    fi
    popd >/dev/null
    exit 2
  fi
else
  info "📚 Step 2/4: bibtex skipped (no bibliography directives found)"
fi

info "📝 Step 3/4: pdflatex (resolve citations)"
run_pdflatex "${BASE}.tex" "${TIMEOUT_CMD}" "${TIMEOUT_SECONDS}"

info "📝 Step 4/4: pdflatex (resolve cross-references)"
run_pdflatex "${BASE}.tex" "${TIMEOUT_CMD}" "${TIMEOUT_SECONDS}"

info "✅ Compilation successful! Created ${WORKDIR}/${BASE}.pdf"

if [[ "${OPEN_MODE}" == "true" ]]; then
  info "🖥️  Opening PDF..."
  open_pdf "${BASE}.pdf"
fi

# ------------------------------- Cleaning -------------------------------------

if [[ "${CLEAN_MODE}" == "true" ]]; then
  info "🧹 Cleaning auxiliary files..."
  rm -f \
    "${BASE}.aux" "${BASE}.log" "${BASE}.bbl" "${BASE}.blg" "${BASE}.out" \
    "${BASE}.toc" "${BASE}.nav" "${BASE}.snm" "${BASE}.vrb" \
    "${BASE}.fls" "${BASE}.fdb_latexmk" "${BASE}.synctex.gz" \
    "${BASE}.lof" "${BASE}.lot" "${BASE}.bcf" "${BASE}.run.xml"
  info "✨ Cleaned."
fi

popd >/dev/null
