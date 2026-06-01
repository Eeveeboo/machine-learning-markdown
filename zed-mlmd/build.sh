#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Checking for tree-sitter CLI..."
if ! command -v tree-sitter &> /dev/null; then
  if ! npx tree-sitter --version &> /dev/null 2>&1; then
    echo "tree-sitter CLI not found. Install with: npm install -g tree-sitter-cli"
    exit 1
  fi
  TS="npx tree-sitter"
else
  TS="tree-sitter"
fi

echo "==> Installing grammar dependencies..."
cd grammars/mlmd
npm install

echo "==> Generating parser..."
$TS generate

echo "==> Building WASM (requires emscripten or docker)..."
if $TS build --wasm 2>/dev/null; then
  echo "==> WASM built successfully."
else
  echo ""
  echo "⚠️  WASM build failed. This usually means emscripten is not installed."
  echo "   To build WASM, install emscripten: https://emscripten.org/docs/getting_started/downloads.html"
  echo "   OR use Docker: tree-sitter build --wasm (docker must be running)"
  echo ""
  echo "   The parser C source is still generated and can be used directly."
  echo "   For Zed dev extension, Zed will compile the grammar from source."
fi
