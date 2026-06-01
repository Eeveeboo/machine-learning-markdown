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

echo ""
echo "==> Building Rust extension WASM..."
cd "$(dirname "$0")/.."
if command -v cargo &> /dev/null; then
  if rustup target list --installed 2>/dev/null | grep -q wasip2; then
    cargo build --target wasm32-wasip2 --release
    cp target/wasm32-wasip2/release/zed_mlmd.wasm extension.wasm 2>/dev/null || true
    echo "==> Rust extension WASM built: extension.wasm"
  else
    echo "⚠️  wasm32-wasip2 target not installed. Run: rustup target add wasm32-wasip2"
  fi
else
  echo "⚠️  cargo not found. Install Rust: https://rustup.rs"
fi

echo ""
echo "==> Done."
