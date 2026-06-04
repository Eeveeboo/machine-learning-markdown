CARGO ?= cargo
MLMD_RELEASE ?= target/release/mlmd

.PHONY: help build build-release test test-python-examples test-rust-examples install install-zed \
        install-vscode lint fmt fmt-fix clean examples generate-examples check

help:  ## Show this help
	@awk -F '## ' '/^[a-zA-Z_-]+:.*##/ { printf "\033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

build:  ## Build the workspace (all crates)
	@echo "Building workspace..."
	$(CARGO) build --workspace

test: test-python-examples test-rust-examples  ## Run all tests (Rust + Python examples)
	@echo "Running tests..."
	$(CARGO) test --workspace

test-python-examples:  ## Run Python example tests
	cd examples && uv sync && uv run pytest -v

test-rust-examples:  ## Run Rust example tests
	$(CARGO) test -p mlmd-examples

build-release:  ## Build release binary
	$(CARGO) build --release -p mlmd

install:  ## Build release binary and install to ~/.cargo/bin
	cargo install --path mlmd

install-zed: build-release  ## Install mlmd Zed extension
	$(MLMD_RELEASE) install zed

install-vscode: build-release  ## Install mlmd VS Code extension
	$(MLMD_RELEASE) install vscode

lint:  ## Run clippy linting on the workspace
	$(CARGO) clippy --workspace -- -D warnings 2>/dev/null || $(CARGO) clippy --workspace

fmt:  ## Check Rust formatting
	$(CARGO) fmt --check

fmt-fix:  ## Fix Rust formatting
	$(CARGO) fmt

clean:  ## Clean all build artifacts
	@echo "Cleaning build artifacts..."
	$(CARGO) clean

examples: generate-examples  ## Generate and run all examples
	$(CARGO) test -p mlmd-examples
	cd examples && uv run pytest -v

generate-examples: build-release  ## Generate output from all .mlmd example files
	for f in examples/*.mlmd; do \
		echo "Generating from $$f..."; \
		$(MLMD_RELEASE) generate -t pytorch -o examples "$$f"; \
		$(MLMD_RELEASE) generate -t candle -o examples "$$f"; \
		$(MLMD_RELEASE) generate -t keras -o examples "$$f"; \
		$(MLMD_RELEASE) visualize "$$f" -o "examples/$$(basename "$$f" .mlmd).svg"; \
	done

check:  ## Run cargo check on the workspace (quick compile check)
	@echo "Checking code..."
	$(CARGO) check --workspace
