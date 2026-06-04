CARGO ?= cargo
MLMD ?= cargo run -p mlmd --

.PHONY: help build test test-examples test-rust-examples install install-zed \
        install-vscode lint fmt clean examples generate-examples check

help:  ## Show this help
	@awk -F '## ' '/^[a-zA-Z_-]+:.*##/ { printf "\033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

build:  ## Build the workspace (all crates)
	@echo "Building workspace..."
	$(CARGO) build --workspace

test: test-examples test-rust-examples  ## Run all tests (Rust + Python examples)
	@echo "Running tests..."
	$(CARGO) test --workspace

test-examples:  ## Run Python example tests
	cd examples && uv sync && uv run pytest -v

test-rust-examples:  ## Run Rust example tests
	$(CARGO) test -p mlmd-examples

install:  ## Build release binary and install to /usr/local/bin
	$(CARGO) build --release -p mlmd
	cp target/release/mlmd /usr/local/bin/

install-zed:  ## Install mlmd Zed extension
	$(MLMD) install zed

install-vscode:  ## Install mlmd VS Code extension
	$(MLMD) install vscode

lint:  ## Run clippy linting on the workspace
	$(CARGO) clippy --workspace -- -D warnings 2>/dev/null || $(CARGO) clippy --workspace

fmt:  ## Check Rust formatting
	$(CARGO) fmt --check

clean:  ## Clean all build artifacts
	@echo "Cleaning build artifacts..."
	$(CARGO) clean

examples: generate-examples  ## Generate and run all examples
	$(CARGO) test -p mlmd-examples
	cd examples && uv run pytest -v

generate-examples:  ## Generate output from all .mlmd example files
	for f in examples/*.mlmd; do \
		echo "Generating from $$f..."; \
		$(MLMD) generate -t pytorch -o examples "$$f"; \
		$(MLMD) generate -t candle -o examples "$$f"; \
		$(MLMD) generate -t keras -o examples "$$f"; \
	done

check:  ## Run cargo check on the workspace (quick compile check)
	@echo "Checking code..."
	$(CARGO) check --workspace
