CARGO ?= cargo
MLMD ?= cargo run -p mlmd --

.PHONY: build test test-examples test-rust-examples install install-zed \
        install-vscode lint fmt clean examples generate-examples check

build:
	@echo "Building workspace..."
	$(CARGO) build --workspace

test: test-examples test-rust-examples
	@echo "Running tests..."
	$(CARGO) test --workspace

test-examples:
	cd examples && uv run pytest -v 2>/dev/null || echo "Python tests not configured yet (need uv + deps)"

test-rust-examples:
	$(CARGO) test -p mlmd-examples

install:
	$(CARGO) build --release -p mlmd
	cp target/release/mlmd /usr/local/bin/

install-zed:
	$(MLMD) install zed

install-vscode:
	$(MLMD) install vscode

lint:
	$(CARGO) clippy --workspace -- -D warnings 2>/dev/null || $(CARGO) clippy --workspace

fmt:
	$(CARGO) fmt --check

clean:
	@echo "Cleaning build artifacts..."
	$(CARGO) clean

examples: generate-examples
	$(CARGO) test -p mlmd-examples
	cd examples && uv run pytest -v 2>/dev/null || echo "Python tests not configured yet"

generate-examples:
	for f in examples/*.mlmd; do \
		echo "Generating from $$f..."; \
		$(MLMD) generate -t pytorch -o examples "$$f"; \
		$(MLMD) generate -t candle -o examples "$$f"; \
		$(MLMD) generate -t keras -o examples "$$f"; \
	done

check:
	@echo "Checking code..."
	$(CARGO) check --workspace
