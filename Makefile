# Makefile

# --- Configuration ---
# Define variables for paths and names
# Installation type: 'user' for ~/.local, 'system' for /usr/local
INSTALL_TYPE ?= user

# Set PREFIX based on INSTALL_TYPE
ifeq ($(INSTALL_TYPE), system)
    PREFIX ?= /usr/local
else
    PREFIX ?= $(HOME)/.local
endif

CARGO_TARGET_DIR ?= target

PROJECT_NAME := koko

# Source paths for data files (relative to the project root where Makefile is)
# These will be created/populated by the download process
CHECKPOINT_DIR := checkpoints
CHECKPOINT_FILE := $(CHECKPOINT_DIR)/kokoro-v1.0.onnx
CHECKPOINT_FILE_URL := https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
CHECKPOINT_HASH_FILE := $(CHECKPOINT_DIR)/kokoro-v1.0.onnx.sha256
VOICES_DIR := data
VOICES_FILE := $(VOICES_DIR)/voices-v1.0.bin
VOICES_FILE_URL := https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
VOICES_HASH_FILE := $(VOICES_DIR)/voices-v1.0.bin.sha256

# Destination paths (using XDG conventions for user install)
XDG_DATA_HOME ?= $(HOME)/.local/share
INSTALL_DATA_DIR := $(XDG_DATA_HOME)/$(PROJECT_NAME)

# For system-wide installs, data files go to $(PREFIX)/share
SYSTEM_DATA_DIR := $(PREFIX)/share/$(PROJECT_NAME)

# Actual data directory to use based on installation type
ifeq ($(INSTALL_TYPE), system)
    DATA_DIR := $(SYSTEM_DATA_DIR)
else
    DATA_DIR := $(INSTALL_DATA_DIR)
endif

# --- Targets ---

# Default target (often 'all' or the main binary)
.PHONY: all
all: build

# Build the project using cargo
.PHONY: build
build:
	cargo build --release

# Download required data files (models, voices) with integrity check
.PHONY: data setup
data setup:
	@echo "Setting up required data files..."
	@mkdir -p "$(CHECKPOINT_DIR)" "$(VOICES_DIR)"
	
	# Check and download checkpoint file
	@if [ ! -f "$(CHECKPOINT_FILE)" ]; then \
		echo "File $(CHECKPOINT_FILE) not found. Downloading..."; \
		wget -O "$(CHECKPOINT_FILE)" "$(CHECKPOINT_FILE_URL)" || exit 1; \
		echo "Downloaded $(CHECKPOINT_FILE)"; \
	elif [ -f "$(CHECKPOINT_FILE)" ] && [ ! -f "$(CHECKPOINT_HASH_FILE)" ]; then \
		echo "Hash file $(CHECKPOINT_HASH_FILE) not found."; \
	elif [ -f "$(CHECKPOINT_FILE)" ] && [ -f "$(CHECKPOINT_HASH_FILE)" ]; then \
		echo "File $(CHECKPOINT_FILE) exists. Verifying integrity..."; \
		EXPECTED_HASH=$$(cut -d ' ' -f 1 "$(CHECKPOINT_HASH_FILE)"); \
		ACTUAL_HASH=$$(sha256sum "$(CHECKPOINT_FILE)" | cut -d ' ' -f 1); \
		if [ "$$ACTUAL_HASH" != "$$EXPECTED_HASH" ]; then \
			echo "Hash mismatch for $(CHECKPOINT_FILE). Expected: $$EXPECTED_HASH, Got: $$ACTUAL_HASH. Re-downloading..."; \
			wget -O "$(CHECKPOINT_FILE)" "$(CHECKPOINT_FILE_URL)" || exit 1; \
			echo "Re-downloaded $(CHECKPOINT_FILE)"; \
		else \
			echo "Hash verified for $(CHECKPOINT_FILE)."; \
		fi; \
	fi
	
	# Check and download voices file
	@if [ ! -f "$(VOICES_FILE)" ]; then \
		echo "File $(VOICES_FILE) not found. Downloading..."; \
		wget -O "$(VOICES_FILE)" "$(VOICES_FILE_URL)" || exit 1; \
		echo "Downloaded $(VOICES_FILE)"; \
	elif [ -f "$(VOICES_FILE)" ] && [ ! -f "$(VOICES_HASH_FILE)" ]; then \
		echo "Hash file $(VOICES_HASH_FILE) not found."; \
	elif [ -f "$(VOICES_FILE)" ] && [ -f "$(VOICES_HASH_FILE)" ]; then \
		echo "File $(VOICES_FILE) exists. Verifying integrity..."; \
		EXPECTED_HASH=$$(cut -d ' ' -f 1 "$(VOICES_HASH_FILE)"); \
		ACTUAL_HASH=$$(sha256sum "$(VOICES_FILE)" | cut -d ' ' -f 1); \
		if [ "$$ACTUAL_HASH" != "$$EXPECTED_HASH" ]; then \
			echo "Hash mismatch for $(VOICES_FILE). Expected: $$EXPECTED_HASH, Got: $$ACTUAL_HASH. Re-downloading..."; \
			wget -O "$(VOICES_FILE)" "$(VOICES_FILE_URL)" || exit 1; \
			echo "Re-downloaded $(VOICES_FILE)"; \
		else \
			echo "Hash verified for $(VOICES_FILE)."; \
		fi; \
	fi
	
	@echo "Data setup complete."

# Install the binary and data files
.PHONY: install
install: build data
	# Install the binary
	install -Dm755 "$(CARGO_TARGET_DIR)/release/$(PROJECT_NAME)" "$(DESTDIR)$(PREFIX)/bin/$(PROJECT_NAME)"

	# Create the destination data directory
	install -dm755 "$(DESTDIR)$(DATA_DIR)"

	# Install the data files from their local downloaded locations
	install -Dm644 "$(CHECKPOINT_FILE)" "$(DESTDIR)$(DATA_DIR)/"
	install -Dm644 "$(VOICES_FILE)" "$(DESTDIR)$(DATA_DIR)/"
	# Also install the hash files
	install -Dm644 "$(CHECKPOINT_HASH_FILE)" "$(DESTDIR)$(DATA_DIR)/"
	install -Dm644 "$(VOICES_HASH_FILE)" "$(DESTDIR)$(DATA_DIR)/"

	@echo "Installed $(PROJECT_NAME) to $(DESTDIR)$(PREFIX)/bin/"
	@echo "Installed data files to $(DESTDIR)$(DATA_DIR)/"
	@echo "To use the application, ensure your application looks for data in the correct location:"
	@if [ "$(INSTALL_TYPE)" = "system" ]; then \
		echo "  - For system-wide installs: $(SYSTEM_DATA_DIR)"; \
	else \
		echo "  - For user installs: $$XDG_DATA_HOME/$(PROJECT_NAME)"; \
	fi

# Clean build artifacts
.PHONY: clean
clean:
	cargo clean
	# Note: Downloaded data (checkpoints/, data/) is not automatically removed.
	# Run 'make clean-data' if you wish to remove it as well.

# Clean downloaded data files
.PHONY: clean-data
clean-data:
	rm -rf "$(CHECKPOINT_DIR)" "$(VOICES_DIR)"
	@echo "Removed downloaded data directories: $(CHECKPOINT_DIR), $(VOICES_DIR)"

# Run tests
.PHONY: test
test:
	cargo test

# Run linter (clippy)
.PHONY: lint
lint:
	cargo clippy --all-targets --all-features

# Help
.PHONY: help
help:
	@echo "Usage: make [target] [INSTALL_TYPE=user|system]"
	@echo ""
	@echo "Available targets:"
	@echo "  all          Build the project (default)"
	@echo "  build        Build the project in release mode"
	@echo "  data         Download required data files (aliases: setup) with integrity check"
	@echo "  install      Install the binary and data files (requires 'build' and 'data')"
	@echo "               Use INSTALL_TYPE=system for system-wide installation (default: user)"
	@echo "               Use PREFIX=... to set installation prefix"
	@echo "               Use DESTDIR=... for staged installs"
	@echo "  clean        Remove build artifacts"
	@echo "  clean-data   Remove downloaded data files (checkpoints/, data/)"
	@echo "  test         Run project tests"
	@echo "  lint         Run the linter (clippy)"
	@echo "  help         Show this help message"
