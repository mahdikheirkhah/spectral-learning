# --- Configuration ---
PROJECT_NAME = spectral-learning
PORT = 8891
PYTHON_VERSION = 3.10

# OS Detection
ifeq ($(OS),Windows_NT)
    # Windows Settings
    PYTHON = python
    OPEN = start
    RM = rmdir /s /q
    CLEAN_FILES = .venv
    # Port killing for Windows
    KILL_PORT = powershell -Command "Stop-Process -Id (Get-NetTCPConnection -LocalPort $(PORT)).OwningProcess -Force -ErrorAction SilentlyContinue"
else
    # macOS / Linux Settings
    PYTHON = python3
    OPEN = open
    RM = rm -rf
    CLEAN_FILES = .venv .jupyter
    # Port killing for Unix
    KILL_PORT = lsof -ti:$(PORT) | xargs kill -9 2>/dev/null || true
endif

# --- Professional Workflow Targets ---

.PHONY: setup install kernel run clean test

## setup: Initialize poetry and install dependencies
setup:
	@echo "Initializing Poetry for $(PROJECT_NAME)..."
	poetry env use $(PYTHON_VERSION)
	poetry install
	@$(MAKE) kernel

## install: Install all dependencies from pyproject.toml
install:
	poetry install

## kernel: Register the poetry environment as a Jupyter Kernel
kernel:
	@echo "Installing Jupyter kernel..."
	poetry run python -m ipykernel install --user --name=$(PROJECT_NAME) --display-name="Python ($(PROJECT_NAME))"

## run: Launch Jupyter Notebook on the specified port
run:
	@echo "Starting Jupyter server on port $(PORT)..."
	@$(KILL_PORT)
	poetry run jupyter notebook --port $(PORT) --no-browser

## test: Run the pytest suite (as per Contributing.md)
test:
	poetry run pytest tests/

## clean: Remove virtual environments and temporary files
clean:
	@echo "Cleaning up..."
	@$(KILL_PORT)
	poetry env remove --all || true
	$(RM) $(CLEAN_FILES)
	@echo "✅ Cleanup complete."

## help: Show this help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'