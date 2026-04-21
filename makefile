# --- Configuration ---
PROJECT_NAME = spectral-learning
PORT = 8891
PYTHON_VERSION = 3.11
VENV = .venv

# OS Detection and Path Configuration
ifeq ($(OS),Windows_NT)
    PYTHON = python
    BIN = $(VENV)/Scripts
    ACTIVATE_CMD = .\$(VENV)\Scripts\activate
    RM = rmdir /s /q
else
    PYTHON = python3
    BIN = $(VENV)/bin
    ACTIVATE_CMD = source $(VENV)/bin/activate
    RM = rm -rf
endif

PIP = $(BIN)/pip
POETRY = $(BIN)/poetry

# --- Professional Workflow Targets ---

.PHONY: setup install add activate kernel run clean help boilerplate
## setup: Full environment initialization with Poetry init and boilerplate files
setup: boilerplate
	@echo "Step 1: Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "------------------------------------------------"
	@echo "✅ Virtual environment created at $(VENV)"
	@echo "⚠️  To activate this environment in your terminal, run:"
	@echo "   $(ACTIVATE_CMD)"
	@echo "------------------------------------------------"
	@echo "Step 2: Installing Poetry inside the environment..."
	$(PIP) install --upgrade pip poetry
	@echo "Step 3: Checking for pyproject.toml..."
	@if [ ! -f pyproject.toml ]; then \
		echo "No pyproject.toml found. Initializing Poetry..."; \
		$(POETRY) init; \
	fi
	@echo "Step 4: Configuring Poetry..."
	$(POETRY) config virtualenvs.in-project true --local
	@echo "Step 5: Installing dependencies..."
	$(POETRY) install --no-root
	@$(MAKE) kernel

## boilerplate: Create folders and files so Poetry and Python work correctly
boilerplate:
	@echo "Generating boilerplate project structure..."
	@# Create package directories
	@mkdir -p models utils tests
	@touch models/__init__.py utils/__init__.py tests/__init__.py
	@# Generate .gitignore
	@echo ".venv\n.DS_Store\n.ipynb_checkpoints\n__pycache__\n*.pyc\n.vscode\n.idea\n.env\nbuild/\ndist/\n.pytest_cache/\n*.coverage\nhtmlcov/\n*.csv\n*/*.npy" > .gitignore
	@# Generate README.md if missing
	@if [ ! -f README.md ]; then \
		echo "# $(PROJECT_NAME)\n\nSpectral methods for dimensionality reduction." > README.md; \
		echo "✅ README.md created."; \
	fi
	@echo "✅ Project structure and .gitignore updated."

## activate: Show the command to activate the environment
activate:
	@echo "Run this command to activate your environment:"
	@echo "$(ACTIVATE_CMD)"

## install: Install all dependencies defined in pyproject.toml
install:
	$(POETRY) install

## add: Install a new dependency (Usage: make add pkg=loguru)
add:
	@if [ -z "$(pkg)" ]; then \
		echo "Usage: make add pkg=<package_name>"; \
	else \
		$(POETRY) add $(pkg); \
	fi

## kernel: Register the poetry environment as a Jupyter Kernel
kernel:
	$(POETRY) run python -m ipykernel install --user --name=$(PROJECT_NAME) --display-name="Python ($(PROJECT_NAME))"

## run: Launch Jupyter Notebook
run:
	$(POETRY) run jupyter notebook --port $(PORT) --no-browser

## clean: Wipe the local environment
clean:
	@$(RM) $(VENV)
	@echo "✅ Cleanup complete."

## help: Show this help message
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'