NAME := tlo
SRC := ./src/core/tlo
INSTALL_STAMP := .install.tlo
POETRY := $(shell command -v poetry 2> /dev/null)
PTR_TML := pyproject.toml 
PTR_LCK ?= poetry.lock

default: help
all: install lint format test clean

help:
		@echo "Please use 'make <target>' where <target> is one of"
		@echo ""
		@echo "  install     install packages and prepare environment"
		@echo "  clean       remove all temporary files"
		@echo "  lint        run the code linters"
		@echo "  format      reformat code"
		@echo "  test        run all the tests"
		@echo "  run         run train agent script"
		@echo ""
		@echo "Check the Makefile to know exactly what each target is doing."

install: $(INSTALL_STAMP)
$(INSTALL_STAMP): $(PTR_TML)
		@if [ -z $(POETRY) ]; then echo "Poetry could not be found. See https://python-poetry.org/docs/"; exit 2; fi
		export SUMO_HOME="/usr/share/sumo"
		export PYTHONPATH=$(PYTHONPATH):$(PWD)
		$(POETRY) install
		touch $(INSTALL_STAMP)

train: $(INSTALL_STAMP)
		echo $(POETRY)
		$(POETRY) run python3 $(SRC)/train.py

inference: $(INSTALL_STAMP)
		echo $(POETRY)
		$(POETRY) run python3 $(SRC)/inference.py

clean:
		find . -type d -name "__pycache__" | xargs rm -rf {};
		rm -rf $(INSTALL_STAMP) .coverage .mypy_cache .pytest_cache

lint: $(INSTALL_STAMP)
		$(POETRY) run isort --profile=black --lines-after-imports=2 --check-only $(SRC)
		$(POETRY) run black --check $(SRC) --diff
		$(POETRY) run flake8 --ignore=W503,E501 $(SRC)
		$(POETRY) run mypy $(SRC)
		$(POETRY) run bandit -r $(SRC) -s B608

pre_commit: $(INSTALL_STAMP)
		$(POETRY) run pre-commit run --files src/tlo/*

format: $(INSTALL_STAMP)
		$(POETRY) run isort --profile=black --lines-after-imports=2 $(SRC)
		$(POETRY) run black $(SRC)

test: $(INSTALL_STAMP)
		$(POETRY) run pytest $(SRC) --cov-report html --cov-report xml --cov-fail-under 100 --cov=$(NAME)

.PHONY: help lint format test clean all
