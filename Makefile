VENV_DIR ?= .venv
PYTHON ?= python3
PIP := $(VENV_DIR)/bin/pip

.PHONY: venv syntax-check check encrypt-vault deploy deploy-pass task

venv:
	@if [ ! -x "$(PIP)" ]; then \
		$(PYTHON) -m venv "$(VENV_DIR)"; \
		"$(PIP)" install --upgrade pip; \
		"$(PIP)" install -r requirements.txt; \
	fi

train: venv
	python3 src/main.py

predict: venv
	python3 src/predict.py	