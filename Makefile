SHELL := /bin/bash

# Defaults for backtest args
DATE ?= 2025-08-13
START ?= 06:00
END ?= 11:30
ACCOUNT ?= 30000

.PHONY: setup backtest lint fmt

setup:
	@echo "[setup] Creating/activating venv and installing requirements..."
	@if [ ! -d venv ]; then python3 -m venv venv; fi
	@source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "[setup] Done. Activate with: source venv/bin/activate"

backtest:
	@echo "[backtest] Running backtest for $(DATE) $(START)-$(END) ..."
	@source venv/bin/activate && \
		python3 warrior_backtest_main.py \
		--date $(DATE) \
		--start-time $(START) \
		--end-time $(END) \
		--account $(ACCOUNT) \
		--log-level INFO
	@echo "[backtest] Log written to results/logs/backtest_$(DATE).log"

