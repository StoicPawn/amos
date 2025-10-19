.PHONY: lint test format

lint:
ruff traderx traderx_ibkr tests

format:
black traderx traderx_ibkr tests

test:
pytest
