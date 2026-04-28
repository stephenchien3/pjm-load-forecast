.PHONY: install test coverage lint clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=pjm_load_forecast --cov-report=term-missing --cov-fail-under=80

coverage:
	pytest --cov=pjm_load_forecast --cov-report=html
	@echo "Open htmlcov/index.html in a browser"

lint:
	ruff check pjm_load_forecast tests

clean:
	rm -rf .pytest_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
