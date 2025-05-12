# Build, package, test, and clean
PROJECT=verde
CHECK_STYLE=src/$(PROJECT) doc test

.PHONY: build install test format check check-format check_style check-actions clean

help:
	@echo "Commands:"
	@echo ""
	@echo "  install   install in editable mode"
	@echo "  test      run the test suite (including doctests) and report coverage"
	@echo "  format    automatically format the code"
	@echo "  check     run code style and quality checks"
	@echo "  build     build source and wheel distributions"
	@echo "  clean     clean up build and generated files"
	@echo ""

build:
	python -m build .

install:
	python -m pip install --no-deps -e .

test:
	pytest --cov-report=term-missing --cov --doctest-modules --verbose test src/$(PROJECT)

format:
	isort $(CHECK_STYLE)
	black $(CHECK_STYLE)
	burocrata --extension=py $(CHECK_STYLE)

check: check-format check-style

check-format:
	black --check $(CHECK_STYLE)
	isort --check $(CHECK_STYLE)
	burocrata --check --extension=py $(CHECK_STYLE)

check-style:
	flake8 $(CHECK_STYLE)

clean:
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".coverage.*" -exec rm -v {} \;
	find . -name "*.orig" -exec rm -v {} \;
	find . -name "__pycache__" -exec rm -v {} \;
	rm src/$(PROJECT)/_version.py
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache .pytest_cache
	rm -rvf dask-worker-space
