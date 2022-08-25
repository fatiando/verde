# Build, package, test, and clean
PROJECT=verde
TESTDIR=tmp-test-dir-with-unique-name
PYTEST_ARGS=--cov-config=../.coveragerc --cov-report=term-missing --cov=$(PROJECT) --doctest-modules -v --pyargs
CHECK_STYLE=$(PROJECT) doc tools

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
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); MPLBACKEND='agg' pytest $(PYTEST_ARGS) $(PROJECT)
	cp $(TESTDIR)/.coverage* .
	rm -rvf $(TESTDIR)

format: license isort black

check: black-check isort-check license-check flake8

black:
	black $(CHECK_STYLE)

black-check:
	black --check $(CHECK_STYLE)

license:
	python tools/license_notice.py

license-check:
	python tools/license_notice.py --check

isort:
	isort $(CHECK_STYLE)

isort-check:
	isort --check $(CHECK_STYLE)

flake8:
	flake8 $(CHECK_STYLE)

clean:
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name ".coverage.*" -exec rm -v {} \;
	find . -name "*.orig" -exec rm -v {} \;
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache .pytest_cache
	rm -rvf $(TESTDIR) dask-worker-space
