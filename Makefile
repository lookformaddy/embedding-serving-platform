.PHONY: test demo install
install:
	pip install -r requirements.txt
test:
	python -m pytest -q
demo:
	PYTHONPATH=src python examples/run_demo.py
