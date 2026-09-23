# aether-ai-core 开发/部署命令（作者：晨星）

.PHONY: install dev test eval lint serve lock docker build clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,world-class]"

test:
	pytest -q

eval:
	python -m aether eval

lint:
	ruff check src tests

serve:
	python -m aether serve

lock:
	pip freeze > requirements.lock.txt

docker:
	docker build -t aether-ai-core .

build: docker

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .chroma *.egg-info
