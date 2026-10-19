.PHONY: build up down logs test test-coverage cli clean

# Build all images
build:
	docker compose build

# Start Streamlit UI (port 8501)
up:
	docker compose up -d code-review-agent

# Start UI and follow logs
up-logs:
	docker compose up code-review-agent

# Stop all services
down:
	docker compose down

# View logs
logs:
	docker compose logs -f

# Run unit tests
test:
	docker compose --profile test run --rm test

# Run tests with coverage
test-coverage:
	docker compose --profile test run --rm test-coverage

# Run CLI against a diff file (usage: make cli DIFF=sample.diff)
cli:
	docker compose --profile cli run --rm code-review-cli --diff /workspace/$(DIFF)

# Clean up containers, images, volumes
clean:
	docker compose down -v --rmi all --remove-orphans

# Full rebuild and test
ci: build test