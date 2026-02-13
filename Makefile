BACKEND_DIR := backend
COMPOSE_FILE := $(BACKEND_DIR)/docker-compose.corpus.yml
PODMAN_COMPOSE := podman compose -f $(COMPOSE_FILE)

.PHONY: up down test-unit test-integration

up:
	$(PODMAN_COMPOSE) up -d

down:
	$(PODMAN_COMPOSE) down --remove-orphans

test-unit:
	cd $(BACKEND_DIR) && uv run pytest -m "not integration"

test-integration:
	@set -e; \
	$(MAKE) up; \
	test_status=0; \
	(cd $(BACKEND_DIR) && uv run pytest tests/integration -m integration -v) || test_status=$$?; \
	$(MAKE) -C $(CURDIR) down; \
	exit $$test_status
