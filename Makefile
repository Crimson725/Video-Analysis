BACKEND_DIR := backend
COMPOSE_FILE := $(BACKEND_DIR)/docker-compose.corpus.yml
PODMAN := podman
PODMAN_COMPOSE := $(PODMAN) compose -f $(COMPOSE_FILE)
CORPUS_NEO4J_HTTP_PORT ?= 47474
CORPUS_NEO4J_BOLT_PORT ?= 47687
CORPUS_PGVECTOR_PORT ?= 45433
CORPUS_NEO4J_AUTH ?= neo4j/local-dev-password
CORPUS_PGVECTOR_USER ?= video_analysis
CORPUS_PGVECTOR_PASSWORD ?= video_analysis
CORPUS_PGVECTOR_DB ?= video_analysis
COMPOSE_ENV := \
	CORPUS_NEO4J_HTTP_PORT=$(CORPUS_NEO4J_HTTP_PORT) \
	CORPUS_NEO4J_BOLT_PORT=$(CORPUS_NEO4J_BOLT_PORT) \
	CORPUS_PGVECTOR_PORT=$(CORPUS_PGVECTOR_PORT) \
	CORPUS_NEO4J_AUTH=$(CORPUS_NEO4J_AUTH) \
	CORPUS_PGVECTOR_USER=$(CORPUS_PGVECTOR_USER) \
	CORPUS_PGVECTOR_PASSWORD=$(CORPUS_PGVECTOR_PASSWORD) \
	CORPUS_PGVECTOR_DB=$(CORPUS_PGVECTOR_DB)

.PHONY: ensure-podman up down test-unit test-integration

ensure-podman:
	@set -eu; \
	command -v $(PODMAN) >/dev/null 2>&1 || { \
		echo "podman is required but not installed."; \
		exit 127; \
	}; \
	if $(PODMAN) info >/dev/null 2>&1; then \
		exit 0; \
	fi; \
	machine_name="$$($(PODMAN) machine list --format '{{.Name}} {{.Default}}' 2>/dev/null | awk '$$2 == "true" { print $$1; exit }')"; \
	if [ -z "$$machine_name" ]; then \
		machine_name="podman-machine-default"; \
	fi; \
	echo "Podman is not reachable. Starting machine '$$machine_name'..."; \
	$(PODMAN) machine start "$$machine_name" >/dev/null 2>&1 || { \
		echo "Machine '$$machine_name' not ready. Initializing if needed..."; \
		$(PODMAN) machine init "$$machine_name" >/dev/null 2>&1 || true; \
		$(PODMAN) machine start "$$machine_name"; \
	}; \
	$(PODMAN) info >/dev/null 2>&1 || { \
		echo "Unable to connect to Podman after starting '$$machine_name'."; \
		echo "Check: podman system connection list"; \
		exit 125; \
	}

up: ensure-podman
	@set -eu; \
	python_cmd="$$(command -v python3 || command -v python || true)"; \
	[ -n "$$python_cmd" ] || { \
		echo "python or python3 is required to auto-select free ports for make up."; \
		exit 127; \
	}; \
	port_in_use() { \
		"$$python_cmd" -c 'import socket, sys; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(0.2); in_use = s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0; s.close(); print("1" if in_use else "0")' "$$1"; \
	}; \
	pick_port() { \
		port="$$1"; \
		reserved_a="$${2:-}"; \
		reserved_b="$${3:-}"; \
		while :; do \
			if [ "$$port" = "$$reserved_a" ] || [ "$$port" = "$$reserved_b" ] || [ "$$(port_in_use "$$port")" = "1" ]; then \
				port=$$((port + 1)); \
				if [ "$$port" -gt 65535 ]; then \
					echo "Unable to find an available TCP port."; \
					exit 125; \
				fi; \
				continue; \
			fi; \
			printf '%s' "$$port"; \
			return 0; \
		done; \
	}; \
	neo4j_http_port="$$(pick_port "$(CORPUS_NEO4J_HTTP_PORT)")"; \
	neo4j_bolt_port="$$(pick_port "$(CORPUS_NEO4J_BOLT_PORT)" "$$neo4j_http_port")"; \
	pgvector_port="$$(pick_port "$(CORPUS_PGVECTOR_PORT)" "$$neo4j_http_port" "$$neo4j_bolt_port")"; \
	if [ "$$neo4j_http_port" != "$(CORPUS_NEO4J_HTTP_PORT)" ] || [ "$$neo4j_bolt_port" != "$(CORPUS_NEO4J_BOLT_PORT)" ] || [ "$$pgvector_port" != "$(CORPUS_PGVECTOR_PORT)" ]; then \
		echo "Some requested ports are in use. Using free ports instead:"; \
	fi; \
	echo "  Neo4j HTTP: $$neo4j_http_port (requested $(CORPUS_NEO4J_HTTP_PORT))"; \
	echo "  Neo4j Bolt: $$neo4j_bolt_port (requested $(CORPUS_NEO4J_BOLT_PORT))"; \
	echo "  pgvector:   $$pgvector_port (requested $(CORPUS_PGVECTOR_PORT))"; \
	CORPUS_NEO4J_HTTP_PORT="$$neo4j_http_port" \
	CORPUS_NEO4J_BOLT_PORT="$$neo4j_bolt_port" \
	CORPUS_PGVECTOR_PORT="$$pgvector_port" \
	CORPUS_NEO4J_AUTH="$(CORPUS_NEO4J_AUTH)" \
	CORPUS_PGVECTOR_USER="$(CORPUS_PGVECTOR_USER)" \
	CORPUS_PGVECTOR_PASSWORD="$(CORPUS_PGVECTOR_PASSWORD)" \
	CORPUS_PGVECTOR_DB="$(CORPUS_PGVECTOR_DB)" \
	$(PODMAN_COMPOSE) up -d

down: ensure-podman
	$(COMPOSE_ENV) $(PODMAN_COMPOSE) down --remove-orphans

test-unit:
	cd $(BACKEND_DIR) && uv run pytest -m "not integration"

test-integration:
	@set -e; \
	$(MAKE) ensure-podman; \
	python_cmd="$$(command -v python3 || command -v python || true)"; \
	[ -n "$$python_cmd" ] || { \
		echo "python or python3 is required to auto-select free ports for test-integration."; \
		exit 127; \
	}; \
	port_in_use() { \
		"$$python_cmd" -c 'import socket, sys; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(0.2); in_use = s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0; s.close(); print("1" if in_use else "0")' "$$1"; \
	}; \
	pick_port() { \
		port="$$1"; \
		reserved_a="$${2:-}"; \
		reserved_b="$${3:-}"; \
		while :; do \
			if [ "$$port" = "$$reserved_a" ] || [ "$$port" = "$$reserved_b" ] || [ "$$(port_in_use "$$port")" = "1" ]; then \
				port=$$((port + 1)); \
				if [ "$$port" -gt 65535 ]; then \
					echo "Unable to find an available TCP port."; \
					exit 125; \
				fi; \
				continue; \
			fi; \
			printf '%s' "$$port"; \
			return 0; \
		done; \
	}; \
	neo4j_http_port="$$(pick_port "$(CORPUS_NEO4J_HTTP_PORT)")"; \
	neo4j_bolt_port="$$(pick_port "$(CORPUS_NEO4J_BOLT_PORT)" "$$neo4j_http_port")"; \
	pgvector_port="$$(pick_port "$(CORPUS_PGVECTOR_PORT)" "$$neo4j_http_port" "$$neo4j_bolt_port")"; \
	if [ "$$neo4j_http_port" != "$(CORPUS_NEO4J_HTTP_PORT)" ] || [ "$$neo4j_bolt_port" != "$(CORPUS_NEO4J_BOLT_PORT)" ] || [ "$$pgvector_port" != "$(CORPUS_PGVECTOR_PORT)" ]; then \
		echo "Some requested ports are in use for test-integration. Using free ports instead:"; \
	fi; \
	echo "  Neo4j HTTP: $$neo4j_http_port (requested $(CORPUS_NEO4J_HTTP_PORT))"; \
	echo "  Neo4j Bolt: $$neo4j_bolt_port (requested $(CORPUS_NEO4J_BOLT_PORT))"; \
	echo "  pgvector:   $$pgvector_port (requested $(CORPUS_PGVECTOR_PORT))"; \
	CORPUS_NEO4J_HTTP_PORT="$$neo4j_http_port" CORPUS_NEO4J_BOLT_PORT="$$neo4j_bolt_port" CORPUS_PGVECTOR_PORT="$$pgvector_port" CORPUS_NEO4J_AUTH="$(CORPUS_NEO4J_AUTH)" CORPUS_PGVECTOR_USER="$(CORPUS_PGVECTOR_USER)" CORPUS_PGVECTOR_PASSWORD="$(CORPUS_PGVECTOR_PASSWORD)" CORPUS_PGVECTOR_DB="$(CORPUS_PGVECTOR_DB)" $(PODMAN_COMPOSE) down -v --remove-orphans >/dev/null 2>&1 || true; \
	CORPUS_NEO4J_HTTP_PORT="$$neo4j_http_port" CORPUS_NEO4J_BOLT_PORT="$$neo4j_bolt_port" CORPUS_PGVECTOR_PORT="$$pgvector_port" CORPUS_NEO4J_AUTH="$(CORPUS_NEO4J_AUTH)" CORPUS_PGVECTOR_USER="$(CORPUS_PGVECTOR_USER)" CORPUS_PGVECTOR_PASSWORD="$(CORPUS_PGVECTOR_PASSWORD)" CORPUS_PGVECTOR_DB="$(CORPUS_PGVECTOR_DB)" $(PODMAN_COMPOSE) up -d; \
	test_status=0; \
	(cd $(BACKEND_DIR) && uv run pytest tests/integration -m integration -v) || test_status=$$?; \
	CORPUS_NEO4J_HTTP_PORT="$$neo4j_http_port" CORPUS_NEO4J_BOLT_PORT="$$neo4j_bolt_port" CORPUS_PGVECTOR_PORT="$$pgvector_port" CORPUS_NEO4J_AUTH="$(CORPUS_NEO4J_AUTH)" CORPUS_PGVECTOR_USER="$(CORPUS_PGVECTOR_USER)" CORPUS_PGVECTOR_PASSWORD="$(CORPUS_PGVECTOR_PASSWORD)" CORPUS_PGVECTOR_DB="$(CORPUS_PGVECTOR_DB)" $(PODMAN_COMPOSE) down -v --remove-orphans >/dev/null 2>&1 || true; \
	exit $$test_status
