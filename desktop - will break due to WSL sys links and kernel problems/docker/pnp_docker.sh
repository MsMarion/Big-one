#!/usr/bin/env bash
# `docker`, but always talking to the project's Docker Engine inside WSL (its own socket), never to
# Docker Desktop. Usage: bash ~/pnp_repo/docker/pnp_docker.sh ps
exec /usr/bin/docker -H unix:///var/run/docker-pnp.sock "$@"
