#!/usr/bin/env bash

# Create docker net
docker network create lithium-net

# docker-compose
docker-compose up -d