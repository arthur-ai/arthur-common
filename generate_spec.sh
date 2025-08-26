#!/usr/bin/env bash
# Must run via poetry to avoid a stale python cache after making changes.

poetry run python generate_openapi_spec.py
