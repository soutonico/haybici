#!/usr/bin/env bash
export PYTHONPATH=.
uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload
