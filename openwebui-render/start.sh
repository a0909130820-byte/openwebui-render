#!/usr/bin/env bash
PORT="${PORT:-10000}"
open-webui serve --host 0.0.0.0 --port "$PORT"