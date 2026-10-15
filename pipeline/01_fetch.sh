#!/bin/bash
# Fetch episode lists and replays for every submission in a teacher list.
# usage: pipeline/01_fetch.sh pipeline/teachers/pretrain67.txt
set -e
cd "$(dirname "$0")/.."
for s in $(tr ',' ' ' < "$1"); do
  python tools/fetch/fetch_replays.py "$s"
done
