#!/usr/bin/env bash
# wait_then.sh <pattern-of-running-job> <script> <log>: start <script> once no process matches <pattern>.
cd "$(dirname "$0")/../.."
while pgrep -f "$1" > /dev/null; do sleep 30; done
bash "$2" > "$3" 2>&1
echo "exit $?" >> "$3"
