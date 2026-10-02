#!/usr/bin/env bash
# wait_then.sh <pattern> <script> <log>: start <script> once no process matches <pattern>.
# Pass a bracketed pattern (e.g. "[e]1_job.sh") so pgrep cannot match this waiter's own command line.
cd "$(dirname "$0")/../.."
while pgrep -f "$1" > /dev/null; do sleep 30; done
bash "$2" > "$3" 2>&1
echo "exit $?" >> "$3"
