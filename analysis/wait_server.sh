#!/bin/bash
# Poll the scaled 9B SGLang server until healthy (up to 30 min cold start).
# Prints SERVER_HEALTHY on success, POLL_TIMEOUT_30MIN otherwise.
URL="https://<your-workspace>--cgr-sglang-serve.modal.run/v1/models"
i=0
until curl -sf -o /dev/null "$URL"; do
  i=$((i + 1))
  if [ "$i" -ge 180 ]; then
    echo "POLL_TIMEOUT_60MIN"
    exit 1
  fi
  sleep 20
done
echo "SERVER_HEALTHY"
