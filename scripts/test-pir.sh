#!/bin/bash
set -e

cd "$(dirname "$0")/.."

SERVER="${1:-localhost:8088}"
ROW="${2:-42}"

echo "=== PIR Correctness Test ==="
echo "Server: $SERVER"
echo "Row:    $ROW"
echo ""

echo "--- Server info ---"
curl -s "http://$SERVER/api/info" | python3 -m json.tool 2>/dev/null || curl -s "http://$SERVER/api/info"
echo ""

echo "--- Direct fetch (cleartext, no privacy) ---"
curl -s "http://$SERVER/api/direct?row=$ROW" -o /tmp/pir_direct.bin
echo "Size: $(wc -c < /tmp/pir_direct.bin) bytes"
echo "Hex:  $(xxd /tmp/pir_direct.bin | head -1)"

echo ""
echo "--- PIR fetch (private) ---"
./target/release/pir-query --server "$SERVER" --row "$ROW" -o /tmp/pir_private.bin -v
echo "Hex:  $(xxd /tmp/pir_private.bin | head -1)"

echo ""
echo "--- Comparing ---"
if diff /tmp/pir_direct.bin /tmp/pir_private.bin > /dev/null 2>&1; then
    echo "PASS: PIR output matches direct output"
else
    MATCHING=$(cmp -l /tmp/pir_direct.bin /tmp/pir_private.bin 2>/dev/null | wc -l)
    TOTAL=$(wc -c < /tmp/pir_direct.bin)
    echo "FAIL: $MATCHING / $TOTAL bytes differ"
fi
