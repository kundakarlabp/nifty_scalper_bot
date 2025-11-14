#!/bin/bash
# Production-grade validation script for PollingStreamer fixes

set -e

echo "====== PollingStreamer Production Validation ======"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Run Pytest
echo -e "${YELLOW}[Step 1/4]${NC} Running pytest..."
if pytest tests/streaming/test_polling_streamer.py -v --tb=short; then
    echo -e "${GREEN}✓ Pytest passed${NC}"
else
    echo -e "${RED}✗ Pytest failed${NC}"
    exit 1
fi
echo ""

# Step 2: Run Ruff Format Check
echo -e "${YELLOW}[Step 2/4]${NC} Running ruff format..."
if ruff format src/nifty_scalper_bot/streaming/polling_streamer.py; then
    echo -e "${GREEN}✓ Ruff format passed${NC}"
else
    echo -e "${RED}✗ Ruff format failed${NC}"
    exit 1
fi
echo ""

# Step 3: Run Ruff Lint Check
echo -e "${YELLOW}[Step 3/4]${NC} Running ruff check..."
if ruff check src/nifty_scalper_bot/streaming/polling_streamer.py; then
    echo -e "${GREEN}✓ Ruff check passed${NC}"
else
    echo -e "${RED}✗ Ruff check failed${NC}"
    exit 1
fi
echo ""

# Step 4: Verify file structure
echo -e "${YELLOW}[Step 4/4]${NC} Verifying file structure..."
if python -m py_compile src/nifty_scalper_bot/streaming/polling_streamer.py; then
    echo -e "${GREEN}✓ Python compilation passed${NC}"
else
    echo -e "${RED}✗ Python compilation failed${NC}"
    exit 1
fi
echo ""

echo -e "${GREEN}====== All Validations Passed ======${NC}"
echo ""
echo "Next steps:"
echo "  1. git add src/nifty_scalper_bot/streaming/polling_streamer.py tests/streaming/test_polling_streamer.py"
echo "  2. git commit -m 'Production hardening: PollingStreamer diagnostics, metrics, validation, and thread safety'"
echo "  3. git push origin fix/pollingstreamer-diagnostics"
echo "  4. Open PR on GitHub"
