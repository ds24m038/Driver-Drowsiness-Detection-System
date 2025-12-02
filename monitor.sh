#!/bin/bash
cd "/Users/sinahaghgoo/Library/CloudStorage/OneDrive-Persönlich/Solution Deployment/Project"

echo "═══════════════════════════════════════════════════════"
echo "  Driver Drowsiness Detection System - Log Monitor"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""
echo "=== System Status ==="
docker compose ps
echo ""
echo "=== Following logs in real-time ==="
echo ""
docker compose logs -f
