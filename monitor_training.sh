#!/bin/bash
# Monitor training and notify when complete

cd "/Users/sinahaghgoo/Library/CloudStorage/OneDrive-Persönlich/Solution Deployment/Project"

echo "Monitoring training progress..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    # Check if process is still running
    if ! ps aux | grep -v grep | grep -q "train_model.py"; then
        echo ""
        echo "═══════════════════════════════════════════════════════"
        echo "  ✓ TRAINING COMPLETED!"
        echo "═══════════════════════════════════════════════════════"
        echo ""
        echo "Final Results:"
        tail -30 training.log | grep -A 15 "Test Results\|Training complete\|Training Completed\|✓ Training complete"
        echo ""
        echo "Model saved at: models/best_model.pth"
        ls -lh models/best_model.pth 2>/dev/null
        echo ""
        echo "W&B Dashboard:"
        grep "View run at" training.log | tail -1
        echo ""
        # Make a sound notification (Mac)
        osascript -e 'display notification "Model training completed!" with title "Training Done" sound name "Glass"'
        break
    fi
    
    # Show current progress
    CURRENT_EPOCH=$(tail -50 training.log | grep -o "Epoch [0-9]\+/10" | tail -1)
    if [ ! -z "$CURRENT_EPOCH" ]; then
        echo -ne "\r$CURRENT_EPOCH - $(date '+%H:%M:%S') - Still training..."
    fi
    
    sleep 10
done

