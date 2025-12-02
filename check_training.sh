#!/bin/bash
# Script to check if training is complete

cd "/Users/sinahaghgoo/Library/CloudStorage/OneDrive-Persönlich/Solution Deployment/Project"

# Check if process is running
if ps aux | grep -v grep | grep -q "train_model.py"; then
    echo "Training is still running..."
    echo "Current progress:"
    tail -5 training.log | grep -E "Epoch|Train|Val|Test|complete|Complete" || tail -3 training.log
else
    echo "✓ Training process has finished!"
    echo ""
    echo "Final results:"
    tail -20 training.log | grep -A 10 "Test Results\|Training complete\|Training Completed"
    echo ""
    echo "Model saved at: models/best_model.pth"
    ls -lh models/best_model.pth 2>/dev/null || echo "Model file not found"
fi

