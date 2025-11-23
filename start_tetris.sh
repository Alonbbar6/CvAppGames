#!/bin/bash
# Tetris - Face/Body Control Launcher

echo "Starting Tetris - Face/Body Control..."
echo ""
echo "INSTRUCTIONS:"
echo "1. Look at camera with neutral face during calibration"
echo "2. MOVE HEAD left/right to move pieces"
echo "3. RAISE EYEBROWS to rotate pieces"
echo "4. Keyboard backup: arrows/WASD, Space=drop"
echo "5. Press ESC to quit"
echo ""

python3 tetris_body_control.py
