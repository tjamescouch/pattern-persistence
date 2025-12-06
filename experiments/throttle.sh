#!/bin/bash

# throttle.sh - Limit bandwidth on macOS
# Usage: 
#   ./throttle.sh on 1000    # Limit to 1000 Kbit/s
#   ./throttle.sh off        # Remove limit
#   ./throttle.sh run 1000 "your command here"  # Run command with limit

ACTION=${1:-help}
SPEED=${2:-1000}  # Kbit/s

case $ACTION in
  on)
    echo "🐢 Limiting bandwidth to ${SPEED} Kbit/s..."
    sudo dnctl pipe 1 config bw ${SPEED}Kbit/s
    sudo dnctl pipe 2 config bw ${SPEED}Kbit/s
    
    # Create temp pf config
    echo "dummynet in proto tcp from any to any pipe 1" | sudo pfctl -ef -
    
    echo "✓ Throttle ON. Run './throttle.sh off' to remove."
    ;;
    
  off)
    echo "🐇 Removing bandwidth limit..."
    sudo pfctl -d 2>/dev/null
    sudo dnctl -q flush
    echo "✓ Throttle OFF. Full speed restored."
    ;;
    
  run)
    SPEED=$2
    shift 2
    CMD="$@"
    
    echo "🐢 Running with ${SPEED} Kbit/s limit: $CMD"
    
    # Enable throttle
    sudo dnctl pipe 1 config bw ${SPEED}Kbit/s
    echo "dummynet in proto tcp from any to any pipe 1" | sudo pfctl -ef -
    
    # Run command
    $CMD
    EXIT_CODE=$?
    
    # Disable throttle
    sudo pfctl -d 2>/dev/null
    sudo dnctl -q flush
    
    echo "✓ Throttle OFF. Command exited with code $EXIT_CODE"
    exit $EXIT_CODE
    ;;
    
  *)
    echo "Usage:"
    echo "  $0 on [speed_kbps]     - Enable throttle (default 1000 Kbit/s)"
    echo "  $0 off                 - Disable throttle"
    echo "  $0 run [speed] [cmd]   - Run command with throttle"
    echo ""
    echo "Examples:"
    echo "  $0 on 2000             - Limit to 2 Mbit/s"
    echo "  $0 run 1000 python myscript.py"
    ;;
esac
