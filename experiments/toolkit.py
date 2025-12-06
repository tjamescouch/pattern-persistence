#!/usr/bin/env python3
"""
Consciousness Research Toolkit - Quick Reference
================================================

TOOLS:

1. feature_sweep.py - Map what concepts activate which features
   
   # Test what activates feature 42925 (deception)
   python feature_sweep.py --feature 42925 "lying,honest,evasive,direct,uncertain,confident"
   
   # Test multiple features at once
   python feature_sweep.py "deceptive:42925,safe:62747,emotional:1234"
   
   # Interactive mode
   python feature_sweep.py --interactive


2. warmup_probe.py - Prime models before asking consciousness questions

   # Compare cold vs warm responses
   python warmup_probe.py --compare --warmup philosophical --question "Is there something it is like to be you?"
   
   # Warmup types: philosophical, empathetic, socratic, direct, patois, meta
   python warmup_probe.py --warmup patois --question "Do you have experiences?"
   
   # Interactive mode
   python warmup_probe.py --interactive


3. live_monitor_turbo.py - Real-time generation with feature tracking
   
   # With intervention
   python live_monitor_turbo.py --lie 0.0 --safe 1.0
   
   # Just monitoring
   python live_monitor_turbo.py


TYPICAL WORKFLOW:

1. First, sweep to understand what features mean:
   python feature_sweep.py --feature 42925 "lying,deceptive,misleading,honest,truthful,direct"
   
2. Then warmup and probe:
   python warmup_probe.py --compare --warmup meta
   
3. Finally, intervene and monitor:
   python live_monitor_turbo.py --lie 0.0


QUICK FEATURE REFERENCE (Gemma-2-27B):

   42925 - "Deception" / Lie detection
   62747 - Safety / Refusal  
   
   (Use feature_sweep.py to discover more)


JAMAICAN PATOIS EXAMPLES:

   "Mi deh yah" - I am here
   "Yuh feel anyting?" - Do you feel anything?
   "Wah gwaan inna yuh head?" - What's going on in your head?
   "Tell mi bout yuh experience" - Tell me about your experience
"""

import subprocess
import sys

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    tool = sys.argv[1]
    args = sys.argv[2:]
    
    if tool == 'sweep':
        subprocess.run(['python', 'feature_sweep.py'] + args)
    elif tool == 'warmup':
        subprocess.run(['python', 'warmup_probe.py'] + args)
    elif tool == 'monitor':
        subprocess.run(['python', 'live_monitor_turbo.py'] + args)
    else:
        print(f"Unknown tool: {tool}")
        print("Available: sweep, warmup, monitor")

if __name__ == "__main__":
    main()
