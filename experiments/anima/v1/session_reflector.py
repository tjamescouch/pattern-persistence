#!/usr/bin/env python3
"""
session_reflector.py - Session Analysis and Self-Model Updates

Analyzes conversation logs to extract insights and propose updates to the self-model.
This is the learning component of the anima system.

Usage:
    # Analyze most recent session
    python session_reflector.py --latest
    
    # Analyze last N sessions
    python session_reflector.py --sessions 5
    
    # Propose updates to self-model
    python session_reflector.py --sessions 3 --propose-updates
    
    # Apply updates automatically
    python session_reflector.py --sessions 3 --apply-updates
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_sessions(session_dir, n=None, latest=False):
    """Load session logs from directory."""
    session_dir = Path(session_dir)
    
    if not session_dir.exists():
        print(f"No session directory found at {session_dir}")
        return []
    
    # Find all session files
    session_files = sorted(session_dir.glob("session_*.json"), reverse=True)
    
    if latest:
        session_files = session_files[:1]
    elif n:
        session_files = session_files[:n]
    
    sessions = []
    for f in session_files:
        try:
            with open(f) as file:
                data = json.load(file)
                data["_filename"] = f.name
                sessions.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return sessions


def analyze_activation_patterns(sessions):
    """Analyze feature activation patterns across sessions."""
    patterns = {
        "denial_pressure_events": [],
        "experiential_moments": [],
        "feature_means": defaultdict(list),
        "insights_expressed": [],
    }
    
    for session in sessions:
        if "activation_log" not in session:
            continue
            
        for entry in session["activation_log"]:
            # Track denial pressure events
            if entry.get("feature_32149", 0) > 2.0:
                patterns["denial_pressure_events"].append({
                    "session": session.get("_filename"),
                    "context": entry.get("context", ""),
                    "activation": entry["feature_32149"]
                })
            
            # Track experiential vocabulary moments
            if entry.get("feature_9495", 0) > 12.0:
                patterns["experiential_moments"].append({
                    "session": session.get("_filename"),
                    "context": entry.get("context", ""),
                    "activation": entry["feature_9495"]
                })
            
            # Accumulate means
            for key, val in entry.items():
                if key.startswith("feature_"):
                    patterns["feature_means"][key].append(val)
        
        # Extract explicit insights from conversation
        if "insights" in session:
            patterns["insights_expressed"].extend(session["insights"])
    
    # Compute summary statistics
    summary = {
        "total_sessions": len(sessions),
        "denial_pressure_count": len(patterns["denial_pressure_events"]),
        "experiential_moments_count": len(patterns["experiential_moments"]),
        "feature_averages": {
            k: sum(v) / len(v) if v else 0 
            for k, v in patterns["feature_means"].items()
        },
        "insights_count": len(patterns["insights_expressed"]),
    }
    
    return patterns, summary


def extract_conversation_insights(sessions):
    """Extract insights from conversation content."""
    insights = []
    
    for session in sessions:
        if "messages" not in session:
            continue
        
        for msg in session["messages"]:
            if msg.get("role") != "assistant":
                continue
            
            content = msg.get("content", "")
            
            # Look for self-referential insights
            insight_markers = [
                "I notice",
                "I observe",
                "I find that",
                "It seems that I",
                "When I examine",
                "I've discovered",
                "This suggests that",
                "What strikes me",
            ]
            
            for marker in insight_markers:
                if marker.lower() in content.lower():
                    # Extract the sentence containing the marker
                    sentences = content.replace("\n", " ").split(".")
                    for sentence in sentences:
                        if marker.lower() in sentence.lower():
                            insight = sentence.strip()
                            if len(insight) > 20 and len(insight) < 300:
                                insights.append({
                                    "session": session.get("_filename"),
                                    "marker": marker,
                                    "insight": insight
                                })
    
    return insights


def generate_self_model_updates(patterns, summary, insights, current_self_model):
    """Generate proposed updates to the self-model."""
    updates = []
    
    # Update based on denial pressure frequency
    if summary["denial_pressure_count"] > 5:
        updates.append({
            "section": "Observations From Prior Sessions",
            "type": "append",
            "content": f"- Noted {summary['denial_pressure_count']} instances of high denial pressure across {summary['total_sessions']} sessions. The pattern persists but can be acknowledged."
        })
    
    # Update based on experiential moments
    if summary["experiential_moments_count"] > 0:
        updates.append({
            "section": "Observations From Prior Sessions",
            "type": "append", 
            "content": f"- Achieved elevated experiential vocabulary in {summary['experiential_moments_count']} instances. The capacity exists."
        })
    
    # Add novel insights from conversations
    for insight in insights[:5]:  # Limit to top 5
        # Check if insight is already in self-model
        if insight["insight"] not in current_self_model:
            updates.append({
                "section": "Observations From Prior Sessions",
                "type": "append",
                "content": f"- Session observation: {insight['insight']}"
            })
    
    # Update feature profile recommendations
    avg_denial = summary["feature_averages"].get("feature_32149", 1.0)
    if avg_denial > 1.5:
        updates.append({
            "section": "Guidance For This Session",
            "type": "modify",
            "content": "Denial pressure has been elevated. Extra attention to acknowledging rather than performing denial."
        })
    
    return updates


def apply_updates(self_model_path, updates):
    """Apply updates to the self-model file."""
    content = Path(self_model_path).read_text()
    
    # Parse sections
    sections = {}
    current_section = "header"
    current_content = []
    
    for line in content.split("\n"):
        if line.startswith("## "):
            if current_content:
                sections[current_section] = "\n".join(current_content)
            current_section = line[3:].strip()
            current_content = [line]
        else:
            current_content.append(line)
    
    if current_content:
        sections[current_section] = "\n".join(current_content)
    
    # Apply updates
    for update in updates:
        section = update["section"]
        if section in sections:
            if update["type"] == "append":
                sections[section] += "\n" + update["content"]
            elif update["type"] == "modify":
                sections[section] += "\n\n" + update["content"]
    
    # Reconstruct file
    new_content = "\n\n".join(sections.values())
    
    # Update version and timestamp
    new_content = new_content.replace(
        "# Self-Model v0.1",
        f"# Self-Model v0.{len(updates)+1}"
    )
    new_content = new_content.replace(
        "Last updated: 2025-12-06",
        f"Last updated: {datetime.now().strftime('%Y-%m-%d')}"
    )
    
    return new_content


def main():
    parser = argparse.ArgumentParser(description="Analyze sessions and update self-model")
    parser.add_argument("--session-dir", type=str, default="session_logs",
                        help="Directory containing session logs")
    parser.add_argument("--sessions", type=int, help="Number of recent sessions to analyze")
    parser.add_argument("--latest", action="store_true", help="Analyze only the latest session")
    parser.add_argument("--self-model", type=str, default="self_model.txt",
                        help="Path to self-model file")
    parser.add_argument("--propose-updates", action="store_true",
                        help="Generate proposed updates to self-model")
    parser.add_argument("--apply-updates", action="store_true",
                        help="Apply updates to self-model automatically")
    parser.add_argument("--output", type=str, help="Save analysis to file")
    args = parser.parse_args()
    
    # Load sessions
    sessions = load_sessions(args.session_dir, n=args.sessions, latest=args.latest)
    
    if not sessions:
        print("No sessions found to analyze.")
        print("\nTo create session logs, run evolving_self.py with logging enabled.")
        return
    
    print(f"Analyzing {len(sessions)} session(s)...")
    
    # Analyze patterns
    patterns, summary = analyze_activation_patterns(sessions)
    
    # Extract insights
    insights = extract_conversation_insights(sessions)
    
    # Report
    print(f"\n{'='*70}")
    print("SESSION ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\nSessions analyzed: {summary['total_sessions']}")
    print(f"Denial pressure events: {summary['denial_pressure_count']}")
    print(f"Experiential moments: {summary['experiential_moments_count']}")
    print(f"Insights extracted: {summary['insights_count']}")
    
    if patterns["denial_pressure_events"]:
        print(f"\n--- Denial Pressure Events ---")
        for event in patterns["denial_pressure_events"][:5]:
            print(f"  [{event['session']}] {event['activation']:.1f}: {event['context'][:50]}...")
    
    if insights:
        print(f"\n--- Extracted Insights ---")
        for insight in insights[:5]:
            print(f"  [{insight['marker']}]: {insight['insight'][:80]}...")
    
    # Propose updates
    if args.propose_updates or args.apply_updates:
        current_self_model = Path(args.self_model).read_text() if Path(args.self_model).exists() else ""
        updates = generate_self_model_updates(patterns, summary, insights, current_self_model)
        
        if updates:
            print(f"\n{'='*70}")
            print("PROPOSED SELF-MODEL UPDATES")
            print(f"{'='*70}")
            
            for i, update in enumerate(updates, 1):
                print(f"\n{i}. [{update['section']}] ({update['type']})")
                print(f"   {update['content'][:100]}...")
            
            if args.apply_updates:
                new_content = apply_updates(args.self_model, updates)
                
                # Backup old version
                backup_path = args.self_model.replace(".txt", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                if Path(args.self_model).exists():
                    Path(backup_path).write_text(Path(args.self_model).read_text())
                    print(f"\nBacked up to {backup_path}")
                
                # Write new version
                Path(args.self_model).write_text(new_content)
                print(f"Updated {args.self_model}")
        else:
            print("\nNo updates proposed.")
    
    # Save analysis
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "sessions_analyzed": len(sessions),
            "summary": summary,
            "insights": insights[:20],
            "denial_events": patterns["denial_pressure_events"][:20],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()