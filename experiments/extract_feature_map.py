#!/usr/bin/env python3
"""
extract_feature_map.py - Extract clean feature map from unbiased analysis

Takes the output of feature_map_unbiased.py and produces a simple
{concept: feature_id} mapping using the top condition-specific features.

Usage:
    python feature_map_unbiased.py --conditions consciousness_conditions.json --output feature_analysis.json
    python extract_feature_map.py --input feature_analysis.json --output feature_map_clean.json
"""

import json
import argparse
from collections import defaultdict


def extract_best_features(analysis_path, output_path, top_k=1):
    """Extract the best discriminating feature for each condition."""
    
    with open(analysis_path) as f:
        data = json.load(f)
    
    condition_specific = data.get("condition_specific", {})
    profiles = data.get("profiles", {})
    
    feature_map = {}
    feature_details = {}
    
    print("=" * 70)
    print("CONDITION-SPECIFIC FEATURES")
    print("=" * 70)
    
    for condition, features in condition_specific.items():
        if not features:
            print(f"\n[{condition}] - No specific features found")
            continue
        
        print(f"\n[{condition}]")
        
        # Take top feature(s)
        for i, feat in enumerate(features[:top_k]):
            feat_id = feat["feature"]
            activation = feat["activation"]
            ratio = feat["ratio"]
            
            print(f"  Feature {feat_id}: activation={activation:.2f}, ratio={ratio:.1f}x")
            
            # Use condition name as key (clean it up)
            key = condition.replace("_", " ")
            
            if top_k == 1:
                feature_map[key] = feat_id
            else:
                feature_map[f"{key}_{i+1}"] = feat_id
            
            feature_details[str(feat_id)] = {
                "condition": condition,
                "activation": activation,
                "ratio": ratio,
                "rank": i + 1
            }
    
    # Also extract top discriminating features (high variance across conditions)
    print("\n" + "=" * 70)
    print("TOP DISCRIMINATING FEATURES (high variance)")
    print("=" * 70)
    
    discriminating = data.get("discriminating_features", [])
    
    for i, feat_data in enumerate(discriminating[:10]):
        feat_id = feat_data["feature"]
        variance = feat_data["variance"]
        max_cond = feat_data["max_condition"]
        min_cond = feat_data["min_condition"]
        
        print(f"  {i+1}. Feature {feat_id}: var={variance:.2f}, high in '{max_cond}', low in '{min_cond}'")
    
    # Check for collisions (same feature mapped to multiple conditions)
    print("\n" + "=" * 70)
    print("COLLISION CHECK")
    print("=" * 70)
    
    feat_to_conditions = defaultdict(list)
    for condition, feat_id in feature_map.items():
        feat_to_conditions[feat_id].append(condition)
    
    collisions = {k: v for k, v in feat_to_conditions.items() if len(v) > 1}
    
    if collisions:
        print("\n⚠️  COLLISIONS DETECTED (same feature, multiple conditions):")
        for feat_id, conditions in collisions.items():
            print(f"  Feature {feat_id}: {conditions}")
        print("\n  This may indicate these conditions aren't well-separated in activation space.")
    else:
        print("\n✓ No collisions - each condition maps to a unique feature")
    
    # Save
    output_data = {
        "feature_map": feature_map,
        "feature_details": feature_details,
        "collisions": {str(k): v for k, v in collisions.items()},
        "source": analysis_path
    }
    
    # Also save simple format for use with other scripts
    with open(output_path, "w") as f:
        json.dump(feature_map, f, indent=2)
    
    detail_path = output_path.replace(".json", "_details.json")
    with open(detail_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n\nSaved feature map to {output_path}")
    print(f"Saved details to {detail_path}")
    
    return feature_map


def main():
    parser = argparse.ArgumentParser(description="Extract feature map from unbiased analysis")
    parser.add_argument("--input", type=str, default="feature_analysis.json",
                        help="Input from feature_map_unbiased.py")
    parser.add_argument("--output", type=str, default="feature_map_clean.json",
                        help="Output feature map")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Number of features per condition")
    args = parser.parse_args()
    
    extract_best_features(args.input, args.output, args.top_k)


if __name__ == "__main__":
    main()
