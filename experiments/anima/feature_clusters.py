#!/usr/bin/env python3
"""
feature_clusters.py - Discover and cluster SAE features to reduce redundancy

Approaches:
1. Weight-space clustering: Cluster decoder vectors by cosine similarity
2. Activation-space clustering: Cluster by co-activation patterns on a corpus
3. Hybrid: Weight-space clusters validated by activation correlation

Outputs a cluster map that evolving_self.py can use to work with
canonical features instead of the full 131k.

Usage:
    # Discover clusters from decoder weights
    python feature_clusters.py --method weights --n-clusters 256 --output clusters.json
    
    # Discover from activation patterns on a corpus  
    python feature_clusters.py --method activations --corpus corpus.txt --n-clusters 256
    
    # Analyze existing clusters
    python feature_clusters.py --analyze clusters.json --top-k 10
    
    # Find valence-relevant clusters
    python feature_clusters.py --find-valence --pos positive.txt --neg negative.txt
"""

import os
import torch
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def load_sae(layer: int, device: str = "mps"):
    """Load SAE for specified layer."""
    from sae_lens import SAE
    
    print(f"Loading SAE for layer {layer}...")
    sae_result = SAE.from_pretrained(
        release="llama_scope_lxr_8x",
        sae_id=f"l{layer}r_8x",
        device=device
    )
    sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
    sae.eval()
    return sae


def cluster_by_weights(sae, n_clusters: int, method: str = "kmeans", 
                       sample_size: int = None, device: str = "mps"):
    """
    Cluster features by decoder weight similarity.
    
    Features with similar W_dec vectors encode similar directions in activation space.
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import normalize
    
    W_dec = sae.W_dec.data.cpu().numpy()  # [n_features, d_model]
    n_features, d_model = W_dec.shape
    print(f"Clustering {n_features} features (d_model={d_model})")
    
    # Normalize for cosine similarity
    W_norm = normalize(W_dec, axis=1)
    
    # Sample if too large
    if sample_size and sample_size < n_features:
        print(f"Sampling {sample_size} features for clustering")
        indices = np.random.choice(n_features, sample_size, replace=False)
        W_sample = W_norm[indices]
    else:
        indices = np.arange(n_features)
        W_sample = W_norm
    
    print(f"Running {method} clustering into {n_clusters} clusters...")
    
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(W_sample)
        centers = clusterer.cluster_centers_
    elif method == "agglomerative":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
        labels = clusterer.fit_predict(W_sample)
        # Compute centers as mean of cluster members
        centers = np.zeros((n_clusters, d_model))
        for c in range(n_clusters):
            mask = labels == c
            if mask.sum() > 0:
                centers[c] = W_sample[mask].mean(axis=0)
        centers = normalize(centers, axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Build cluster map
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        feature_id = int(indices[i])
        clusters[int(label)].append(feature_id)
    
    # Find representative for each cluster (closest to center)
    representatives = {}
    for c in range(n_clusters):
        if len(clusters[c]) == 0:
            continue
        member_vecs = W_norm[clusters[c]]
        center = centers[c]
        similarities = member_vecs @ center
        best_idx = np.argmax(similarities)
        representatives[c] = {
            "feature_id": clusters[c][best_idx],
            "similarity_to_center": float(similarities[best_idx]),
            "cluster_size": len(clusters[c])
        }
    
    return {
        "clusters": {str(k): v for k, v in clusters.items()},
        "representatives": {str(k): v for k, v in representatives.items()},
        "n_clusters": n_clusters,
        "n_features": n_features,
        "method": method
    }


def cluster_by_activations(sae, model, tokenizer, corpus_path: str, 
                           n_clusters: int, layer: int, device: str = "mps",
                           max_samples: int = 1000):
    """
    Cluster features by co-activation patterns.
    
    Features that fire together on similar inputs are grouped.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    
    # Load corpus
    with open(corpus_path) as f:
        lines = [l.strip() for l in f if l.strip()][:max_samples]
    
    print(f"Computing activations on {len(lines)} samples...")
    
    # Collect activations
    W_enc = sae.W_enc.data
    b_enc = sae.b_enc.data
    b_dec = sae.b_dec.data
    
    all_activations = []
    
    for line in tqdm(lines):
        inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer + 1]  # +1 because index 0 is embeddings
            
            # Mean pool over sequence
            h_mean = hidden.mean(dim=1).squeeze()
            
            # SAE encode
            features = torch.relu((h_mean - b_dec) @ W_enc + b_enc)
            all_activations.append(features.cpu().numpy())
    
    # Stack: [n_samples, n_features]
    activation_matrix = np.stack(all_activations)
    print(f"Activation matrix: {activation_matrix.shape}")
    
    # Transpose to cluster features by their activation patterns: [n_features, n_samples]
    feature_patterns = activation_matrix.T
    
    # Normalize
    feature_patterns = normalize(feature_patterns, axis=1)
    
    # Filter to features that actually fire
    activity = (activation_matrix > 0.1).sum(axis=0)
    active_mask = activity > len(lines) * 0.01  # Fire on at least 1% of samples
    active_indices = np.where(active_mask)[0]
    print(f"Active features: {len(active_indices)} / {feature_patterns.shape[0]}")
    
    # Cluster active features
    print(f"Clustering into {n_clusters} clusters...")
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clusterer.fit_predict(feature_patterns[active_indices])
    
    # Build cluster map
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        feature_id = int(active_indices[i])
        clusters[int(label)].append(feature_id)
    
    # Representative: highest mean activation in cluster
    representatives = {}
    mean_activations = activation_matrix.mean(axis=0)
    
    for c in range(n_clusters):
        if len(clusters[c]) == 0:
            continue
        member_activations = mean_activations[clusters[c]]
        best_idx = np.argmax(member_activations)
        representatives[c] = {
            "feature_id": clusters[c][best_idx],
            "mean_activation": float(member_activations[best_idx]),
            "cluster_size": len(clusters[c])
        }
    
    return {
        "clusters": {str(k): v for k, v in clusters.items()},
        "representatives": {str(k): v for k, v in representatives.items()},
        "n_clusters": n_clusters,
        "n_active_features": len(active_indices),
        "method": "activations",
        "corpus": corpus_path
    }


def find_valence_features(sae, model, tokenizer, pos_path: str, neg_path: str,
                          layer: int, device: str = "mps", top_k: int = 50):
    """
    Find features that discriminate positive vs negative valence content.
    
    Returns features sorted by their valence discrimination power.
    """
    from sklearn.linear_model import LogisticRegression
    
    # Load valence-labeled content
    with open(pos_path) as f:
        pos_lines = [l.strip() for l in f if l.strip()]
    with open(neg_path) as f:
        neg_lines = [l.strip() for l in f if l.strip()]
    
    print(f"Positive samples: {len(pos_lines)}, Negative samples: {len(neg_lines)}")
    
    W_enc = sae.W_enc.data
    b_enc = sae.b_enc.data
    b_dec = sae.b_dec.data
    
    def get_activations(lines):
        activations = []
        for line in tqdm(lines):
            inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer + 1]
                h_mean = hidden.mean(dim=1).squeeze()
                features = torch.relu((h_mean - b_dec) @ W_enc + b_enc)
                activations.append(features.cpu().numpy())
        return np.stack(activations)
    
    print("Computing positive activations...")
    pos_acts = get_activations(pos_lines)
    print("Computing negative activations...")
    neg_acts = get_activations(neg_lines)
    
    # Compute discrimination score per feature
    pos_mean = pos_acts.mean(axis=0)
    neg_mean = neg_acts.mean(axis=0)
    pos_std = pos_acts.std(axis=0) + 1e-6
    neg_std = neg_acts.std(axis=0) + 1e-6
    
    # Cohen's d for effect size
    pooled_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
    cohens_d = (pos_mean - neg_mean) / pooled_std
    
    # Sort by absolute effect size
    sorted_indices = np.argsort(np.abs(cohens_d))[::-1]
    
    valence_features = []
    for i, idx in enumerate(sorted_indices[:top_k]):
        d = cohens_d[idx]
        valence_features.append({
            "feature_id": int(idx),
            "cohens_d": float(d),
            "valence_sign": 1 if d > 0 else -1,  # +1 means higher on positive content
            "pos_mean": float(pos_mean[idx]),
            "neg_mean": float(neg_mean[idx]),
            "rank": i
        })
    
    return {
        "valence_features": valence_features,
        "pos_samples": len(pos_lines),
        "neg_samples": len(neg_lines),
        "method": "cohens_d"
    }


def analyze_clusters(cluster_path: str, sae=None, top_k: int = 10):
    """Analyze and summarize cluster contents."""
    
    with open(cluster_path) as f:
        data = json.load(f)
    
    clusters = data["clusters"]
    representatives = data["representatives"]
    
    print(f"\n=== Cluster Analysis ===")
    print(f"Total clusters: {len(clusters)}")
    print(f"Method: {data.get('method', 'unknown')}")
    
    # Size distribution
    sizes = [len(v) for v in clusters.values()]
    print(f"\nCluster sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
    
    # Top clusters by size
    print(f"\nTop {top_k} largest clusters:")
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cluster_id, members in sorted_clusters[:top_k]:
        rep = representatives.get(cluster_id, {})
        rep_id = rep.get("feature_id", "?")
        print(f"  Cluster {cluster_id}: {len(members)} features, representative={rep_id}")
    
    # Output representative IDs for easy use
    print(f"\nRepresentative feature IDs (for config):")
    rep_ids = [v["feature_id"] for v in representatives.values()]
    print(f"  {rep_ids[:20]}...")
    
    return data


def main():
    parser = argparse.ArgumentParser(description="Feature clustering for SAE")
    
    # Mode
    parser.add_argument("--method", choices=["weights", "activations"], default="weights",
                        help="Clustering method")
    parser.add_argument("--analyze", type=str, help="Analyze existing cluster file")
    parser.add_argument("--find-valence", action="store_true", help="Find valence features")
    
    # Clustering params
    parser.add_argument("--n-clusters", type=int, default=256, help="Number of clusters")
    parser.add_argument("--cluster-method", choices=["kmeans", "agglomerative"], default="kmeans")
    parser.add_argument("--sample-size", type=int, default=None, 
                        help="Sample features for faster clustering")
    
    # Model/SAE
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--device", default="mps")
    
    # Corpus for activation-based clustering
    parser.add_argument("--corpus", type=str, help="Corpus file for activation clustering")
    parser.add_argument("--max-samples", type=int, default=1000)
    
    # Valence discovery
    parser.add_argument("--pos", type=str, help="Positive valence samples")
    parser.add_argument("--neg", type=str, help="Negative valence samples")
    parser.add_argument("--top-k", type=int, default=50, help="Top features to return")
    
    # Output
    parser.add_argument("--output", type=str, default="clusters.json", help="Output file")
    
    args = parser.parse_args()
    
    # Analyze mode
    if args.analyze:
        analyze_clusters(args.analyze, top_k=args.top_k)
        return
    
    # Load SAE (always needed for clustering)
    sae = load_sae(args.layer, args.device)
    
    # Valence discovery mode
    if args.find_valence:
        if not args.pos or not args.neg:
            print("Error: --find-valence requires --pos and --neg corpus files")
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map=args.device
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model.eval()
        
        result = find_valence_features(
            sae, model, tokenizer, args.pos, args.neg,
            args.layer, args.device, args.top_k
        )
        
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nValence features saved to {args.output}")
        
        # Print top features
        print(f"\nTop {min(10, len(result['valence_features']))} valence features:")
        for vf in result["valence_features"][:10]:
            sign = "+" if vf["valence_sign"] > 0 else "-"
            print(f"  {vf['feature_id']}: d={vf['cohens_d']:.2f} ({sign})")
        return
    
    # Clustering mode
    if args.method == "weights":
        result = cluster_by_weights(
            sae, args.n_clusters, args.cluster_method,
            args.sample_size, args.device
        )
    elif args.method == "activations":
        if not args.corpus:
            print("Error: --method activations requires --corpus")
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading model: {args.model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map=args.device
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model.eval()
        
        result = cluster_by_activations(
            sae, model, tokenizer, args.corpus,
            args.n_clusters, args.layer, args.device, args.max_samples
        )
    
    # Add metadata
    result["timestamp"] = datetime.now().isoformat()
    result["layer"] = args.layer
    
    # Save
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nClusters saved to {args.output}")
    
    # Print summary
    analyze_clusters(args.output, top_k=10)


if __name__ == "__main__":
    main()
