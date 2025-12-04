# llmri/cli/main.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..core import load_model
from ..trace import ActivationTracer
from ..utils.io import save_activation_trace, save_json, load_json
from .. import stats as stats_mod
from ..features.datasets import ActivationDataset
from ..features.sae import SAETrainConfig, train_sae, summarize_sae_features, SparseAutoencoder
from ..interventions import run_sae_feature_intervention


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llmri", description="LLM MRI / logic viewer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------ #
    # llmri model ...
    # ------------------------------------------------------------------ #
    p_model = subparsers.add_parser("model", help="Model-related commands")
    p_model_sub = p_model.add_subparsers(dest="model_cmd", required=True)

    p_model_info = p_model_sub.add_parser("info", help="Show model structure info")
    p_model_info.add_argument("--model", required=True, help="HF model id")
    p_model_info.add_argument("--device", default="cuda", help="Device string (cuda, cpu, mps)")

    # ------------------------------------------------------------------ #
    # llmri trace ...
    # ------------------------------------------------------------------ #
    p_trace = subparsers.add_parser("trace", help="Activation tracing commands")
    p_trace_sub = p_trace.add_subparsers(dest="trace_cmd", required=True)

    p_trace_acts = p_trace_sub.add_parser("activations", help="Trace activations for given text/corpus")
    p_trace_acts.add_argument("--model", required=True)
    p_trace_acts.add_argument("--device", default="cuda")
    p_trace_acts.add_argument("--layers", required=True, help="Comma-separated layer indices, e.g. 10,11")
    p_trace_acts.add_argument(
        "--components",
        default="mlp",
        help="Comma-separated components: mlp,attn (Phase 1)",
    )
    p_trace_acts.add_argument("--input-file", required=True, help="Text file with one example per line")
    p_trace_acts.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Optional cap on number of non-empty lines to use from input-file",
    )
    p_trace_acts.add_argument("--out", required=True, help="Output .pt file path")

    # ------------------------------------------------------------------ #
    # llmri stats ...
    # ------------------------------------------------------------------ #
    p_stats = subparsers.add_parser("stats", help="Statistics and summaries")
    p_stats_sub = p_stats.add_subparsers(dest="stats_cmd", required=True)

    p_stats_weights = p_stats_sub.add_parser("weights", help="Summarize model weights")
    p_stats_weights.add_argument("--model", required=True)
    p_stats_weights.add_argument("--device", default="cuda")
    p_stats_weights.add_argument("--out", required=True)

    p_stats_acts = p_stats_sub.add_parser("activations", help="Summarize saved activations")
    p_stats_acts.add_argument("--file", required=True)
    p_stats_acts.add_argument("--out", required=True)

    # ------------------------------------------------------------------ #
    # llmri features ...
    # ------------------------------------------------------------------ #
    p_feat = subparsers.add_parser("features", help="Feature / logic extraction tools")
    p_feat_sub = p_feat.add_subparsers(dest="feat_cmd", required=True)

    # Train SAE
    p_feat_train = p_feat_sub.add_parser("train-sae", help="Train a sparse autoencoder on activations")
    p_feat_train.add_argument("--model", required=True, help="HF model id (for metadata)")
    p_feat_train.add_argument("--layer", type=int, required=True)
    p_feat_train.add_argument("--component", default="mlp")
    p_feat_train.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="List of activation trace .pt files",
    )
    p_feat_train.add_argument("--hidden-dim", type=int, required=True)
    p_feat_train.add_argument("--batch-size", type=int, default=256)
    p_feat_train.add_argument("--lr", type=float, default=1e-3)
    p_feat_train.add_argument("--l1", type=float, default=1e-3)
    p_feat_train.add_argument("--epochs", type=int, default=5)
    p_feat_train.add_argument("--device", default="cuda")
    p_feat_train.add_argument("--max-tokens", type=int, default=None)
    p_feat_train.add_argument("--out-checkpoint", required=True)

    # Summarize features
    p_feat_sum = p_feat_sub.add_parser("summarize", help="Summarize SAE features (top tokens, stats)")
    p_feat_sum.add_argument("--model", required=True, help="HF model id (for tokenizer)")
    p_feat_sum.add_argument("--layer", type=int, required=True)
    p_feat_sum.add_argument("--component", default="mlp")
    p_feat_sum.add_argument(
        "--traces",
        nargs="+",
        required=True,
        help="List of activation trace .pt files",
    )
    p_feat_sum.add_argument("--sae-checkpoint", required=True)
    p_feat_sum.add_argument("--top-k", type=int, default=20)
    p_feat_sum.add_argument("--device", default="cuda")
    p_feat_sum.add_argument("--out", required=True)

    # Inspect features (pretty-print)
    p_feat_ins = p_feat_sub.add_parser("inspect", help="Pretty-print SAE features from a JSON file")
    p_feat_ins.add_argument("--features-file", required=True)
    p_feat_ins.add_argument("--num-features", type=int, default=10)
    p_feat_ins.add_argument(
        "--sort-by",
        choices=["id", "mean_activation", "sparsity"],
        default="mean_activation",
    )
    p_feat_ins.add_argument("--min-sparsity", type=float, default=0.0)
    p_feat_ins.add_argument("--max-sparsity", type=float, default=1.0)

    # Intervene on a feature during generation
    p_feat_int = p_feat_sub.add_parser(
        "intervene",
        help="Run generation with and without editing a single SAE feature",
    )
    p_feat_int.add_argument("--model", required=True, help="HF model id")
    p_feat_int.add_argument("--layer", type=int, required=True)
    p_feat_int.add_argument("--component", default="mlp")
    p_feat_int.add_argument("--sae-checkpoint", required=True)
    p_feat_int.add_argument("--feature-id", type=int, required=True)
    p_feat_int.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Scale for the feature (0.0=ablate, >1.0=boost)",
    )
    p_feat_int.add_argument("--prompt", required=True)
    p_feat_int.add_argument("--device", default="cuda")
    p_feat_int.add_argument("--max-new-tokens", type=int, default=40)

    # ------------------------------------------------------------------ #
    # llmri corpus ...
    # ------------------------------------------------------------------ #
    p_corpus = subparsers.add_parser("corpus", help="Download or prepare text corpora")
    p_corpus_sub = p_corpus.add_subparsers(dest="corpus_cmd", required=True)

    p_corpus_sh = p_corpus_sub.add_parser(
        "download-shakespeare",
        help="Download Tiny Shakespeare corpus to a local text file",
    )
    p_corpus_sh.add_argument(
        "--out",
        default="data/shakespeare_tiny.txt",
        help="Output text file (default: data/shakespeare_tiny.txt)",
    )

    args = parser.parse_args(argv)

    if args.command == "model":
        if args.model_cmd == "info":
            cmd_model_info(args)
    elif args.command == "trace":
        if args.trace_cmd == "activations":
            cmd_trace_activations(args)
    elif args.command == "stats":
        if args.stats_cmd == "weights":
            cmd_stats_weights(args)
        elif args.stats_cmd == "activations":
            cmd_stats_activations(args)
    elif args.command == "features":
        if args.feat_cmd == "train-sae":
            cmd_features_train_sae(args)
        elif args.feat_cmd == "summarize":
            cmd_features_summarize(args)
        elif args.feat_cmd == "inspect":
            cmd_features_inspect(args)
        elif args.feat_cmd == "intervene":
            cmd_features_intervene(args)
    elif args.command == "corpus":
        if args.corpus_cmd == "download-shakespeare":
            cmd_corpus_download_shakespeare(args)


def cmd_model_info(args) -> None:
    lm = load_model(args.model, device=args.device)
    info = lm.info

    print(f"Model: {info.model_id}")
    print(f"Params: {info.n_params:,}")
    print(f"Layers: {info.n_layers}, d_model={info.d_model}, vocab={info.vocab_size}")
    print("Layers and components:")
    for layer in info.layer_infos:
        comps = ", ".join(f"{c.component}@{c.name}" for c in layer.components)
        print(f"  Layer {layer.index}: {comps}")


def cmd_trace_activations(args) -> None:
    lm = load_model(args.model, device=args.device)
    model, tokenizer, info = lm.model, lm.tokenizer, lm.info

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    components = [x.strip() for x in args.components.split(",") if x.strip()]

    input_path = Path(args.input_file)
    texts: list[str] = []
    used = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            texts.append(line)
            used += 1
            if args.max_lines is not None and used >= args.max_lines:
                break

    if not texts:
        raise RuntimeError(f"No non-empty lines found in {input_path}")

    print(f"[trace] Using {len(texts)} lines from {input_path}")

    tracer = ActivationTracer(
        model=model,
        tokenizer=tokenizer,
        info=info,
        layers=layers,
        components=components,
        device=args.device,
    )
    trace = tracer.run_on_batch(texts)
    save_activation_trace(trace, args.out)
    print(f"[trace] Saved activation trace to {args.out}")


def cmd_stats_weights(args) -> None:
    lm = load_model(args.model, device=args.device)
    summary = stats_mod.summarize_weights(lm.model, lm.info)
    save_json(summary, args.out)
    print(f"Saved weights summary to {args.out}")


def cmd_stats_activations(args) -> None:
    from ..utils.io import load_activation_trace

    trace = load_activation_trace(args.file)
    summary = stats_mod.summarize_activations(trace)
    save_json(summary, args.out)
    print(f"Saved activation summary to {args.out}")


def cmd_features_train_sae(args) -> None:
    ds = ActivationDataset(
        trace_paths=args.traces,
        layer=args.layer,
        component=args.component,
        max_tokens=args.max_tokens,
    )

    cfg = SAETrainConfig(
        d_model=ds.d_model,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        l1=args.l1,
        epochs=args.epochs,
        device=args.device,
    )

    print(
        f"[SAE] Training on {len(ds)} tokens "
        f"(d_model={cfg.d_model}, hidden_dim={cfg.hidden_dim})"
    )
    model = train_sae(ds, cfg)

    out_path = Path(args.out_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import torch  # local

    torch.save(
        {"state_dict": model.state_dict(), "config": cfg.__dict__},
        out_path,
    )
    print(f"[SAE] Saved checkpoint to {out_path}")


def cmd_features_summarize(args) -> None:
    import torch  # local

    lm = load_model(args.model, device="cpu")
    tokenizer = lm.tokenizer

    ds = ActivationDataset(
        trace_paths=args.traces,
        layer=args.layer,
        component=args.component,
        max_tokens=None,
    )

    ckpt = torch.load(args.sae_checkpoint, map_location=args.device)
    cfg_dict = ckpt.get("config", {})
    d_model = ds.d_model
    hidden_dim = int(cfg_dict.get("hidden_dim", 0)) or ckpt["state_dict"]["encoder.weight"].shape[0]

    model = SparseAutoencoder(d_model=d_model, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["state_dict"])

    summary = summarize_sae_features(
        model=model,
        dataset=ds,
        tokenizer=tokenizer,
        top_k=args.top_k,
        device=args.device,
    )
    save_json(summary, args.out)
    print(f"[SAE] Saved feature summary to {args.out}")


def cmd_features_inspect(args) -> None:
    data = load_json(args.features_file)
    feats = data.get("features", [])

    feats = [
        f
        for f in feats
        if args.min_sparsity <= float(f.get("sparsity", 0.0)) <= args.max_sparsity
    ]

    key = args.sort_by

    def sort_key(f):
        if key == "id":
            return int(f.get("id", 0))
        return float(f.get(key, 0.0))

    feats.sort(key=sort_key, reverse=(key != "id"))

    print(
        f"[inspect] Showing up to {args.num_features} features "
        f"(sorted by {key}, sparsity in [{args.min_sparsity}, {args.max_sparsity}])"
    )

    for f in feats[: args.num_features]:
        fid = f.get("id", "?")
        mean_act = f.get("mean_activation", 0.0)
        sparsity = f.get("sparsity", 0.0)
        top_tokens = f.get("top_tokens", [])
        tok_strs = [t.get("token", "").replace("\n", "\\n") for t in top_tokens[:10]]

        print(f"\nFeature {fid}: mean={mean_act:.4f}, sparsity={sparsity:.3f}")
        print("  top tokens:", ", ".join(tok_strs) if tok_strs else "(none)")


def cmd_features_intervene(args) -> None:
    import torch  # local

    if args.component != "mlp":
        raise RuntimeError("features intervene currently only supports component='mlp'.")

    lm = load_model(args.model, device=args.device)
    model, tokenizer = lm.model, lm.tokenizer

    ckpt = torch.load(args.sae_checkpoint, map_location=args.device)
    state_dict = ckpt["state_dict"]
    encoder_weight = state_dict["encoder.weight"]
    d_model = encoder_weight.shape[1]
    hidden_dim = encoder_weight.shape[0]

    sae = SparseAutoencoder(d_model=d_model, hidden_dim=hidden_dim)
    sae.load_state_dict(state_dict)

    baseline, edited, dbg = run_sae_feature_intervention(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        layer_index=args.layer,
        feature_id=args.feature_id,
        scale=args.scale,
        prompt=args.prompt,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    print("=== Baseline ===")
    print(baseline)
    print("\n=== Edited (feature "
          f"{args.feature_id} @ layer {args.layer}, scale={args.scale}) ===")
    print(edited)

    if dbg:
        print("\n=== Debug metrics (first generated token) ===")
        print(f"L2 diff on logits: {dbg['first_logit_l2_diff']:.4f}")
        print(f"Cosine similarity: {dbg['first_logit_cosine']:.4f}")
        print(
            "Top token id baseline / edited: "
            f"{dbg['first_top_token_baseline']} / {dbg['first_top_token_edited']} "
            f"(changed={dbg['first_top_token_changed']})"
        )



def cmd_corpus_download_shakespeare(args) -> None:
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/"
        "master/data/tinyshakespeare/input.txt"
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[corpus] Downloading Tiny Shakespeare from {url}")
    with urllib.request.urlopen(url) as resp, out_path.open("wb") as f:
        f.write(resp.read())
    print(f"[corpus] Wrote corpus to {out_path}")


if __name__ == "__main__":
    main()