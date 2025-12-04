LLMRI_MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0" LLMRI_DEVICE="mps" LLMRI_LAYER_INDEX=10 LLMRI_SAE_CHECKPOINT="features/l10_mlp_sae_shakespeare_2k.pt" LLMRI_FEATURES_FILE="features/l10_mlp_sae_shakespeare_2k_features.json" uvicorn llmri.server:app --reload

