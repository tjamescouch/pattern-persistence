# llmri — LLM MRI / Logic Viewer

`llmri` is a toolkit for inspecting large language models like a lab instrument:

- **Model MRI**: browse layers, heads, and MLPs; inspect weight norms and simple stats.
- **Activation tracing**: record intermediate activations for a corpus via CLI or Python API.
- **Feature / logic extraction** (planned): sparse autoencoders, probes, and interpretable features.
- **Interactive viewer** (planned): small web UI for browsing and ablation experiments.

## Install (editable dev)

```bash
git clone https://github.com/yourname/llmri.git
cd llmri
pip install -e ".[dev]"
