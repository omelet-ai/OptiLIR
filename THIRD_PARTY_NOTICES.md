# Third-party notices

This repository contains only code written for OptiLIR and bundles no third-party source code. The
packages below are installed separately by uv or pip and remain under their own licenses. Versions are
the ones pinned in `uv.lock`.

| Package | Version | License | Needed for |
|---|---|---|---|
| pydantic | 2.13.4 | MIT | the package |
| torch | 2.9.1 | BSD-3-Clause | `train`, `eval` |
| transformers | 5.16.1 | Apache-2.0 | `train`, `eval` |
| accelerate | 1.14.0 | Apache-2.0 | `train`, `eval` |
| peft | 0.20.0 | Apache-2.0 | `train`, `eval` |
| trl | 1.12.0 | Apache-2.0 | `train` |
| datasets | 5.0.1 | Apache-2.0 | `train`, `eval` |
| xgrammar | 0.2.3 | Apache-2.0 | `eval` |
| pytest | 9.1.1 | MIT | `test` |

Their own dependencies are pinned in `uv.lock`. On Linux, torch pulls in NVIDIA CUDA runtime libraries
(`nvidia-*-cu12`), most of which are under NVIDIA's proprietary license. The SBOM that the
`oss-security-check` workflow builds on every run lists every component.

## Models and data

The adapters on Hugging Face are LoRA weights. Each adapter runs on its base model, and the base model's
license applies as well:

| Adapter | Base model | Base model license |
|---|---|---|
| [`omelet-ai/OptiLIR-Qwen3-8B`](https://huggingface.co/omelet-ai/OptiLIR-Qwen3-8B) | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | Apache-2.0 |
| [`omelet-ai/OptiLIR-Gemma-4-12B`](https://huggingface.co/omelet-ai/OptiLIR-Gemma-4-12B) | [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it) | Apache-2.0 |

The dataset [`omelet-ai/OptiLIR-Instruct-v1`](https://huggingface.co/datasets/omelet-ai/OptiLIR-Instruct-v1)
is under CC BY 4.0.
