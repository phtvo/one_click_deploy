import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../template/code/1/"))

from openai_server_starter import OpenAI_APIServer
from utils.args_inspector import inspect_function_args

FRAMEWORK_INFO = {
  "lmdeploy": {
    "github": "https://github.com/InternLM/lmdeploy",
    "supported_models": "https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html",
    "requirements": [
      "lmdeploy==0.7.1",
      "transformers==4.49.0"
      ],
    "init_args": inspect_function_args(OpenAI_APIServer.from_lmdeploy_backend, drop_keys=["checkpoints"]),
    "init": "OpenAI_APIServer.from_lmdeploy_backend"
  },
  "vllm": {
    "github": "https://github.com/vllm-project/vllm",
    "supported_models": "https://docs.vllm.ai/en/latest/models/supported_models.html",
    "requirements": [
      "vllm==0.7.3",
      "transformers==4.49.0",
      "backoff==2.2.1",
      "peft==0.13.2",
      "soundfile==0.13.1",
      "scipy==1.15.2",
      "librosa",
      "decord"
      ],
    "init_args": inspect_function_args(OpenAI_APIServer.from_vllm_backend, drop_keys=["checkpoints"]),
    "init": "OpenAI_APIServer.from_vllm_backend"
  },
  "sglang": {
    "github": "https://github.com/sgl-project/sgl-project.github.io",
    "supported_models": "https://docs.sglang.ai/references/supported_models.html",
    "requirements": [
      "--find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python",
      "sglang[all]>=0.4.3.post2",
      "transformers==4.48.3",
    ],
    "init_args": inspect_function_args(OpenAI_APIServer.from_sglang_backend, drop_keys=["checkpoints"]),
    "init": "OpenAI_APIServer.from_sglang_backend"
  },
  "llamacpp": {
    "github": "https://github.com/ggml-org/llama.cpp/tree/master",
    "supported_models": "llama.cpp supports GGUF (Grok GGUF Unified Format) models, which are optimized for efficient inference on CPUs and some GPUs. It mainly supports LLaMA-based, Mistral-based, and other Transformer-based models.",
    "requirements": [],
    "init_args": inspect_function_args(OpenAI_APIServer.from_llamacpp_backend, drop_keys=["checkpoints"]),
    "init": "OpenAI_APIServer.from_llamacpp_backend"
  },
}