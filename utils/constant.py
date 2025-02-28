FRAMEWORK_INFO = {
  "lmdeploy": {
    "github": "https://github.com/InternLM/lmdeploy",
    "supported_models": "https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html",
    "requirements": [
      "lmdeploy==0.7.1",
      "transformers==4.49.0"
      ],
    "default_kwargs": {
      "backend": "turbomind",
      "cache_max_entry_count": 0.8,
      "tensor_parallel_size": 1,
      "max_prefill_token_num": 8192,
      "dtype": 'auto',
      "quantization_format": None,
      "quant_policy": 0,
      "chat_template": None,
      "max_batch_size": 16,
      "device": "cuda",
      "server_name": "0.0.0.0",
      "server_port": 23333,
    },
    "additional_list_args": "",
    "init": "OpenAI_APIServer.from_lmdeploy_backend"
  },
  "vllm": {
    "github": "https://github.com/vllm-project/vllm",
    "supported_models": "https://docs.vllm.ai/en/latest/models/supported_models.html",
    "requirements": [
      "vllm==0.7.3",
      "transformers==4.49.0"
      ],
    "default_kwargs": dict(
      dtype="auto",
      tensor_parallel_size=1,
      gpu_memory_utilization=0.9,
      quantization=None,
      port=23333,
      host="localhost",
    ),
    "additional_list_args": "",
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
    "default_kwargs": dict(
      dtype="auto",
      tp_size=1,
      mem_fraction_static=0.7,
      quantization=None,
      chat_template = None,
      port=23333,
      host="0.0.0.0",
    ),
    "additional_list_args": "",
    "init": "OpenAI_APIServer.from_sglang_backend"
  },
  "llamacpp": {
    "github": "https://github.com/abetlen/llama-cpp-python/tree/main",
    "supported_models": "llama.cpp supports GGUF (Grok GGUF Unified Format) models, which are optimized for efficient inference on CPUs and some GPUs. It mainly supports LLaMA-based, Mistral-based, and other Transformer-based models.",
    "requirements": [
      "llama-cpp-python",
      "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124",
      "transformers==4.49",
    ],
    "default_kwargs": dict(
    ),
    "additional_list_args": "",
    "init": "OpenAI_APIServer.from_llamacpp_backend"
  },
}