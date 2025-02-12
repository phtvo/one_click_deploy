FRAMEWORK_INFO = {
  "lmdeploy": {
    "github": "https://github.com/InternLM/lmdeploy",
    "supported_models": "https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html",
    "requirements": [
      "lmdeploy==0.7.0.post2"
      ],
    "default_kwargs": {
      "server_port": 23333,
      "backend": "turbomind",
      "cache_max_entry_count": 0.5,
      "tensor_parallel_size": 1,
      "max_prefill_token_num": 4096,
      "dtype": 'auto',
      "quantization_format": None,
      "quant_policy": 0,
      "chat_template": None,
      "max_batch_size": 16,
      "server_name": "0.0.0.0",
      "device": "cuda",
    },
    "additional_list_args": "",
    "init": "OpenAI_APIServer.from_lmdeploy_backend"
  },
  "vllm": {
    "github": "https://github.com/vllm-project/vllm",
    "supported_models": "https://docs.vllm.ai/en/latest/models/supported_models.html",
    "requirements": [
      "orjson==3.10.11",
      "python-multipart==0.0.17",
      "sgl-kernel",
      "--find-links https://flashinfer.ai/whl/cu124/torch2.5*/",
      "sglang[all]==0.4.2.post4",
    ],
    "default_kwargs": dict(
      dtype="auto",
      tensor_parallel_size=1,
      gpu_memory_utilization=0.8,
      port=23333,
      host="localhost",
      quantization=None,
    ),
    "additional_list_args": "",
    "init": "OpenAI_APIServer.from_vllm_backend"
  },
  "vllm": {
    "github": "https://github.com/sgl-project/sgl-project.github.io",
    "supported_models": "https://docs.sglang.ai/references/supported_models.html",
    "requirements": [
      "vllm==0.7.2"
      ],
    "default_kwargs": dict(
      dtype="auto",
      tp_size=1,
      mem_fraction_static=0.7,
      port=23333,
      host="0.0.0.0",
      quantization=None,
      chat_template = None,
    ),
    "additional_list_args": "",
    "init": "OpenAI_APIServer.from_sglang_backend"
  },
}