import os
import sys
ROOT = os.path.dirname(__file__)
sys.path.append(ROOT)

import shutil
from constant import FRAMEWORK_INFO as FI
import yaml


template_code_dir = os.path.join(ROOT, "../template/code")

def ignore_patterns(path, names):
  """Custom ignore function to exclude specific patterns."""
  ignore_list = {"__pycache__", "*.pyc", "*.pyo", "*.swp"}
  return {name for name in names if any(p in name for p in ignore_list)}

def build_model_upload(
  inference_framework: str,
  custom_framework_kwargs: dict,
  config_yaml_data:dict,
  output_dir:str,
  clarifai_threads:str=16,
):
  assert inference_framework in FI, ValueError()
  os.makedirs(output_dir, exist_ok=True)
  #print(f"Copying from {template_code_dir} to {output_dir}")
  shutil.copytree(template_code_dir, output_dir, ignore=ignore_patterns, dirs_exist_ok=True)
  model_py = os.path.join(output_dir, "1/model.py")
  req_txt = os.path.join(output_dir, "requirements.txt")
  cf_yaml = os.path.join(output_dir, "config.yaml")
  dockerfile = os.path.join(output_dir, "Dockerfile")
  framework_info = FI[inference_framework]
  
  
  with open(cf_yaml, "w") as file:
    yaml.dump(config_yaml_data, file, default_flow_style=False, sort_keys=False)

  with open(req_txt, "r") as f:
    req_txt_default = f.read()
  with open(req_txt, "w") as f:
    add_frwk_req = "\n".join(framework_info["requirements"])
    req_txt_default = req_txt_default.replace("$FRAMEWORK_REQUIREMENTS", add_frwk_req)
    f.write(req_txt_default)
  
  with open(dockerfile, "r") as f:
    dockerfile_default = f.read()
  with open(dockerfile, "w") as f:
    dockerfile_default = dockerfile_default.replace("${CLARIFAI_NUM_THREADS}", str(clarifai_threads))
    download_cmd_docker = """#RUN ["python", "-m", "clarifai.cli", "model", "download-checkpoints", """
    dockerfile_default = dockerfile_default.replace(download_cmd_docker, download_cmd_docker[1:])
    f.write(dockerfile_default)
  
  with open(model_py, "r") as f:
    model_py_default = f.read()
  with open(model_py, "w") as f:
    model_py_default = model_py_default.replace("$INIT_SERVER", framework_info["init"])
    model_py_default = model_py_default.replace(
        "$SERVER_ARGS", str(custom_framework_kwargs))
    f.write(model_py_default)

if __name__ == "__main__":
  
  default_config = {
    "model": {
        "id": "openai-qwen2vl-2b-mpo",
        "user_id": "phatvo",
        "app_id": "text-generation",
        "model_type_id": "text-to-text"
    },
    "build_info": {
        "python_version": "3.11"
    },
    "inference_compute_info": {
        "cpu_limit": "4",
        "cpu_memory": "16Gi",
        "num_accelerators": 1,
        "accelerator_type": ["NVIDIA-A10G"],
        "accelerator_memory": "24Gi"
    }
  }
  build_model_upload(
    inference_framework="sglang",
    custom_framework_kwargs=dict(
      additional_list_args=["--enable-prefix-caching"],
      chat_template="qwen",
      checkpoints="sadfsad",
    ),
    config_yaml_data=default_config,
    output_dir="tmp/test_build"
  )