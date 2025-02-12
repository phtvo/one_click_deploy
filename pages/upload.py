import os
####
import streamlit as st

from PIL import Image
from clarifai.client import Model, Inputs
from clarifai_grpc.grpc.api import resources_pb2
from utils.constant import FRAMEWORK_INFO as FI

def display():
  # Sidebar settings
  st.sidebar.header("Settings")
  with st.sidebar:
      base_url = st.text_input("Base URL", value="https://api.clarifai.com")
      pat = st.text_input("PAT", type="password")
      output_folder = st.text_input("Path to save generated model upload folder", "~/temp/",)
  if pat:
      os.environ["CLARIFAI_PAT"] = pat
  if not os.environ.get("CLARIFAI_PAT"):
      st.error("Please insert your PAT")
      st.stop()

  
  def make_model_id(hf_id):
    model_id = hf_id.split("/")[1]
    model_id = model_id.replace(".", "_")
    return model_id
    
  # Main input section
  upload_config_col, model_server_col = st.columns([3, 3])
  
  with upload_config_col:
    #
    st.markdown("#### Model Settings")
    hf_model_id = st.text_input("Insert HF model ID")
    valid_model_id = ""
    if hf_model_id:
      try:
        valid_model_id = make_model_id(hf_model_id)
      except:
        st.error("Invalid HF model id")
    
    model_id = st.text_input("Model ID", valid_model_id)
    user_id = st.text_input("User ID", "")
    app_id = st.text_input("App ID", "")
    model_type_id = st.selectbox("Select model type", ["text-to-text", "multimodal-to-text"])

    # Build info
    st.markdown("#### Build Info")
    python_version = st.selectbox("Python Version", ["3.11", "3.12"])

    # Inference compute info
    st.markdown("#### Inference Compute Info")
    cpu_limit = st.slider("CPU Limit", min_value=1, max_value=8, value=2)
    cpu_memory = st.slider("CPU Memory", min_value=1, max_value=32, value=4)
    cpu_memory = f"{cpu_memory}Gi"
    num_accelerators = st.slider("Num Accelerators", min_value=0, max_value=4, value=1)
    accelerator_type = st.selectbox("Accelerator Type", ["NVIDIA-A10G", "NVIDIA-L4", "NVIDIA-T4", "NVIDIA-L40S", "NVIDIA-A100", "NVIDIA-H100",])
    accelerator_memory = st.number_input("Accelerator Memory",  value=8)
    accelerator_memory = f"{accelerator_memory}Gi"

    # Checkpoint settings (optional)
    show_checkpoints = st.toggle("Enable cache checkpoints", value=True)
    if show_checkpoints:
        repo_id = hf_model_id
        hf_token = st.text_input("HuggingFace Token", "", type="password")
    else:
        repo_id, hf_token = None, None

    yaml_config = {
      "model": {
          "id": model_id,
          "user_id": user_id,
          "app_id": app_id,
          "model_type_id": model_type_id
      },
      "build_info": {
          "python_version": python_version
      },
      "inference_compute_info": {
          "cpu_limit": cpu_limit,
          "cpu_memory": cpu_memory,
          "num_accelerators": num_accelerators,
          "accelerator_type": accelerator_type.split(", "),
          "accelerator_memory": accelerator_memory
      }
    }
    if show_checkpoints:
      yaml_config["checkpoints"] = {
          "type": "huggingface",
          "repo_id": repo_id,
          "hf_token": hf_token
      }

  with model_server_col:
    st.markdown("### Inference Settings")
    infer_framework = st.selectbox("Select inference framework", FI)
  


if __name__ == "__main__":
  display()