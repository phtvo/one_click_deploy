import os
import logging
import queue
import subprocess
import streamlit as st
st.set_page_config(layout="wide")

from PIL import Image
from clarifai.client import Model, Inputs
from clarifai_grpc.grpc.api import resources_pb2
from utils.constant import FRAMEWORK_INFO as FI
from utils.build import build_model_upload
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.utils.logging import logger
import shlex

def run_subprocess(command):
    env_vars = os.environ.copy()
    process = subprocess.Popen(
        command, 
        stdin=subprocess.PIPE,  # Enable input to process
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True,
        env=env_vars
    )

    process.stdin.write("\n")  # Send Enter key
    process.stdin.flush()  # Ensure the input is processed

    log_queue = queue.Queue(maxsize=30)
    st_log_title.markdown(f"Running command `{' '.join(command)}`, see its log below")
    for line in iter(process.stdout.readline, ""):
      print(line.strip())
      if log_queue.full():
        log_queue.get()
      log_queue.put(line.strip())
      log = "\n".join(list(log_queue.queue))
      st_log.code(log)
    process.stdout.close()
    process.wait()
    process.kill()
    if process.returncode != 0:  # Non-zero exit code = error
      error_msg = process.stderr.read().strip() or f"Command failed with exit code {process.returncode}"
      error_area.error(error_msg)
      st.stop()

st_log_title = st.empty()
st_log = st.empty()
error_area = st.empty()
  

def display():
  # Sidebar settings
  st.sidebar.header("Settings")
  with st.sidebar:
      base_url = st.text_input("Base URL", value="https://api.clarifai.com")
      pat = st.text_input("PAT", type="password")
      output_folder = st.text_input("Path to save generated model upload folder", "./tmp/",)
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
  upload_config_col, _, model_server_col, _,  control_col = st.columns([6, 1, 6, 1, 6])
  
  with upload_config_col:
    #
    st.markdown("#### Model Settings")
    st.text("(*) required")
    hf_model_id = st.text_input("HF model ID *")
    valid_model_id = ""
    if hf_model_id:
      try:
        valid_model_id = make_model_id(hf_model_id)
      except:
        st.error("Invalid HF model id")
    else:
      st.error("Please input HF model id")
    
    model_id = st.text_input("Model ID", value=valid_model_id, placeholder=valid_model_id)
    if not model_id:
      model_id = valid_model_id
    
    user_id = st.text_input("User ID *", "")
    if not user_id:
      st.error("Please input this user_id")
    
    app_id = st.text_input("App ID *", "")
    if not app_id:
      st.error("Please input this app_id")
    model_type_id = st.selectbox("Select model type", ["text-to-text", "multimodal-to-text"])

    # Build info
    st.markdown("#### Build Info")
    python_version = st.selectbox(
      "Python Version", 
        [
          "3.11", 
          # use fixed env
          #"3.12"
        ]
      )

    # Inference compute info
    st.markdown("#### Inference Compute Info")
    cpu_limit = st.slider("CPU Limit", min_value=1, max_value=8, value=2)
    cpu_memory = st.slider("CPU Memory", min_value=1, max_value=32, value=8)
    cpu_memory = f"{cpu_memory}Gi"
    num_accelerators = st.slider("Num Accelerators", min_value=0, max_value=4, value=1)
    accelerator_type = st.selectbox("Accelerator Type", ["NVIDIA-A10G", "NVIDIA-L4", "NVIDIA-T4", "NVIDIA-L40S", "NVIDIA-A100", "NVIDIA-H100",])
    accelerator_memory = st.number_input("Accelerator Memory",  value=24)
    accelerator_memory = f"{accelerator_memory}Gi"

    # Checkpoint settings (optional)
    download_checkpoints = st.toggle("Enable cache checkpoints", value=True)
    if download_checkpoints:
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
          "cpu_limit": str(cpu_limit),
          "cpu_memory": cpu_memory,
          "num_accelerators": num_accelerators,
          "accelerator_type": [accelerator_type],
          "accelerator_memory": accelerator_memory
      }
    }
    if download_checkpoints:
      yaml_config["checkpoints"] = {
          "type": "huggingface",
          "repo_id": repo_id,
          "hf_token": hf_token
      }
      
  with model_server_col:
    st.markdown("### Inference Settings")
    infer_framework = st.selectbox("Select inference framework", FI)
    fi = FI[infer_framework]
    st.markdown(
        f"""
        <a href="{fi['github']}" target="_blank">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30">
        </a>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f"[🤖 Supported models]({fi['supported_models']})", unsafe_allow_html=True)
    default_server_args = fi["default_kwargs"]
    with st.expander("#### Server args"):
      st.markdown("These args are used to start up OpenAI compatible server, please refer to framework page for details")
      custom_server_args = {}
      for arg_name, arg_value in default_server_args.items():
        if arg_value is None or isinstance(arg_value, str):
          custom_server_args.update({arg_name: st.text_input("--" + arg_name, value=arg_value)})
        elif isinstance(arg_value, int):
          custom_server_args.update({arg_name: st.number_input("--" + arg_name, value=arg_value)})
        elif isinstance(arg_value, bool):
          custom_server_args.update({arg_name: st.checkbox("--" + arg_name, value=arg_value)})
        elif isinstance(arg_value, float):
          custom_server_args.update({arg_name: st.number_input(
            "--" + arg_name, value=arg_value, min_value=0., max_value=1., step=0.05)})
      additional_args = st.text_area(
        "Additional args", help="Additional args (exclude all of above args) to run the server e.g. '--abc 123 --enable-something --float 0.4'")
    print(additional_args)
    if additional_args != "":
      custom_server_args.update(dict(additional_list_args=shlex.split(additional_args)))
    custom_server_args.update(dict(checkpoints="checkpoints" if download_checkpoints else hf_model_id))
    #st.code(custom_server_args)
  
  with control_col:
    st.markdown("### Run")
    generate_btn = st.button("Generate code only")
    upload_btn = st.button("Generate and Upload model")
    test_locally_btn = st.button("Test model locally")
    def validate_chat_template():
      """ Validate server args """
      if model_type_id == "multimodal-to-text" and infer_framework in ["lmdeploy", "sglang"]:
        if custom_server_args.get("chat_template") is None:
          st.error("You must input `chat_template` for this model type to use in lmdeploy or slang")
          st.stop()
      tmp_check = dict(hf_model_id=hf_model_id, user_id=user_id, app_id=app_id)
      for k, v in tmp_check.items():
        if not v:
          st.error(f"Please input {k} in **Model Settings**")
      if not all(tmp_check.values()):
        st.stop()
    
    if any([generate_btn, upload_btn, test_locally_btn]):
      validate_chat_template()
      generated_model_dir = os.path.join(output_folder, valid_model_id)
      build_model_upload(
        inference_framework=infer_framework,
        custom_framework_kwargs=custom_server_args,
        config_yaml_data=yaml_config,
        output_dir=generated_model_dir
      )
      st.info(f"Model code was generated at '{generated_model_dir}'")
      
      if upload_btn:
        with st.spinner("Uploading model.."):
          #upload_model(generated_model_dir, False, False)
          run_subprocess(["clarifai", "model", "upload", "--model_path", str(generated_model_dir), "--skip_dockerfile"],)
          builder = ModelBuilder(generated_model_dir)
          st.success(f"Uploaded model, please see model at {builder.model_url} or use this url for inference in SDK.")
      elif test_locally_btn:
        with st.spinner("Testing model locally..."):
          cmds = [
            "clarifai", "model", "test-locally", "--model_path", str(generated_model_dir), "--keep_env", "--mode", "container"]
          run_subprocess(cmds)
        


if __name__ == "__main__":
  display()