from copy import deepcopy
import io
import os
import logging
import queue
import re
import subprocess
import sys
from typing import Any, Dict
import zipfile
import streamlit as st
st.set_page_config(layout="wide")

from PIL import Image
from clarifai.client import Model, Inputs
from clarifai_grpc.grpc.api import resources_pb2
from utils.constant import FRAMEWORK_INFO as FI
from utils.build import build_model_upload
from utils.patch_infer_parms import patch_infer_params
from clarifai.runners.models.model_builder import ModelBuilder
from clarifai.utils.logging import logger
import shlex

PYTHON_EXEC = sys.executable

def create_model_note(model_type, model_name, hf_model_id, model_url, model_out_dir, inference_framework:dict, server_args:Dict[str, Any]):
  root = os.path.dirname(__file__)
  note_dir = os.path.join(root, "../template/model_notes")
  with open(os.path.join(note_dir, f"{model_type}.md"), "r") as f:
    note_data = f.read()
  
  # Create cmd
  cmds = []
  serverargs = deepcopy(server_args)
  input_data_type = list(set(serverargs.pop("modalities") + ["text"]))
  for k, v in serverargs.items():
    if v is not None:
      if k == "additional_list_args":
        for each in v:  
          cmds.extend([str(each)])
      elif k in ["checkpoints"]:
        continue
      else:
        if not k.startswith("--"):
          k = "--" + k
        cmds.extend([str(k), str(v)])
  cmds = " ".join(cmds)
  
  new_note = note_data.format(
    model_name=model_name,
    hf_model_id=hf_model_id,
    model_url=model_url,
    inference_framework=inference_framework,
    input_data_type=input_data_type,
    server_args=cmds
  )
  with open(os.path.join(model_out_dir, "MODEL_NOTES.MD"), "w") as f:
    f.write(new_note)
  
  return new_note


def extract_version(text):
    match = re.search(r"version:\s*([\w\d]+)", text)
    return match.group(1) if match else None

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
    
    # get model version at last log line
    model_version = extract_version(log)
    
    if process.returncode != 0:  # Non-zero exit code = error
      error_msg = f"Command failed with exit code {process.returncode}"
      error_area.error(error_msg)
      st.stop()
    
    return model_version

st_log_title = st.empty()
st_log = st.empty()
error_area = st.empty()
  

def display():
  # Sidebar settings
  st.sidebar.header("Settings")
  with st.sidebar:
      base_url = st.text_input("Base URL", value=os.environ.get("CLARIFAI_API_BASE","https://api.clarifai.com"))
      pat = st.text_input("PAT", type="password")
  if base_url:
    os.environ["CLARIFAI_API_BASE"] = base_url
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
  #upload_config_col, _, model_server_col, _,  params_col, _, control_col = st.columns([6, 1, 6, 1, 6, 1, 6])
  
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
    if model_type_id == "multimodal-to-text":
      input_modalities = st.multiselect("Select data types for input", options=["image", "audio", "video"], default="image")
    else:
      input_modalities = ["text"]
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
    cpu_limit = st.slider("CPU Limit", min_value=1, max_value=32, value=2)
    cpu_memory = st.slider("CPU Memory", min_value=1, max_value=128, value=8)
    cpu_memory = f"{cpu_memory}Gi"
    use_cpu_only = st.toggle("Use CPU only", value=False)
    if use_cpu_only:
      num_accelerators = 0
      accelerator_type = None
      accelerator_memory = None
    else:
      num_accelerators = st.slider("Num Accelerators", min_value=0, max_value=4, value=1)
      gpus = ["NVIDIA-A10G", "NVIDIA-L4", "NVIDIA-T4", "NVIDIA-L40S", "NVIDIA-A100", "NVIDIA-H100",]
      default_gpus = ["NVIDIA-A10G","NVIDIA-L40S", "NVIDIA-A100", "NVIDIA-H100",]
      accelerator_type = st.multiselect("Accelerator Type", options=gpus, default=default_gpus)
      accelerator_memory = st.number_input("Accelerator Memory",  value=22)
      accelerator_memory = f"{accelerator_memory}Gi"

    clarifai_threads = st.slider("Clarifai runner threads", min_value=1, max_value=128, value=32)
    # Checkpoint settings (optional)
    download_checkpoints = st.toggle("Enable cache checkpoints", value=False)
    hf_token = st.text_input("HuggingFace Token", "", type="password", help="Provide HF token to access gated model or to cache checkpoints")
    repo_id = None
    when = None
    # HF token is required to cache checkpoints
    if download_checkpoints and not hf_token:
      st.error("HuggingFace Token is required to cache checkpoints")
    elif download_checkpoints and hf_token:
      repo_id = hf_model_id
      when = "build"
    elif hf_token:
      repo_id = hf_model_id
      when = "runtime"

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
          "accelerator_type": accelerator_type,
          "accelerator_memory": accelerator_memory
      }
    }
    if when:
      yaml_config["checkpoints"] = {
          "type": "huggingface",
          "repo_id": repo_id,
          "hf_token": hf_token,
          "when": when
      }
  
  with st.sidebar:
    if os.environ.get("PROD") == "1":
      default_output_folder = '/tmp'
    else:
      default_output_folder = './tmp'
    if user_id and app_id:
      default_output_folder = os.path.join(default_output_folder, f"{user_id}__{app_id}")
    output_folder = st.text_input(
        "Path to save generated model upload folder", default_output_folder,)
  
  with model_server_col:
    st.markdown("### Inference Settings")
    infer_framework = st.selectbox("Select inference framework", FI, index=1)
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
    default_server_args = fi["init_args"]
    with st.expander("#### Server args"):
      st.markdown("These args are used to start up OpenAI compatible server, please refer to framework page for details")
      custom_server_args = {"modalities": input_modalities}
      required_args = default_server_args["required_args"]
      optional_args = default_server_args["optional_args"]
      def create_args_input(data_args, required=False):
        for arg_name, arg_info in data_args.items():
          if arg_name == "additional_list_args":
            continue
          desc = arg_info["desc"]
          arg_type = arg_info["type"]
          arg_default_value = arg_info["default"]
          input_data = None
          if arg_default_value is None or arg_type == str:
            input_data = st.text_input(
                "--" + arg_name, value=arg_default_value, help=desc)
            custom_server_args.update({arg_name: input_data})
          elif isinstance(arg_type, int) or arg_type == int:
            input_data = st.number_input("--" + arg_name, value=int(arg_default_value), help=desc)
            custom_server_args.update({arg_name: input_data})
          elif isinstance(arg_type, bool) or arg_type == bool:
            input_data = st.checkbox("--" + arg_name, value=bool(arg_default_value), help=desc)
            custom_server_args.update({arg_name: input_data})
          elif isinstance(arg_type, float) or arg_type == float:
            input_data = st.number_input("--" + arg_name, value=float(arg_default_value), min_value=0., max_value=1., step=0.01, help=desc)
            custom_server_args.update({arg_name: input_data})
          if required and not input_data:
            st.error(f"Please input `{arg_name}`, see (ⓘ) for information.")
            st.stop()
            
      if required_args:
        st.markdown("**Required args**")
        create_args_input(required_args, required=True)
      st.markdown("**Optional args**")
      create_args_input(optional_args)
      additional_args = st.text_area(
          "Additional args", help=optional_args.get("additional_list_args", {}).get("desc", "Additional args (exclude all of above args) to run the server e.g. '--abc 123 --enable-something --float 0.4'"))
    if additional_args != "":
      print("additional_args: ", additional_args)
      custom_server_args.update(dict(additional_list_args=shlex.split(additional_args)))
    custom_server_args.update(dict(checkpoints=when if when else hf_model_id))
  
  with control_col:
    st.markdown("### Run")
    generate_btn = st.button("Generate and Download code.")
    upload_btn = st.button("Upload model")
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
      generated_model_dir = os.path.join(output_folder, model_id)
      build_model_upload(
        inference_framework=infer_framework,
        custom_framework_kwargs=custom_server_args,
        config_yaml_data=yaml_config,
        output_dir=generated_model_dir,
        clarifai_threads=clarifai_threads
      )
      
      builder = ModelBuilder(generated_model_dir)
      model_note = create_model_note(
        model_type_id, hf_model_id=hf_model_id, model_name=valid_model_id, 
        model_url=builder.model_url, model_out_dir=generated_model_dir, 
        inference_framework=infer_framework, server_args=custom_server_args
      )
      
      if os.path.exists(generated_model_dir) and os.listdir(generated_model_dir):
        def zip_folder(folder_path):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zip_file.write(file_path, os.path.relpath(file_path, folder_path))
            zip_buffer.seek(0)
            return zip_buffer
        
        zip_file = zip_folder(generated_model_dir)
        st.markdown(
        """
          <style>
          .blue-button {
              display: inline-block;
              padding: 10px 20px;
              font-size: 16px;
              color: white;
              background-color: blue;
              border: none;
              border-radius: 5px;
              text-align: center;
              text-decoration: none;
              cursor: pointer;
          }
          </style>
        """,
          unsafe_allow_html=True
        )

        st.download_button(
            label="📁 Download Model Code",
            data=zip_file,
            file_name=f"{model_id}.zip",
            mime="application/zip"
        )
      
      if upload_btn:
        with st.spinner("Uploading model.."):
          #upload_model(generated_model_dir, False, False)
          model_version = run_subprocess(["clarifai", "model", "upload", "--model_path", generated_model_dir, "--skip_dockerfile"],)
          if model_version:
            st.success(f"Uploaded model, please see model at {builder.model_url} version **{model_version}** or use this url for inference in SDK.")
            
            # hardcoded params
            infer_params = [
                {
                    "path": "max_tokens",
                    "description": "The maximum number of tokens to generate. Shorter token lengths will provide faster performance.",
                    "field_type": 3,
                    "default_value": 512
                },
                {
                    "path": "temperature",
                    "description": "A decimal number that determines the degree of randomness in the response",
                    "field_type": 3,
                    "default_value": 0.7
                },
                {
                    "path": "top_p",
                    "description": "An alternative to sampling with temperature, where the model considers the results of the tokens with top_p probability mass.",
                    "field_type": 3,
                    "default_value": 0.95
                }
            ]
            
            patch_infer_params(
              builder.model_proto,
              version_id=model_version,
              infer_params=infer_params
            )
          
      elif test_locally_btn:
        if infer_framework == "llamacpp":
          st.error("Not support testing llamacpp with virtual env.")
          st.stop()
        with st.spinner("Testing model locally..."):
          cmds = [
            "clarifai", "model", "test-locally", "--model_path", str(generated_model_dir), "--keep_env", "--mode", "env"]
          run_subprocess(cmds)
      else:
        st.info(f"Model code was generated at '{generated_model_dir}'")
      
      #st.markdown("`Model Note:`")
      with st.expander("`Model Note`", expanded=True):
        st.text_area("Copy this for model note", model_note, height=1000)        


if __name__ == "__main__":
  display()