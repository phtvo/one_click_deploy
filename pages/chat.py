import os
import sys
import time

sys.path.append(os.path.dirname(__file__))
from utils.utils import chat_history_to_input_proto

##
import streamlit as st
st.set_page_config(layout="wide")

from clarifai.client import Model
from clarifai_grpc.grpc.api.status import status_code_pb2

def display():
  st.title("Clarifai chat demo")

  # Sidebar
  st.sidebar.header("Model Setting")
  model_url = st.sidebar.text_input("Model URL", value="https://clarifai.com/phatvo/text-generation/models/Qwen2_5-0_5B-Instruct-vllm")
  pat = st.sidebar.text_input("PAT", type="password")
  if pat:
    os.environ["CLARIFAI_PAT"] = pat
  if not os.environ.get("CLARIFAI_PAT"):
    st.error("Please insert your PAT")
    st.stop()

  is_generate = st.sidebar.toggle("Stream response", value=True)

  st.sidebar.header("Inference Parameters")
  temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
  max_tokens = st.sidebar.slider("Max Tokens", min_value=10, max_value=2000, value=100, step=10)
  top_p = st.sidebar.slider("Top-p (nucleus sampling)", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
  # top_k = st.sidebar.slider("Top-k (sampling)", min_value=0, max_value=100, value=50, step=5)
  # frequency_penalty = st.sidebar.slider("Frequency Penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
  # presence_penalty = st.sidebar.slider("Presence Penalty", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

  inference_kwargs = dict(
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    chat_history=True
  )

  # Sidebar for system prompt
  st.sidebar.header("System Prompt")
  system_prompt = st.sidebar.text_area("Enter system prompt", "You are a helpful assistant.")
  reset_btn = st.sidebar.button("Reset")
  if reset_btn:
    st.session_state.pop("messages", "")

  # Init model
  if not "model_url" in st.session_state or st.session_state["model_url"] != model_url:
    st.session_state["model_url"] = model_url
  model = Model(url=st.session_state["model_url"], pat=os.environ.get("CLARIFAI_PAT", "xx"))
  print("Model: ", st.session_state["model_url"])
  # system prompt
  if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = system_prompt

  # Initialize session state for chat history if not exists. System prompt will place at 1st position.
  if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": system_prompt},]

  # Assign new system prompt
  if st.session_state["system_prompt"] != system_prompt:
    st.session_state["messages"][0] = {"role": "system", "content": system_prompt}
    st.session_state["system_prompt"] = system_prompt

  chat_col, proto_col = st.columns([3, 2])
  
  # Display chat messages from session state
  for msg in st.session_state["messages"]:
    if msg["role"] != "system":
      with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

  # Input for user message
  user_input = st.chat_input("Type your message...")
  input_proto = None
  if user_input:
      # Append user message to chat history
      st.session_state["messages"].append({"role": "user", "content": user_input})
      with st.chat_message("user"):
          st.markdown(user_input)
      # Stream assistant response and measure throughput
      input_proto = chat_history_to_input_proto(st.session_state["messages"])
      print("HISTORY:\n", st.session_state["messages"])
      #print(input_proto)
      print(inference_kwargs)
      model_response_text = ""
      if is_generate:
        with st.spinner("Loading the model..."):
          start_time = time.time()
          response_generator = model.generate([input_proto], inference_params=inference_kwargs)
          with st.chat_message("assistant"):
            response_placeholder = st.empty()
            model_response_text = ""
            token_count = 0
            for resp in response_generator:
              if resp.status.code == status_code_pb2.SUCCESS:
                word = resp.outputs[0].data.text.raw
                model_response_text += word
                print(word)
                response_placeholder.markdown(model_response_text)
                token_count += 1
              else:
                st.error(status_code_pb2)
                #time.sleep(0.01)
            end_time = time.time()
            throughput = token_count / (end_time - start_time)
        # Display throughput
        st.markdown(f"**Token Throughput:** {throughput:.2f} tokens/sec. Output tokens: {token_count}")
      else:
        with st.spinner("Loading the model..."):
          response = model.predict([input_proto], inference_params=inference_kwargs)
          with st.chat_message("assistant"):
            if model_response_text.status.code == status_code_pb2.SUCCESS:
              model_response_text = response.outputs[0].data.text.raw
              st.markdown(model_response_text)
            else:
              st.error(status_code_pb2)
      # Append assistant message to chat history
      st.session_state["messages"].append({"role": "assistant", "content": model_response_text})
  with st.sidebar:
    st.markdown("---")
    st.subheader("Input format")
    
    st.markdown("##### Must set inference params: `chat_history=True`")
    st.markdown("##### Input proto")
    st.write(input_proto)
      
if __name__ == "__main__":
  display()