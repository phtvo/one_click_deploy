from clarifai.client import Model
from clarifai.client.input import Inputs
from clarifai_grpc.grpc.api.status import status_code_pb2


model = Model(base_url="http://localhost:8000", model_id="123", user_id="123", app_id="123", pat="123")
prompt = "What’s the future of AI?"

inference_params = dict(temperature=0.2, max_tokens=100, frequency_penalty=0.6, presence_penalty=0.8) # Optional

results = model.predict_by_bytes(prompt.encode("utf-8"), "text", inference_params=inference_params)
if results.status.code == status_code_pb2.SUCCESS:
  print(results.outputs[0].data.text.raw)


### Multimodal  
prompt = "What time of day is it?"
image_url = "https://samples.clarifai.com/metro-north.jpg"

input_data = Inputs.get_multimodal_input(input_id="",image_url=image_url, raw_text=prompt)
inference_params = dict(temperature=0.2, max_tokens=100, frequency_penalty=0.6, presence_penalty=0.8) # Optional

results = model.predict([input_data], inference_params=inference_params)

if results.status.code == status_code_pb2.SUCCESS:
  print(results.outputs[0].data.text.raw)