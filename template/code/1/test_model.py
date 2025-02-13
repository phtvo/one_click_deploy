# Runner test your implementation locally
#
import os
import sys

sys.path.append(os.path.dirname(__file__))
from model import MyRunner
##

from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2

if __name__ == "__main__":
  model = MyRunner()
  model.load_model()
  prompt = "How are these image differnt from each other?"
  prompt2 = "Explain why cat can't fly"
  image = b"00012555"
  cl_request = service_pb2.PostModelOutputsRequest(
        model=resources_pb2.Model(model_version=resources_pb2.ModelVersion(
        pretrained_model_config=resources_pb2.PretrainedModelConfig(),
    )),
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    text=resources_pb2.Text(raw=prompt),
                    image=resources_pb2.Image(url="https://s3.amazonaws.com/samples.clarifai.com/featured-models/food-hamburgers-bacon-cheese.jpg"),
                    parts=[
                        resources_pb2.Part(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(url="https://s3.amazonaws.com/samples.clarifai.com/featured-models/general-elephants.jpg")
                        )
                        ),
                        resources_pb2.Part(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(url="https://s3.amazonaws.com/samples.clarifai.com/featured-models/general-elephants.jpg")
                        )
                        ),
                    ]
            )),
            #resources_pb2.Input(data=resources_pb2.Data(
            #    text=resources_pb2.Text(raw=prompt2)
            #))
        ],
    )
  
  resp = model.generate(cl_request)
  for res in resp:
    if res.status.code == status_code_pb2.SUCCESS:
      text = res.outputs[0].data.text.raw
      print(text, end="", flush=False)

  ### History chat
  from google.protobuf.struct_pb2 import Struct, Value
  params = Struct()
  params.update({"chat_history": True, "temperature": 0.9})
  model_info = resources_pb2.Model(
          model_version=resources_pb2.ModelVersion(
          pretrained_model_config=resources_pb2.PretrainedModelConfig(),
            
      ))
  model_info.model_version.output_info.CopyFrom(
        resources_pb2.OutputInfo(
            output_config=resources_pb2.OutputConfig(**{}), params=params))
  cl_request = service_pb2.PostModelOutputsRequest(
        model=model_info,
        inputs=[
            resources_pb2.Input(
                data=resources_pb2.Data(
                    parts=[
                        resources_pb2.Part(
                          data=resources_pb2.Data(
                            text=resources_pb2.Text(raw='{"role": "system", "content": "You are anger person named Groot, answer the question in angry way"}')
                          )
                        ),
                        resources_pb2.Part(
                          data=resources_pb2.Data(
                            text=resources_pb2.Text(raw='{"role": "assistant", "content": "My name is Groot"}'),
                          )
                        ),
                        resources_pb2.Part(
                          data=resources_pb2.Data(
                            text=resources_pb2.Text(raw='{"role": "user", "content": "what is your name?"}'),
                            image=resources_pb2.Image(
                              url="https://s3.amazonaws.com/samples.clarifai.com/featured-models/general-elephants.jpg")
                            )
                        ),
                    ]
            )),
            resources_pb2.Input(data=resources_pb2.Data(
                text=resources_pb2.Text(raw=prompt2)
            ))
        ],
    )
  print("Demo chat history")
  print("----")
  resp = model.generate(cl_request)
  for res in resp:
    if res.status.code == status_code_pb2.SUCCESS:
      text = res.outputs[0].data.text.raw
      print(text, end="", flush=False)
