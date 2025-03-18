from typing import List
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf.struct_pb2 import Struct, Value
from google.protobuf import json_format
from clarifai.client import BaseClient


def patch_infer_params(model: resources_pb2.Model, version_id:str, infer_params: List[dict], pat=None):
  param_specs = []
  for each in infer_params:
    params_proto = json_format.ParseDict(each, resources_pb2.ModelTypeField())
    param_specs.append(params_proto)
  reqs = service_pb2.PatchModelVersionsRequest(
    user_app_id=resources_pb2.UserAppIDSet(user_id=model.user_id, app_id=model.app_id),
    model_id=model.id,
    model_versions=[
      resources_pb2.ModelVersion(
          id=version_id,
        output_info=resources_pb2.OutputInfo(
            params_specs=param_specs
        )
      )
    ],
    action="overwrite"
  )
  client = BaseClient(user_id=model.user_id, app_id=model.app_id, pat=None)
  response = client._grpc_request(client.STUB.PatchModelVersions, reqs)
  print(response)
  
if __name__ == "__main__":
  from clarifai.runners.models.model_builder import ModelBuilder
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
  builder = ModelBuilder("tmp/Qwen2_5-0_5B-Instruct-params-specs")
  version_id = "85c56dd13d2b498e94136e6e4097d1b5"
  patch_infer_params(builder.model_proto,
                     version_id=version_id, infer_params=infer_params)
