from typing import Dict, List
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2
import json

def text_to_proto(prompt):
  return resources_pb2.Text(raw=prompt)

def any_image_to_proto(image):
  if isinstance(image, str):
    return resources_pb2.Image(url=image)
  elif isinstance(image, bytes):
    return resources_pb2.Image(base64=image)

def chat_history_to_input_proto(chat:List[Dict]):
  data_parts = []
  for each in chat:
    role = each["role"]
    content = each["content"]
    data = resources_pb2.Data()
    # if multimodal data
    if isinstance(content, list):
      for each_content in content:
        if each_content["type"] == "text":
          prompt = """{{"role": "user", "content": "{content}"}}""".format(content=each_content["text"])
          data.text.CopyFrom(text_to_proto(prompt))
        elif each_content["type"] == "image_url":
          data.parts.append(
            resources_pb2.Part(
              data=resources_pb2.Data(
                image=any_image_to_proto(each_content["image_url"]["url"])
                )
              )
            )
    else:
      data.text.CopyFrom(text_to_proto(json.dumps(each)))
    
    data_parts.append(resources_pb2.Part(data=data))

  return resources_pb2.Input(data=resources_pb2.Data(parts=data_parts))
  
if __name__ == "__main__":
  messages=[
    {
      "role": "user",
      "content": [{ "type": "text", "text": "knock knock." }]
    },
    {
      "role": "assistant",
      "content": [{ "type": "text", "text": "Who's there?" }]
    },
    {
      "role": "user",
      "content": [{ "type": "text", "text": "Orange." }]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is in this image?",
            },
            {
                "type": "image_url",
                "image_url": {"url": "http"},
            },
        ],
    }
  ]
  input_proto = chat_history_to_input_proto(messages)
  print(input_proto)