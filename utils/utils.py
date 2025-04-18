import base64
from copy import deepcopy
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

def any_audio_to_proto(audio):
  if isinstance(audio, str):
    return resources_pb2.Audio(url=audio)
  elif isinstance(audio, bytes):
    return resources_pb2.Audio(base64=audio)

def any_video_to_proto(video):
  if isinstance(video, str):
    return resources_pb2.Video(url=video)
  elif isinstance(video, bytes):
    return resources_pb2.Video(base64=video)

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
          data.image.CopyFrom(any_image_to_proto(each_content["image_url"]["url"]))
          # data.parts.append(
          #   resources_pb2.Part(
          #     data=resources_pb2.Data(
          #       image=any_image_to_proto(each_content["image_url"]["url"])
          #       )
          #     )
          #   )
        elif each_content["type"] == "input_audio":
          data.image.CopyFrom(
            any_audio_to_proto(each_content["input_audio"]["data"]))
          # data.parts.append(
          #   resources_pb2.Part(
          #     data=resources_pb2.Data(
          #         audio=any_audio_to_proto(
          #             each_content["input_audio"]["data"])
          #       )
          #     )
          #   )
        elif each_content["type"] == "video_url":
          data.image.CopyFrom(
            any_video_to_proto(each_content["video_url"]["url"]))
          # data.parts.append(
          #   resources_pb2.Part(
          #     data=resources_pb2.Data(
          #         video=any_video_to_proto(
          #             each_content["video_url"]["url"])
          #       )
          #     )
          #   )
    else:
      data.text.CopyFrom(text_to_proto(json.dumps(each)))
    
    data_parts.append(resources_pb2.Part(data=data))

  return resources_pb2.Input(data=resources_pb2.Data(parts=data_parts))


def chat_history_from_st_cache(chat:List[Dict]):
  copied_chat = deepcopy(chat)
  for msg in copied_chat:
    if msg["role"] != "system":
      if isinstance(msg["content"], list):
        for content in msg["content"]:
          if content["type"] == "text":
            pass
          elif content["type"] == "image_url":
            b64_img = base64.b64encode(
                content["image_url"]["url"]).decode('utf-8')
            
            content["image_url"]["url"] = f"data:image/jpeg;base64,{b64_img}"
          elif content["type"] == "input_audio":
            audio = base64.b64encode(
                content["input_audio"]["data"]).decode('utf-8')
            content["input_audio"].update(
              {
                "data": audio,
                "format": "wav"
              }
            )
          elif content["type"] == "video_url":
            video = "data:video/mp4;base64," + \
                base64.b64encode(content["video_url"]["url"]).decode("utf-8")
            content["video_url"]["url"] = video
  
  return copied_chat


def chat_history_from_st_cache_for_render(chat: List[Dict]):
  copied_chat = deepcopy(chat)
  for msg in copied_chat:
    if msg["role"] != "system":
      if isinstance(msg["content"], list):
        for content in msg["content"]:
          if content["type"] == "text":
            pass
          elif content["type"] == "image_url":
            b64_img = len(content["image_url"]["url"])
            content["image_url"]["url"] = f"data:image/jpeg;base64,<encoded base64 of {b64_img} bytes of image>"
          elif content["type"] == "input_audio":
            audio = len(
                content["input_audio"]["data"])
            content["input_audio"].update(
                {
                    "data": f"<encoded base64 of {audio} bytes of audio>",
                    "format": "wav"
                }
            )
          elif content["type"] == "video_url":
            video = len(content["video_url"]["url"])
            content["video_url"]["url"] = f"<encoded base64 of {video} bytes of video>"

  return copied_chat
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
            {
                "type": "image_url",
                "image_url": {"url": "http2"},
            },
            {
                "type": "input_audio",
                "input_audio": {"data": "http"},
            },
            {
                "type": "video_url",
                "video_url": {"url": "http"},
            },
        ],
    }
  ]
  input_proto = chat_history_to_input_proto(messages)
  print(input_proto)