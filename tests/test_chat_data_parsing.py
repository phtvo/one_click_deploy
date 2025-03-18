import base64
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../template/code/1/"))

import pytest
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from google.protobuf.struct_pb2 import Struct, Value

from openai_client_wrapper import proto_to_chat, parse_request


def test_multi_images_single_request():
  proto_data = resources_pb2.Input(
    data=resources_pb2.Data(
        text=resources_pb2.Text(raw="abc"),
        image=resources_pb2.Image(url="fake_image_url"),
        parts=[
          resources_pb2.Part(
            data=resources_pb2.Data(
              image=resources_pb2.Image(url="fake_image_url2")
            )
          ),
          resources_pb2.Part(
            data=resources_pb2.Data(
              image=resources_pb2.Image(base64=b"fake_image_bytes")
            )
          ),
        ]
    )
  )
  
  expected_output_chat_data = {
      "role": "user",
      "content": [
        {"type": "text", "text": "abc"},
        {"type": "image_url","image_url": {"url": "fake_image_url"}},
        {"type": "image_url","image_url": {"url": "fake_image_url2"}},
        {"type": "image_url","image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b"fake_image_bytes").decode("utf-8")}},
      ]
  }
  chat_data = proto_to_chat(proto_data)
  assert chat_data == expected_output_chat_data



def test_multi_audios_single_request():
  proto_data = resources_pb2.Input(
    data=resources_pb2.Data(
        text=resources_pb2.Text(raw="abc"),
        audio=resources_pb2.Audio(url="fake_audio_url"),
        parts=[
          resources_pb2.Part(
            data=resources_pb2.Data(
              audio=resources_pb2.Audio(url="fake_audio_url2")
            )
          ),
          resources_pb2.Part(
            data=resources_pb2.Data(
                audio=resources_pb2.Audio(base64=b"fake_audio_bytes")
            )
          ),
        ]
    )
  )
  
  expected_output_chat_data = {
      "role": "user",
      "content": [
        {"type": "text", "text": "abc"},
        {"type": "audio_url","audio_url": {"url": "fake_audio_url"}},
        {"type": "audio_url","audio_url": {"url": "fake_audio_url2"}},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(
            b"fake_audio_bytes").decode("utf-8"), "format": "wav"}},
      ]
  }
  chat_data = proto_to_chat(proto_data)
  assert chat_data == expected_output_chat_data

def test_multi_videos_single_request():
  proto_data = resources_pb2.Input(
    data=resources_pb2.Data(
        text=resources_pb2.Text(raw="abc"),
        video=resources_pb2.Video(url="fake_video_url"),
        parts=[
          resources_pb2.Part(
            data=resources_pb2.Data(
              video=resources_pb2.Video(url="fake_video_url2")
            )
          ),
          resources_pb2.Part(
            data=resources_pb2.Data(
                video=resources_pb2.Video(base64=b"fake_video_bytes")
            )
          ),
        ]
    )
  )
  
  expected_output_chat_data = {
      "role": "user",
      "content": [
        {"type": "text", "text": "abc"},
        {"type": "video_url","video_url": {"url": "fake_video_url"}},
        {"type": "video_url","video_url": {"url": "fake_video_url2"}},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," +  base64.b64encode(
            b"fake_video_bytes").decode("utf-8")}},
      ]
  }
  chat_data = proto_to_chat(proto_data)
  assert chat_data == expected_output_chat_data
  

def test_multi_videos_single_request():
  proto_data = resources_pb2.Input(
      data=resources_pb2.Data(
          text=resources_pb2.Text(raw="abc"),
          video=resources_pb2.Video(url="fake_video_url"),
          parts=[
              resources_pb2.Part(
                  data=resources_pb2.Data(
                      video=resources_pb2.Video(url="fake_video_url2")
                  )
              ),
              resources_pb2.Part(
                  data=resources_pb2.Data(
                      video=resources_pb2.Video(base64=b"fake_video_bytes")
                  )
              ),
          ]
      )
  )

  expected_output_chat_data = {
      "role": "user",
      "content": [
          {"type": "text", "text": "abc"},
          {"type": "video_url", "video_url": {"url": "fake_video_url"}},
          {"type": "video_url", "video_url": {"url": "fake_video_url2"}},
          {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(
              b"fake_video_bytes").decode("utf-8")}},
      ]
  }
  chat_data = proto_to_chat(proto_data)
  assert chat_data == expected_output_chat_data


def test_select_mm_data():
  proto_data = resources_pb2.Input(
    data=resources_pb2.Data(
        text=resources_pb2.Text(raw="abc"),
        video=resources_pb2.Video(url="video"),
        image=resources_pb2.Image(url="image"),
        audio=resources_pb2.Audio(url="audio"),
    )
  )
  
  video_chat_data = proto_to_chat(proto_data, ["video"])
  expected_output_chat_data = {
      "role": "user",
      "content": [
        {"type": "text", "text": "abc"},
        {"type": "video_url","video_url": {"url": "video"}},
      ]
  }
  assert video_chat_data == expected_output_chat_data

  image_chat_data = proto_to_chat(proto_data, ["image"])
  expected_output_chat_data = {
      "role": "user",
      "content": [
          {"type": "text", "text": "abc"},
          {"type": "image_url", "image_url": {"url": "image"}},
      ]
  }
  assert image_chat_data == expected_output_chat_data
  
  audio_chat_data = proto_to_chat(proto_data, ["audio"])
  expected_output_chat_data = {
      "role": "user",
      "content": [
          {"type": "text", "text": "abc"},
          {"type": "audio_url", "audio_url": {"url": "audio"}},
      ]
  }
  assert audio_chat_data == expected_output_chat_data

  image_audio_chat_data = proto_to_chat(proto_data, ["image", "audio"])
  expected_output_chat_data = {
      "role": "user",
      "content": [
          {"type": "text", "text": "abc"},
          {"type": "image_url", "image_url": {"url": "image"}},
          {"type": "audio_url", "audio_url": {"url": "audio"}},
      ]
  }
  assert image_audio_chat_data == expected_output_chat_data
  
  image_video_chat = proto_to_chat(proto_data, ["image", "video"])
  expected_output_chat_data = {
      "role": "user",
      "content": [
          {"type": "text", "text": "abc"},
          {"type": "image_url", "image_url": {"url": "image"}},
          {"type": "video_url", "video_url": {"url": "video"}},
      ]
  }
  assert image_video_chat == expected_output_chat_data


def make_request(proto_data, infer_params:dict = {}):
  params = Struct()
  params.update(infer_params)
  model_info = resources_pb2.Model(
      model_version=resources_pb2.ModelVersion(
          pretrained_model_config=resources_pb2.PretrainedModelConfig(),
      ))
  model_info.model_version.output_info.CopyFrom(
      resources_pb2.OutputInfo(
          output_config=resources_pb2.OutputConfig(**{}), params=params))
  
  cl_request = service_pb2.PostModelOutputsRequest(
      model=model_info,
      inputs=[proto_data]
  )

  return cl_request

def test_parse_request():
  proto_data = resources_pb2.Input(
      data=resources_pb2.Data(
          text=resources_pb2.Text(raw="abc"),
          video=resources_pb2.Video(url="video"),
          image=resources_pb2.Image(url="image"),
          audio=resources_pb2.Audio(url="audio"),
      )
  )
  cl_request = make_request(proto_data)

  batch_msgs, infer_params = parse_request(cl_request)
  
  default_infer_params = dict(
      temperature=0.7,
      max_tokens=256,
      top_p=0.95
  )
  expected_output_chat_data = {
      "role": "user",
      "content": [
          {"type": "text", "text": "abc"},
          {"type": "image_url", "image_url": {"url": "image"}},
          {"type": "audio_url", "audio_url": {"url": "audio"}},
          {"type": "video_url", "video_url": {"url": "video"}},
      ]
  }
  assert default_infer_params == infer_params
  assert expected_output_chat_data == batch_msgs[0][0]


def test_parse_chat_request():
  
  # Empty
  proto_data = resources_pb2.Input(
      data=resources_pb2.Data(
          text=resources_pb2.Text(raw="abc"),
          video=resources_pb2.Video(url="video"),
          image=resources_pb2.Image(url="image"),
          audio=resources_pb2.Audio(url="audio"),
      )
  )
  
  cl_request = make_request(proto_data, infer_params=dict(chat_history=True))

  batch_msgs, infer_params = parse_request(cl_request)

  default_infer_params = dict(
      temperature=0.7,
      max_tokens=256,
      top_p=0.95
  )
  expected_output_chat_data = []
  assert default_infer_params == infer_params
  assert expected_output_chat_data == batch_msgs[0]

  
  ###################
  # Multi turns text chat
  proto_data = resources_pb2.Input(
      data=resources_pb2.Data(
          parts=[
              resources_pb2.Part(
                  data=resources_pb2.Data(
                      text=resources_pb2.Text(
                          raw='{"role": "system", "content": "You are an AI"}')
                  )
              ),
              resources_pb2.Part(
                  data=resources_pb2.Data(
                      text=resources_pb2.Text(
                          raw='{"role": "user", "content": "What is AI?"}')
                  )
              ),
              resources_pb2.Part(
                  data=resources_pb2.Data(
                      text=resources_pb2.Text(
                          raw='{"role": "assistant", "content": "I dont know"}')
                  )
              ),
          ]
      )
  )

  cl_request = make_request(proto_data, infer_params=dict(chat_history=True))

  batch_msgs, infer_params = parse_request(cl_request)

  default_infer_params = dict(
      temperature=0.7,
      max_tokens=256,
      top_p=0.95
  )
  expected_output_chat_data = [
      {"role": "system", "content": "You are an AI" },
      {"role": "user", "content": [{"type": "text", "text": "What is AI?"}]},
      {"role": "assistant", "content": [{"type": "text", "text": "I dont know"}] }
  ]
  assert default_infer_params == infer_params
  assert expected_output_chat_data == batch_msgs[0]


def test_mm_in_chat():
  ###################
  # Multi turns text chat
  proto_data = resources_pb2.Input(
    data=resources_pb2.Data(
      parts=[
        resources_pb2.Part(
            data=resources_pb2.Data(
                text=resources_pb2.Text(
                    raw='{"role": "system", "content": "You are an AI"}')
            )
        ),
        resources_pb2.Part(
            data=resources_pb2.Data(
                text=resources_pb2.Text(
                    raw='{"role": "user", "content": "What is AI?"}'),
                image=resources_pb2.Image(url="fake_image_url"),
                video=resources_pb2.Video(url="fake_video_url"),
                audio=resources_pb2.Audio(url="fake_audio_url")
            )
        ),
        resources_pb2.Part(
            data=resources_pb2.Data(
                text=resources_pb2.Text(
                    raw='{"role": "assistant", "content": "I dont know"}')
            )
        ),
        resources_pb2.Part(
            data=resources_pb2.Data(
                text=resources_pb2.Text(
                    raw='{"role": "user", "content": "What is AI?"}'),
                image=resources_pb2.Image(url="fake_image_url2"),
            )
        ),
        resources_pb2.Part(
            data=resources_pb2.Data(
                text=resources_pb2.Text(
                    raw='{"role": "user", "content": "What is AI?"}'),
                image=resources_pb2.Image(url="fake_image_url3"),
            )
        ),
      ]
    )
  )

  cl_request = make_request(proto_data, infer_params=dict(chat_history=True))

  batch_msgs, infer_params = parse_request(cl_request)

  default_infer_params = dict(
      temperature=0.7,
      max_tokens=256,
      top_p=0.95
  )
  expected_output_chat_data = [
      {"role": "system", "content": "You are an AI" },
      {"role": "user", 
       "content": [
          {"type": "text", "text": "What is AI?"},
          {"type": "image_url", "image_url": {"url": "fake_image_url"}},
          {"type": "audio_url", "audio_url": {"url": "fake_audio_url"}},
          {"type": "video_url", "video_url": {"url": "fake_video_url"}},
        ]
      },
      {"role": "assistant", "content": [{"type": "text", "text": "I dont know"}] },
      {"role": "user", 
       "content": [
          {"type": "text", "text": "What is AI?"},
          {"type": "image_url", "image_url": {"url": "fake_image_url2"}},
        ]
      },
      {"role": "user", 
       "content": [
          {"type": "text", "text": "What is AI?"},
          {"type": "image_url", "image_url": {"url": "fake_image_url3"}},
        ]
      },
  ]
  assert default_infer_params == infer_params
  assert expected_output_chat_data == batch_msgs[0]
  

def test_many_mm_in_single_turn_in_chat():
  ###################
  # Multi turns text chat
  proto_data = resources_pb2.Input(
      data=resources_pb2.Data(
          parts=[
              resources_pb2.Part(
                  data=resources_pb2.Data(
                      text=resources_pb2.Text(
                          raw='{"role": "system", "content": "You are an AI"}')
                  )
              ),
              resources_pb2.Part(
                  data=resources_pb2.Data(
                      text=resources_pb2.Text(
                          raw='{"role": "user", "content": "What is AI?"}'),
                      parts=[
                          resources_pb2.Part(
                              data=resources_pb2.Data(
                                  audio=resources_pb2.Audio(
                                      url="fake_audio_url")
                              )
                          ),
                          resources_pb2.Part(
                              data=resources_pb2.Data(
                                  image=resources_pb2.Image(
                                      url="fake_image_url")
                              )
                          ),
                      ]
                  )
              ),
              resources_pb2.Part(
                  data=resources_pb2.Data(
                      text=resources_pb2.Text(
                          raw='{"role": "assistant", "content": "I dont know"}')
                  )
              ),
              resources_pb2.Part(
                  data=resources_pb2.Data(
                      text=resources_pb2.Text(
                          raw='{"role": "user", "content": "What is AI?"}'),
                      parts=[
                          resources_pb2.Part(
                              data=resources_pb2.Data(
                                  video=resources_pb2.Video(
                                      url="fake_video_url")
                              )
                          ),
                          resources_pb2.Part(
                              data=resources_pb2.Data(
                                  image=resources_pb2.Image(
                                      url="fake_image_url2")
                              )
                          ),
                      ]
                  )
              ),
              resources_pb2.Part(
                  data=resources_pb2.Data(
                      text=resources_pb2.Text(
                          raw='{"role": "user", "content": "What is AI?"}'),
                      image=resources_pb2.Image(url="fake_image_url3"),
                      parts=[
                          resources_pb2.Part(
                              data=resources_pb2.Data(
                                  video=resources_pb2.Video(
                                      url="fake_video_url2")
                              )
                          ),
                          resources_pb2.Part(
                              data=resources_pb2.Data(
                                  image=resources_pb2.Image(
                                      url="fake_image_url4")
                              )
                          ),
                      ]
                  )
              ),
          ]
      )
  )

  cl_request = make_request(proto_data, infer_params=dict(chat_history=True))

  batch_msgs, infer_params = parse_request(cl_request)

  default_infer_params = dict(
      temperature=0.7,
      max_tokens=256,
      top_p=0.95
  )
  expected_output_chat_data = [
      {"role": "system", "content": "You are an AI"},
      {"role": "user",
       "content": [
           {"type": "text", "text": "What is AI?"},
           {"type": "image_url", "image_url": {"url": "fake_image_url"}},
           {"type": "audio_url", "audio_url": {"url": "fake_audio_url"}},
       ]
       },
      {"role": "assistant", "content": [
          {"type": "text", "text": "I dont know"}]},
      {"role": "user",
       "content": [
           {"type": "text", "text": "What is AI?"},
           {"type": "image_url", "image_url": {"url": "fake_image_url2"}},
           {"type": "video_url", "video_url": {"url": "fake_video_url"}},
       ]
       },
      {"role": "user",
       "content": [
           {"type": "text", "text": "What is AI?"},
           {"type": "image_url", "image_url": {"url": "fake_image_url3"}},
           {"type": "image_url", "image_url": {"url": "fake_image_url4"}},
           {"type": "video_url", "video_url": {"url": "fake_video_url2"}},
       ]
       },
  ]
  assert default_infer_params == infer_params
  assert expected_output_chat_data == batch_msgs[0]
