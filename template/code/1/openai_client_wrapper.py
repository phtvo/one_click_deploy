import base64
import itertools
import json
from typing import Iterator, List, Union
from clarifai.utils.logging import logger
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2, status_pb2
from google.protobuf import json_format
from openai import OpenAI

SYSTEM = "system"
USER = "user"
ASSISTANT = "assistant"


def get_inference_params(request) -> dict:
  """Get the inference params from the request."""
  inference_params = {}
  output_info = request.model.model_version.output_info
  output_info = json_format.MessageToDict(
      output_info, preserving_proto_field_name=True)
  if "params" in output_info:
    inference_params = output_info["params"]

  return inference_params


def image_proto_to_chat(image: resources_pb2.Image) -> Union[str, None]:
  if image.base64:
    image = "data:image/jpeg;base64," + \
        base64.b64encode(image.base64).decode("utf-8")
    image = {"type": "image_url", "image_url": {"url": image}}
  elif image.url:
    image = image.url
    image = {"type": "image_url", "image_url": {"url": image}}
  else:
    image = None

  return image


def audio_proto_to_chat(audio: resources_pb2.Audio) -> Union[str, None]:
  if audio.base64:
    audio = base64.b64encode(audio.base64).decode("utf-8")
    audio = {"type": "input_audio", "input_audio": {
        "data": audio, "format": "wav"}, }
  elif audio.url:
    audio = audio.url
    audio = {"type": "audio_url", "audio_url": {"url": audio}, }
  else:
    audio = None

  return audio


def video_proto_to_chat(video: resources_pb2.Video) -> Union[str, None]:
  if video.base64:
    video = "data:video/mp4;base64," + \
        base64.b64encode(video.base64).decode("utf-8")
    video = {"type": "video_url", "video_url": {"url": video}, }
  elif video.url:
    video = video.url
    video = {"type": "video_url", "video_url": {"url": video}, }
  else:
    video = None

  return video


def proto_to_chat(inp: resources_pb2.Input, modalities=["audio", "image", "video"]) -> List:
  prompt = inp.data.text.raw

  # extract role and content in text if possible
  try:
    prompt = json.loads(prompt)
    role = str(prompt.get("role", USER)).lower()
    prompt = prompt.get("content", "")
  except:
    role = USER
  if role != SYSTEM:
    content = [
        {'type': 'text', 'text': str(prompt)},
    ]

    def add_modality(modality_type, process_func):
      logger.debug("Client: Parsing " + modality_type + " using", process_func)
      chat_data = process_func(getattr(inp.data, modality_type))
      _cont = []
      if chat_data:
        _cont.append(chat_data)
        # each turn could have more than 1 data
        for each_part in inp.data.parts:
          sub_chat_data = process_func(getattr(each_part.data, modality_type))
          if sub_chat_data:
            _cont.append(sub_chat_data)
      content.extend(_cont)
      logger.debug(f"Client     * {len(_cont)} {modality_type}(s)")

    if "image" in modalities:
      add_modality("image", image_proto_to_chat)
    if "audio" in modalities:
      add_modality("audio", audio_proto_to_chat)
    if "video" in modalities:
      add_modality("video", video_proto_to_chat)
    msg = {
        'role': role,
        'content': content
    }

  elif prompt:
    msg = {"role": role, "content": str(prompt)}
  else:
    msg = None

  return msg


def parse_request(request: service_pb2.PostModelOutputsRequest, modalities=["audio", "image", "video"]):
  inference_params = get_inference_params(request)
  logger.info(f"Client: inference_params: {inference_params}")
  temperature = inference_params.pop("temperature", 0.7)
  max_tokens = inference_params.pop("max_tokens", 256)
  top_p = inference_params.pop("top_p", .95)
  _ = inference_params.pop("stream", None)
  chat_history = inference_params.pop("chat_history", False)

  batch_messages = []
  try:
    for input_proto in request.inputs:
      # Treat 'parts' as history [0:-1) chat + new chat [-1]
      # And discard everything in input_proto.data
      messages = []
      if chat_history:
        for each_part in input_proto.data.parts:
          extrmsg = proto_to_chat(each_part, modalities=modalities)
          if extrmsg:
            messages.append(extrmsg)
      # If not chat_history, input_proto.data as input
      # And parts as sub data e.g. image1, image2
      else:
        new_message = proto_to_chat(input_proto, modalities=modalities)
        if new_message:
          messages.append(new_message)
      batch_messages.append(messages)
  except Exception as e:
    raise e
  gen_config = dict(
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      **inference_params)

  return batch_messages, gen_config


class OpenAIWrapper():

  def __init__(self, client: OpenAI, modalities=["audio", "image", "video"], **kwargs):
    self.client = client
    models = self.client.models.list()
    self.model_id = models.data[0].id
    self.modalities = modalities

  @staticmethod
  def make_api_url(host, port, ver="v1"):
    return f"http://{host}:{port}/{ver}"

  def predict(
      self,
      request: service_pb2.PostModelOutputsRequest,
      extra_body: dict = {}
  ) -> service_pb2.MultiOutputResponse:

    messages, inference_params = parse_request(
        request, modalities=self.modalities)
    logger.debug("Client: Sending")
    list_kwargs = [
        dict(
            model=self.model_id,
            messages=msg,
            **inference_params,
            extra_body=extra_body,
            stream=True,
            stream_options={"include_usage": True}
        ) for msg in messages
    ]

    streams = [
        self.client.chat.completions.create(**kwargs) for kwargs in list_kwargs]
    outputs = [resources_pb2.Output() for _ in request.inputs]
    for output in outputs:
      output.status.code = status_code_pb2.SUCCESS

    for chunk_batch in itertools.zip_longest(*streams, fillvalue=None):
      for idx, chunk in enumerate(chunk_batch):
        if chunk:
          if chunk.choices:
            outputs[idx].data.text.raw += chunk.choices[0].delta.content if (
                chunk and chunk.choices[0].delta.content) is not None else ''
          if chunk.usage:
            outputs[idx].prompt_tokens = chunk.usage.prompt_tokens
            outputs[idx].completion_tokens = chunk.usage.completion_tokens

    return service_pb2.MultiOutputResponse(outputs=outputs, status=status_pb2.Status(code=status_code_pb2.SUCCESS))

  def generate(
      self,
      request: service_pb2.PostModelOutputsRequest,
      extra_body: dict = {}
  ) -> Iterator[service_pb2.MultiOutputResponse]:

    messages, inference_params = parse_request(
        request, modalities=self.modalities)
    list_kwargs = [
        dict(
            model=self.model_id,
            messages=msg,
            **inference_params,
            extra_body=extra_body,
            stream=True,
            stream_options={"include_usage": True}
        ) for msg in messages
    ]

    streams = [
        self.client.chat.completions.create(**kwargs)
        for kwargs in list_kwargs
    ]

    for chunk_batch in itertools.zip_longest(*streams, fillvalue=None):
      resp = service_pb2.MultiOutputResponse(
          status=status_pb2.Status(code=status_code_pb2.SUCCESS))
      for chunk in chunk_batch:
        output = resp.outputs.add()
        output.status.code = status_code_pb2.SUCCESS
        if chunk:
          if chunk.choices:
            text = (chunk.choices[0].delta.content
                    if (chunk and chunk.choices[0].delta.content) is not None else '')
            output.data.text.raw = text
          if chunk.usage:
            output.prompt_tokens = chunk.usage.prompt_tokens
            output.completion_tokens = chunk.usage.completion_tokens

      yield resp
