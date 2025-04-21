import base64
from typing import Dict, List, Union

from clarifai.runners.utils.data_types import Audio, Image, Video
from openai import OpenAI
from openai.types.chat import (ChatCompletionChunk, ChatCompletion)
from openai.resources.completions import Stream as OpenAIStream
from openai import (APIStatusError, BadRequestError)

class ModelBadRequestError(Exception):
  pass

def _process_image(image: Image) -> Dict:
  """Convert Clarifai Image object to OpenAI image format."""
  if image.bytes:
    b64_img = base64.b64encode(image.bytes).decode('utf-8')
    return {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{b64_img}"}}
  elif image.url:
    return {'type': 'image_url', 'image_url': {'url': image.url}}
  else:
    raise ValueError("Image must contain either bytes or URL")


def _process_audio(audio: Audio) -> Dict:
  if audio.bytes:
    audio = base64.b64encode(audio.base64).decode("utf-8")
    audio = {
        "type": "input_audio",
        "input_audio": {
            "data": audio,
            "format": "wav"
        },
    }
  elif audio.url:
    audio = audio.url
    audio = {
        "type": "audio_url",
        "audio_url": {
            "url": audio
        },
    }
  else:
    raise ValueError("Audio must contain either bytes or URL")

  return audio


def _process_video(video: Video) -> Dict:
  if video.bytes:
    video = "data:video/mp4;base64," + \
        base64.b64encode(video.base64).decode("utf-8")
    video = {
        "type": "video_url",
        "video_url": {
            "url": video
        },
    }
  elif video.url:
    video = video.url
    video = {
        "type": "video_url",
        "video_url": {
            "url": video
        },
    }
  else:
    raise ValueError("Video must contain either bytes or URL")

  return video


def build_messages(
  prompt: str = "", 
  system_prompt: str = "", 
  images: List[Image] = [],
  audios: List[Audio] = [],
  videos: List[Video] = [],
  chat_history: List[Dict] = [],
  **kwargs
) -> List[Dict]:
  """Construct OpenAI-compatible messages from input components."""
  openai_messages = []
  # Add previous conversation history
  if chat_history:
    openai_messages.extend(chat_history)
  # If no history but have system prompt
  elif system_prompt:
    openai_messages.append({
        "role": "system",
        "content": system_prompt
    })

  user_content = []
  if prompt:
    # Build user_content array for current message
    user_content.append({'type': 'text', 'text': prompt})

  # Add multiple images if present
  if images:
    for img in images:
      user_content.append(_process_image(img))
  # Add multiple audios if present
  if audios:
    for audio in audios:
      user_content.append(_process_audio(audio))
  # Add multiple videos if present
  if videos:
    for video in videos:
      user_content.append(_process_video(video))

  if user_content:
    # Append complete user message
    openai_messages.append({'role': 'user', 'content': user_content})

  return openai_messages

class OpenAIWrapper:

  def __init__(self, client: OpenAI, modalities: List[str] = None):
    self.client = client
    self.modalities = modalities or []
    self._validate_modalities()
    self.model_id = self._get_model_id()

  def _validate_modalities(self):
    valid_modalities = {'image', 'audio', 'video', 'text'}
    invalid = set(self.modalities) - valid_modalities
    if invalid:
      raise ValueError(
          f"Invalid modalities: {invalid}. Valid options: {valid_modalities}")

  def _get_model_id(self):
    try:
      return self.client.models.list().data[0].id
    except Exception as e:
      raise ConnectionError("Failed to retrieve model ID from API") from e

  @staticmethod
  def make_api_url(host: str, port: int, version: str = "v1") -> str:
    return f"http://{host}:{port}/{version}"

  def cl_custom_chat(
    self,
    prompt: str = "",
    system_prompt: str = "",
    images: List[Image] = [],
    audios: List[Audio] = [],
    videos: List[Video] = [],
    chat_history: List[Dict] = [],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    stream=False,
    **completion_kwargs
  ) -> dict:
    """Process request through OpenAI API."""
    openai_messages = build_messages(
        prompt, system_prompt, images, audios, videos, chat_history)
    
    if completion_kwargs.get("stream"):
      # Force to use usage
      stream_options = completion_kwargs.pop("stream_options", {})
      stream_options.update({"include_usage": True})
      completion_kwargs["stream_options"] = stream_options
    
    response = self.client.chat.completions.create(
        model=self.model_id,
        messages=openai_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=stream,
        **completion_kwargs
      )

    return response

  def chat(
      self,
      **chat_compl_kwargs
  ) -> Union[ChatCompletion, OpenAIStream[ChatCompletionChunk]]:

    chat_compl_kwargs["model"] = self.model_id
    if chat_compl_kwargs.get("stream"):
      # Force to use usage
      stream_options = chat_compl_kwargs.pop("stream_options", {})
      stream_options.update({"include_usage": True})
      chat_compl_kwargs["stream_options"] = stream_options

    try:
      result = self.client.chat.completions.create(
          **chat_compl_kwargs
      )
      return result
    
    except BadRequestError as e:
      body = getattr(e, "body", {})
      msg = body.get("message", str(e)) if body else str(e)
      raise ModelBadRequestError(f"BadRequestError: {msg}")
    except Exception as e:
      raise e
