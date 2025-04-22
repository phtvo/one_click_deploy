# $model_name

[Model source](https://huggingface.co/$hf_model_id)

Model is serving with `$inference_framework`.

Input data type: `$input_data_type`

# Usage

## Set your PAT
Export your PAT as an environment variable. Then, import and initialize the API Client.

Find your PAT in your security settings.

* Linux/Mac: `export CLARIFAI_PAT="your personal access token"`

* Windows (Powershell): `$$env:CLARIFAI_PAT="your personal access token"`

## Running the API with Clarifai's Python SDK


```python
# Please run `pip install -U clarifai` before running this script

from clarifai.client import Model
from clarifai.runners.utils.data_types import Image


model = Model(url="$model_url")

prompt = "What time of day is it?"
image_url = "https://samples.clarifai.com/metro-north.jpg"


# clarifai style prediction method
## Stream
generated_text = model.generate(prompt=prompt, images=[Image()])
for each in generated_text:
    print(each, end='', flush=True)
## Non stream
generated_text = model.predict(prompt=prompt)
print(generated_text)

# openai chat completion method
conversion = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {'type': 'text', 'text': prompt},
        {'type': 'image_url', 'image_url': {'url': image_url}}
      ]
    },
    # Continue adding messages as the conversation progresses
]
## Stream
stream_generated_text = model.stream_chat(messages=conversion)
for chunk in stream_generated_text:
  # chunk is dict ChatCompletionChunk format
  if chunk["choices"]:
    text = chunk["choices"][0]["delta"].get("content", "")
    print(text, end='', flush=True)

## Non stream
generated_text = model.chat(messages=conversion) # dict of ChatCompletion format
print(generated_text["choices"][0]["message"]["content"])
```

# Server extra args

```
${server_args}
```
