# {model_name}

[Model source](https://huggingface.co/{hf_model_id})

Model is serving with `{inference_framework}`.

Input data type: `{input_data_type}`

# Usage

## Set your PAT
Export your PAT as an environment variable. Then, import and initialize the API Client.

Find your PAT in your security settings.

* Linux/Mac: `export CLARIFAI_PAT="your personal access token"`

* Windows (Powershell): `$env:CLARIFAI_PAT="your personal access token"`

## Running the API with Clarifai's Python SDK


```python
# Please run `pip install -U clarifai` before running this script

from clarifai.client import Model
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai.client.input import Inputs

model = Model(url="{model_url}")

prompt = "What time of day is it?"
image_url = "https://samples.clarifai.com/metro-north.jpg"



input_data = Inputs.get_multimodal_input(input_id="",image_url=image_url, raw_text=prompt)
inference_params = dict(temperature=0.2, max_tokens=100) # Optional

results = model.generate([input_data], inference_params=inference_params)

for res in results:
  if res.status.code == status_code_pb2.SUCCESS:
    print(res.outputs[0].data.text.raw, end='', flush=True)
```

# Server extra args

```
{server_args}
```
