# {model_name}

[Model source](https://huggingface.co/{hf_model_id})

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


model = Model(url="{model_url}")
prompt = "What’s the future of AI?"

results = model.generate_by_bytes(prompt.encode("utf-8"), "text")

for res in results:
  if res.status.code == status_code_pb2.SUCCESS:
    print(res.outputs[0].data.text.raw, end='', flush=True)
```