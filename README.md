# lcm_test

Quick and dirty Streamlit UI for [Latent Consistency Models](https://github.com/luosiallen/latent-consistency-model).

## Usage

Tested on an M2 Mac, so YMMV, but:

```
pip install -r requirements.txt
streamlit run lcm_test.py
```

## Configuration

* If you want to force a given Torch device, set the `LCM_DEVICE` envvar to e.g. `cpu`.
* If you want to force using fp16 precision, set the `LCM_FP16` envvar to anything truthy. This might not work.
