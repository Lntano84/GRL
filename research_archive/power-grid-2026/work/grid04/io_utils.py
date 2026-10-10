"""Strict numeric JSON for NumPy telemetry; never silently stringify unknown data."""
import json
import numpy as np
def python_scalar(value):
    if isinstance(value,np.generic): return value.item()
    raise TypeError(f'Unsupported JSON telemetry type: {type(value).__name__}')
def encode(value,**kwargs):
    return json.dumps(value,ensure_ascii=False,allow_nan=False,default=python_scalar,**kwargs)
