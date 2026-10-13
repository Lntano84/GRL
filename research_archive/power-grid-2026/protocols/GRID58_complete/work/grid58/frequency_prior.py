"""Exact zero-residual STATIC_ID order from public canonical representatives."""
import numpy as np
def frequency_logs(representatives,size):
    order=sorted(map(int,representatives));assert len(set(order))==len(order)
    logs=np.full(size,-1e9,dtype=np.float32)
    logs[order]=-np.log1p(np.arange(len(order),dtype=np.float32))
    return logs
