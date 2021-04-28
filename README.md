# noise_gating
IDL and Python code for noise gating

Code for image noise removal technique based on Craig DeForest's suggestion: https://ui.adsabs.harvard.edu/abs/2017ApJ...838..155D/abstract


# Basis usage

This demonstrates the basic usage for the python version:

```python
from noise_gate import noise_gate

# data is 3d ndarray
out = noise_gate(data)

```

The python version should work with gpu's through the use of cupy.