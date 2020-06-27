# Chamfer Distance for pyTorch

This is an implementation of the Chamfer Distance as a module for pyTorch. It is written as a custom C++/CUDA extension.

As it is using pyTorch's [JIT compilation](https://pytorch.org/tutorials/advanced/cpp_extension.html), there are no additional prerequisite steps that have to be taken. Simply import the module as shown below; CUDA and C++ code will be compiled on the first run.

### Usage
```python
from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

#...
# points and points_reconstructed are n_points x 3 matrices

dist1, dist2 = chamfer_dist(points, points_reconstructed)
loss = (torch.mean(dist1)) + (torch.mean(dist2))


#...
```

### Integration
This code has been integrated into the [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) library for 3D Deep Learning by NVIDIAGameWorks. You should probably take a look at it if you are working on anything 3D :)
