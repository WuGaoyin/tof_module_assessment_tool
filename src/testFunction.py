import function
import numpy as np

#a=np.arange(640*480).reshape(640,480)
a=np.ndarray((640,480))
a=a+2
b=function.cacSpatialNoise(a)
print(b)