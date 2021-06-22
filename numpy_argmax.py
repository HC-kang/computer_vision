import numpy as np
x = np.array([
    [0.1, 0.8, 0.1],
    [0.3, 0.1, 0.6],
    [0.2, 0.5, 0.3],
    [0.8, 0.1, 0.1]
])

y = np.argmax(x)
# y = 1

y0 = np.argmax(x, axis = 0)
# y0 = [3,0,1]

y1 = np.argmax(x, axis = 1)
# y1 = [1,2,1,0]

y2 = np.argmax(x, axis=-1)
y2