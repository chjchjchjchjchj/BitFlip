import matplotlib.pyplot as plt
import numpy as np

x = [i for i in range(100)]
y = [i for i in range(5, 105)]
print(x)
print(y)
figure = plt.figure()
plt.plot(x, y)
plt.show()
plt.savefig("caogao.pdf")