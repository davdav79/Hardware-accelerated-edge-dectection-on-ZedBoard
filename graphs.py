import numpy as np
import matplotlib.pyplot as plt


# gaussian blur results
sigma = np.linspace(0, 10, 11)

values = [0.105, 0.431, 0.554, 0.610, 0.452, 0.328, 0.282, 0.201, 0.189, 0.180, 0.204]

plt.plot(sigma, values, label='Canny', color='#ff7f0e')
plt.xlabel('Sigma')
plt.ylabel('F1-Wert')
plt.title('F1-Wert gegen Sigma')
plt.legend()
plt.show()


# threshold results
threshold = np.linspace(0, 290, 30)
values_3 = [0.020, 0.021, 0.027, 0.039, 0.064, 0.112, 0.189, 0.282, 0.354, 0.398, 0.421, 0.441, 0.463, 
            0.488, 0.516, 0.549, 0.586, 0.626, 0.663, 0.700, 0.719, 0.690, 0.593, 0.443, 0.295, 0.159,
            0.060, 0.015, 0.002, 0.000]

"""
plt.plot(threshold, values_3, label='Sobel')
plt.xlabel('Schwellwert')
plt.ylabel('F1-Wert')
plt.title('F1-Wert gegen Schwellwert')
plt.legend()
plt.show()
"""

noise = np.linspace(0, 100, 11)
sobel = [0.999, 0.975, 0.874, 0.685, 0.436, 0.274, 0.128, 0.067, 0.033, 0.020, 0.022]
canny = [0.998, 0.632, 0.454, 0.358, 0.212, 0.122, 0.074, 0.037, 0.017, 0.010, 0.011]

"""
plt.plot(noise, sobel, label='Sobel')
plt.plot(noise, canny, label='Canny')
plt.xlabel('Rauschniveau')
plt.ylabel('F1-Wert')
plt.title('F1-Wert gegen Rauschniveau')
plt.legend()
plt.show()
"""
