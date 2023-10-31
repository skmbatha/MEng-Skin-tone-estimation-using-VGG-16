import matplotlib.pyplot as plt
import numpy as np

# Create the sequence of numbers
y = [0.0, 7.1, 13.9, 20.73, 29.78, 36.85, 43.69, 50.71, 55.3, 60.7, 66.19, 69.65, 73.58, 77.23, 81.11, 84.56, 86.88, 89.35, 90.86, 92.38]
x = np.linspace(0, 1.9, num=20)

print(x)

# Create a plot
plt.plot(x, y, marker='o', linestyle='-')

# Highlight the point at x=0.5
plt.scatter(0.5, y[5], c='#ffffff', marker='o', label=f'y={y[5]} @ x={0.5}')
plt.scatter(1, y[10], c='#ffffff', marker='o', label=f'y={y[10]} @ x={1}')
plt.scatter(1.9, y[19], c='#ffffff', marker='o', label=f'y={y[19]} @ x={2}')

# Add labels to the axes
plt.xlabel('L2 Distance from target')
plt.ylabel('Percentage accuracy')

# Add a title to the plot
plt.title('Cumulative accuracy')

# Display the plot
plt.legend()
plt.show()
