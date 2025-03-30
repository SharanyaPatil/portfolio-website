import cv2
import numpy as np

# Load a lunar image
image = cv2.imread(r'C:\Users\z00522pb\projects\Sharanya Project\robustness_analysis_tool-rat_modules\crater_3.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate gradients in x and y directions
gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Approximate depth map by combining gradients
depth_map = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

# Normalize the depth map for visualization
depth_map_normalized = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# depth_map_color = cv2.applyColorMap((depth_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)

# # Display the color depth map
# cv2.imshow('Color Depth Map (Plasma)', depth_map_color)
depth_map_smoothed = cv2.GaussianBlur(depth_map, (5, 5), 0)  # Adjust kernel size (5, 5) for desired smoothing

# Normalize the smoothed depth map to 0-1 range
depth_map_normalized = cv2.normalize(depth_map_smoothed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# Apply the plasma colormap
depth_map_color = cv2.applyColorMap((depth_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
cv2.imwrite('depth_map.jpg', depth_map_color)
cv2.imshow('Color Depth Map (Plasma)', depth_map_color)
# Display the depth map
cv2.imshow('Depth Map', depth_map_normalized)



cv2.waitKey(0)
cv2.destroyAllWindows()
##########################################################################
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Load the depth map (grayscale image)
# depth_map = cv2.imread('predicted_depth_map.jpg', cv2.IMREAD_GRAYSCALE)

# # Resize the depth map if needed
# depth_map_resized = cv2.resize(depth_map, (256, 256))  # Ensure manageable size for plotting

# # Normalize depth values to a range (0, 1) for better visualization
# depth_map_normalized = depth_map_resized / 255.0

# # Generate X and Y coordinates for plotting
# h, w = depth_map_normalized.shape
# x = np.linspace(0, 1, w)
# y = np.linspace(0, 1, h)
# x, y = np.meshgrid(x, y)

# # Create a 3D plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot the surface
# ax.plot_surface(x, y, depth_map_normalized, cmap='plasma', edgecolor='none')

# # Set plot attributes
# ax.set_title('3D Surface from Depth Map')
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Depth')
# ax.view_init(elev=45, azim=135)  # Adjust the view angle

# # Show the plot
# plt.show()