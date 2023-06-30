import cv2
import numpy as np

# Read image
image_path = "/media/stannyho/ssd/rc-pda/data_own/img/frame000000.png"  # Replace with the path to your image file
image = cv2.imread(image_path)

# Read radar points from .pcd file
pcd_file = "/media/stannyho/ssd/rc-pda/data_own/pcd/1684075681.146979066.pcd"
with open(pcd_file, 'rb') as file:
    data = file.read()

# Find the start index of the point data
header_end = data.find(b"DATA") + len(b"DATA")

# Extract radar points from the PCD file
radar_points = np.frombuffer(data[header_end:], dtype=np.float32)
num_points = len(radar_points) // 4
print(num_points)
print(radar_points)
radar_points = radar_points.reshape(num_points, 4)[:, :3]  # Keep only x, y, z coordinates

# Project radar points onto image
for point in radar_points:
    x, y, _ = point
    image_x = int(x)
    image_y = int(y)
    cv2.circle(image, (image_x, image_y), radius=5, color=(0, 0, 255), thickness=-1)

# Display the image with radar points
cv2.imshow("Image with Radar Points", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
