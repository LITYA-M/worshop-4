## Coin-Detection-using-OpenCV-in-Python
# AIM
To detect and count the total number of coins present in an image using OpenCV morphological operations, thresholding, and SimpleBlobDetector.

# ALGORITHM
Step 1 — Read the Image

Step 2 — Convert to Grayscale

Step 3 — Split into B, G and R Channels

Step 4 — Perform Thresholding

Step 5 — Perform Morphological Operations

# PROGRAM
Developed by LITYA M REG NO:- 212225230152
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read image
image = cv2.imread("seashell.jpg")

# Display original image
imageCopy = image.copy()
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")
plt.show()


# Step 2: Convert image to grayscale
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12, 12))
plt.subplot(121)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")

plt.subplot(122)
plt.imshow(imageGray, cmap="gray")
plt.title("Grayscale Image")
plt.show()


# Step 3: Split image into B, G and R channels
imageB, imageG, imageR = cv2.split(image)

plt.figure(figsize=(20, 12))
plt.subplot(141)
plt.imshow(image[:, :, ::-1])
plt.title("Original Image")

plt.subplot(142)
plt.imshow(imageB, cmap="gray")
plt.title("Blue Channel")

plt.subplot(143)
plt.imshow(imageG, cmap="gray")
plt.title("Green Channel")

plt.subplot(144)
plt.imshow(imageR, cmap="gray")
plt.title("Red Channel")

plt.show()


# Step 4: Thresholding
_, imageThreshold = cv2.threshold(
    imageG,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

plt.imshow(imageThreshold, cmap="gray")
plt.title("Thresholded Image")
plt.show()


# Step 5: Morphological operations
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

imageDilated = cv2.dilate(
    imageThreshold,
    kernel,
    iterations=1
)

imageDilated2 = cv2.dilate(
    imageThreshold,
    kernel,
    iterations=2
)

plt.imshow(imageDilated2, cmap="gray")
plt.title("Dilated Image Iteration 2")
plt.show()


imageEroded = cv2.erode(
    imageDilated2,
    kernel,
    iterations=1
)

plt.imshow(imageEroded, cmap="gray")
plt.title("Eroded Image")
plt.show()


# Step 6: Create SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0
params.minDistBetweenBlobs = 2

# Filter by Area
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

detector = cv2.SimpleBlobDetector_create(params)


# Step 7: Detect blobs
keypoints = detector.detect(imageEroded)

# Draw detected coins
output = cv2.drawKeypoints(
    image,
    keypoints,
    np.array([]),
    (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(10, 10))
plt.imshow(output[:, :, ::-1])
plt.title("Final Shell Detection")
plt.axis("off")
plt.show()


# Print number of detected coins
print(f"Number of shells detected: {len(keypoints)}")
```
# Output
<img width="732" height="482" alt="Screenshot 2026-09-16 213340" src="https://github.com/user-attachments/assets/ed0d4580-76ac-4076-b66c-0f08ba1944a5" />

<img width="1301" height="440" alt="Screenshot 2026-09-16 213359" src="https://github.com/user-attachments/assets/b2d4f38e-23e0-4ce5-97d4-5c43579663b9" />
<img width="1417" height="275" alt="Screenshot 2026-09-16 213421" src="https://github.com/user-attachments/assets/f8a162fb-e322-49c5-bbd9-220a545937b7" />

<img width="845" height="488" alt="Screenshot 2026-09-16 213448" src="https://github.com/user-attachments/assets/897ec626-b64a-4476-ab01-ef86ef5db7fa" />
<img width="812" height="507" alt="Screenshot 2026-09-16 213523" src="https://github.com/user-attachments/assets/d1146408-9096-46e9-b8b9-93b9ab9deb1a" />

<img width="772" height="500" alt="Screenshot 2026-09-16 213541" src="https://github.com/user-attachments/assets/03dc1261-4f91-4197-b590-9d9a9cda6a07" />
<img width="1167" height="733" alt="Screenshot 2026-09-16 213603" src="https://github.com/user-attachments/assets/de479ea1-36b8-4f61-ac2c-9e92447240e9" />

# RESULT
Thus, Coin Detection using OpenCV in Python is executed successfully.


