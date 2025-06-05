# project
## Aim
To write a python program using OpenCV to do the following image manipulations.
i) Extract ROI from  an image.
ii) Perform handwritting detection in an image.
iii) Perform object detection with label in an image.
## Software Required:
Anaconda - Python 3.7
## Algorithm:
## I)Perform ROI from an image
### Step1:
Import necessary packages 
### Step2:
Read the image and convert the image into RGB
### Step3:
Display the image
### Step4:
Set the pixels to display the ROI 
### Step5:
Perform bit wise conjunction of the two arrays  using bitwise_and 
### Step6:
Display the segmented ROI from an image.

```
# Step 1: Import necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Read the image and convert the image into RGB
image = cv2.imread('your_image.jpg')  # Replace with your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 3: Display the image
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Step 4: Set the pixels to display the ROI
# Define the ROI coordinates - example rectangle (x, y, width, height)
x, y, w, h = 100, 100, 200, 200  # Change values as needed

# Create a mask of zeros (black image) with the same size as the input image
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Fill the ROI area in the mask with white (255)
mask[y:y+h, x:x+w] = 255

# Step 5: Perform bitwise conjunction of the two arrays using bitwise_and
roi = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Step 6: Display the segmented ROI from an image
plt.imshow(roi)
plt.title('Segmented ROI')
plt.axis('off')
plt.show()
```

## II)Perform handwritting detection in an image
### Step1:
Import necessary packages 
### Step2:
Define a function to read the image,Convert the image to grayscale,Apply Gaussian blur to reduce noise and improve edge detection,Use Canny edge detector to find edges in the image,Find contours in the edged image,Filter contours based on area to keep only potential text regions,Draw bounding boxes around potential text regions.
### Step3:
Display the results.

```
# Step 1: Import necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Define the function for handwriting detection
def detect_handwriting(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or path is incorrect.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector to find edges
    edged = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (adjust min_area as needed)
    min_area = 100  # minimum contour area to be considered handwriting
    handwriting_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Draw bounding boxes around potential handwriting regions
    output = image.copy()
    for cnt in handwriting_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image, gray, edged, output

# Step 3: Display the results
image_path = 'handwriting_sample.jpg'  # Replace with your image path
original, gray, edges, detected = detect_handwriting(image_path)

if original is not None:
    plt.figure(figsize=(15,10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges Detected (Canny)')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
    plt.title('Handwriting Detection')
    plt.axis('off')

    plt.show()

```

## III)Perform object detection with label in an image
### Step1:
Import necessary packages 
### Step2:
Set and add the config_file,weights to ur folder.
### Step3:
Use a pretrained Dnn model (MobileNet-SSD v3)
### Step4:
Create a classLabel and print the same
### Step5:
Display the image using imshow()
### Step6:
Set the model and Threshold to 0.5
### Step7:
Flatten the index,confidence.
### Step8:
Display the result.

```
# Step 1: Import necessary packages
import cv2
import numpy as np

# Step 2: Load config file and weights (make sure these files are in your working directory)
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Path to .pbtxt file
weights_file = 'frozen_inference_graph.pb'                   # Path to .pb file

# Step 3: Load the pretrained MobileNet-SSD v3 model
net = cv2.dnn_DetectionModel(weights_file, config_file)

# Step 4: Class labels of COCO dataset used by MobileNet SSD v3
classLabels = []
with open('coco.names', 'rt') as f:
    classLabels = f.read().rstrip('\n').split('\n')

print("Class Labels:")
print(classLabels)

# Step 5: Read and display the image
image = cv2.imread('input.jpg')  # Replace with your image path

if image is None:
    print("Error: Image not found or path is incorrect.")
    exit()

cv2.imshow("Input Image", image)
cv2.waitKey(0)

# Step 6: Set up the model and confidence threshold
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
threshold = 0.5

# Step 7: Perform detection
classIds, confidences, boxes = net.detect(image, confThreshold=threshold)

# Flatten and prepare data for display
if len(classIds) != 0:
    for classId, confidence, box in zip(classIds.flatten(), confidences.flatten(), boxes):
        label = f'{classLabels[classId - 1]}: {confidence:.2f}'
        # Draw bounding box and label on the image
        cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
        cv2.putText(image, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Step 8: Display the result
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

