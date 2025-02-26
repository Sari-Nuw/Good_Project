import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO
from osgeo import gdal
import rasterio
import rasterio.features
import rasterio.warp
import os

def bbox_2_yolo(x1,y1,x2,y2,width,height):

    #Calculating parameters and normalzing
    bbox_width = (x2 - x1)/width
    bbox_height = (y2 - y1)/height

    x_center = x1/width + bbox_width/2
    y_center = y1/height + bbox_height/2

    return x_center,y_center,bbox_width,bbox_height

#Path to images for prediction
image_path = r"C:/Users/nuway/OneDrive/Desktop/GOOD_Project/Images"

#Path to result images
results_path = r"C:/Users/nuway/OneDrive/Desktop/GOOD_Project/Prediction Results/"

#Path to object detection model
model_path = r'runs\detect\train7\weights\best.pt'

# create the result folders
os.makedirs(image_path,exist_ok=True)
os.makedirs(results_path,exist_ok=True)

# Load a model
model = YOLO(model_path)

#Setting threshold for object detection 
threshold = 0.1

#Reading image
for num in range(86,96):
    with rasterio.open(image_path + '\img ({}).tif'.format(num+1)) as image_file:

        mask = image_file.dataset_mask()
        
        # Extract feature shapes and values from the array.
        # for geom, val in rasterio.features.shapes(mask, transform=image_file.transform):

        #     # Transform shapes from the dataset's own coordinate
        #     # reference system to CRS84 (EPSG:4326).
        #     geom = rasterio.warp.transform_geom(image_file.crs, 'EPSG:4326', geom, precision=15)

            # Print GeoJSON shapes to stdout.
            # print('geom')
            # print(geom['coordinates'][0])
            # print('val')
            # print(val)

        #Gets the color bands from the images and also the binary of the image (whether or not data is available)
        r,g,b = image_file.read()
        color_image = np.dstack((r,g,b))
        copy_image = np.copy(color_image)

        #Getting results of object detection
        results = model(color_image)[0]

        #Pixel coordinates of objects from predictions
        prediction_boxes = []

        #Saving the position of the object boxes from predictions
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                #Storing all the predicted boxes
                points = [int(x1),int(y1),int(x2),int(y2)]
                prediction_boxes.append(points)
                #To see bounding box around detected object
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)

        #Printing object pixel coordinates 
        print('boxes')      
        print(prediction_boxes)

        #Show the image with bounding boxes
        # plt.imshow(color_image)
        # plt.show()

        #Saving the images
        cv2.imwrite(results_path + 'Image_{}.jpg'.format(num+1),cv2.cvtColor(color_image,cv2.COLOR_BGR2RGB))


