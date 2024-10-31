
import cv2
import os

tiled_image_path = r"C:/Users/nuway/OneDrive/Desktop/GOOD_Project/Result Images/Tiled Images"
results_path = r"C:/Users/nuway/OneDrive/Desktop/GOOD_Project/Result Images/Annotated Tiles/"
text_path = r"C:/Users/nuway/OneDrive/Desktop/GOOD_Project/Tiled Annotations/"

# create the result folders
os.makedirs(tiled_image_path,exist_ok=True)
os.makedirs(results_path,exist_ok=True)
os.makedirs(text_path,exist_ok=True)

#Reading image
for num in range(28):

    color_image = cv2.imread(tiled_image_path + '/img ({}).jpg'.format(num+1))

    cv2.imshow('image',color_image)
    cv2.waitKey(0)

    img_shape = color_image.shape

    #Pixel coordinates of objects from annotations
    annotation_boxes = []

    #Getting the bounding boxes from annotations
    file = open(text_path + "img ({}).txt".format(num+1))
    lines = file.readlines()
    for line in lines:
    #Saving the annotations and converting to pascal annotation (upper left and bottom right of rectangle)
        numbers = line.split()
        if numbers != []:
            class_id = numbers[0]
            x_center, y_center, bbox_width, bbox_height = float(numbers[1]), float(numbers[2]), float(numbers[3]), float(numbers[4])
            width = color_image.shape[0]
            height = color_image.shape[1]
            x1 = (x_center - bbox_width/2) * width
            x2 = (x_center + bbox_width/2) * width
            y1 = (y_center - bbox_height/2) *height
            y2 = (y_center + bbox_height/2) *height
            points = [int(x1),int(y1),int(x2),int(y2),class_id]
            annotation_boxes.append(points)
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    #Saving the full image
    cv2.imwrite(results_path + 'Image_{}.jpg'.format(num+1),color_image)