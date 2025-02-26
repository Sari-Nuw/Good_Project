import cv2
import numpy as np
import math
import os

#Converting from pascal annotation box to YOLO annotation box
def bbox_2_yolo(x1,y1,x2,y2,width,height):

    #Calculating parameters and normalzing
    bbox_width = (x2 - x1)/width
    bbox_height = (y2 - y1)/height

    x_center = x1/width + bbox_width/2
    y_center = y1/height + bbox_height/2

    return x_center,y_center,bbox_width,bbox_height

#Pathway to the saved image and related annotations folder
saved_image_path = r"C:/Users/nuway/OneDrive/Desktop/GOOD_Project/Images"
saved_annotations_path = r"C:\Users\nuway\OneDrive\Desktop\GOOD_Project/Image Annotations"

#Pathways to the output images and annotations
image_results_path = r"C:/Users/nuway/OneDrive/Desktop/GOOD_Project/Result Images/"
full_image_path = image_results_path + "Full Images/"
sectioned_image_path = image_results_path + "Section Images/"
tiled_image_path = image_results_path + "Tiled Images/"
annotations_results_path = r"C:/Users/nuway/OneDrive/Desktop/GOOD_Project/Tiled Annotations/"

# create the result folders
os.makedirs(saved_image_path,exist_ok=True)
os.makedirs(saved_annotations_path,exist_ok=True)
os.makedirs(full_image_path,exist_ok=True)
os.makedirs(sectioned_image_path,exist_ok=True)
os.makedirs(tiled_image_path,exist_ok=True)
os.makedirs(annotations_results_path,exist_ok=True)

#For visualisation of annotation boxes on tiled images
mark_boxes = False

#Opening images. Images should be names img (1), img(2), img(3), etc. and be of .tif format
#Img_num to name tiled images
img_num = 1
for num in range(85,95):

    color_image = cv2.imread(saved_image_path + '/img ({}).tif'.format(num+1))
    copy_image = np.copy(color_image)

    #For tiling
    img_shape = color_image.shape
    tile_size = (640, 640)
    offset = tile_size

    #Pixel coordinates of objects from annotations
    annotation_boxes = []

    #Getting the bounding boxes from annotations
    file = open(saved_annotations_path + "\img ({}).txt".format(num+1))
    lines = file.readlines()
    for line in lines:
    #Saving the annotations and converting to pascal annotation for visualisation
        numbers = line.split()
        if numbers != []:
            #Get class ID
            class_id = numbers[0]
            #Get annotation box in YOLO format
            x_center, y_center, bbox_width, bbox_height = float(numbers[1]), float(numbers[2]), float(numbers[3]), float(numbers[4])
            #Converting from YOLO format to pascal format for the annotation boxes
            width = img_shape[0]
            height = img_shape[1]
            x1 = (x_center - bbox_width/2) * width
            x2 = (x_center + bbox_width/2) * width
            y1 = (y_center - bbox_height/2) *height
            y2 = (y_center + bbox_height/2) *height
            points = [int(x1),int(y1),int(x2),int(y2),class_id]
            #Saving the converted pascal annotations
            annotation_boxes.append(points)
            #Drawing the annotation box on the image
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    #Saving the full image
    cv2.imwrite(full_image_path + 'Image_{}.jpg'.format(num+1),color_image)

    #Tiling each image 
    for i in range(int(math.ceil(img_shape[0]/offset[1]))):
        for j in range(int(math.ceil(img_shape[1]/offset[0]))):

            #To preserve the annotations of each box
            annotation_string = ''

            #Defining the subsection of the image
            x_min = min(offset[0]*i, img_shape[1] - offset[0])
            y_min = min(offset[1]*j, img_shape[0] - offset[1])
            x_max = min(offset[0]*i+tile_size[0], img_shape[1])
            y_max = min(offset[1]*j+tile_size[1], img_shape[0])

            #Opening the annotation text file
            with open(annotations_results_path + 'img ({}).txt'.format(img_num),"w") as file:

                #Saving the full image with the partition lines for visualisation
                save_image = np.copy(color_image)
                cv2.rectangle(save_image, (x_min,y_min),(x_max,y_max),(0,0,255),3)
                cv2.imwrite(sectioned_image_path + 'Image_{}_Section_{}_{}_{}_{}.jpg'.format(num,x_min,y_min,x_max,y_max),save_image)

                #Cropping and saving the images
                clean_image = np.copy(copy_image)
                cropped_image = clean_image[y_min:y_max,x_min:x_max]

                #Width and height of the cropped images
                width = x_max - x_min
                height = y_max - y_min

                #Iterating across each annotation and translating/discarding it for the cropped image
                for box in annotation_boxes:

                    #Translating the annotation box position for the tile
                    shifted_box = [box[0]-i*x_min,box[1]-j*y_min,box[2]-i*x_min,box[3]-j*y_min]

                    #Annotation box completely out of bounds
                    if box[0] > x_max or box[2] < x_min or box[1] > y_max or box[3] < y_min:
                        #Annotation box not in bounds. Skipped.
                        continue

                    #Annotation box completely in bounds
                    elif box[0] >= x_min and box[2] <= x_max and box[1] >= y_min and box[3] <= y_max:
                        #To visualise the tiled image and annotation boxes
                        if mark_boxes:
                            cv2.putText(cropped_image, str(shifted_box), (shifted_box[0], shifted_box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
                            cv2.rectangle(cropped_image, (shifted_box[0], shifted_box[1]), (shifted_box[2], shifted_box[3]), (0, 255, 0), 3)
                        #YOLO information for the translated bounding box
                        x_center,y_center,bbox_width,bbox_height = bbox_2_yolo(shifted_box[0],shifted_box[1],shifted_box[2],shifted_box[3],width,height)

                    #Top of annotation box out of bounds
                    elif box[0] >= x_min and box[2] <= x_max and box[1] <= y_min and box[3] <= y_max:
                        #To visualise the tiled image and annotation boxes
                        if mark_boxes:
                            cv2.putText(cropped_image, str(shifted_box), (shifted_box[0], y_min-j*y_min), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
                            cv2.rectangle(cropped_image, (shifted_box[0], y_min-j*y_min), (shifted_box[2], shifted_box[3]), (0, 255, 0), 3)
                        #YOLO information for the translated bounding box
                        x_center,y_center,bbox_width,bbox_height = bbox_2_yolo(shifted_box[0],y_min-j*y_min,shifted_box[2],shifted_box[3],width,height)

                    #Bottom of annotation box out of bounds
                    elif box[0] >= x_min and box[2] <= x_max and box[1] >= y_min and box[3] >= y_max:
                        #To visualise the tiled image and annotation boxes
                        if mark_boxes:
                            cv2.putText(cropped_image, str(shifted_box), (shifted_box[0], shifted_box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
                            cv2.rectangle(cropped_image, (shifted_box[0], shifted_box[1]), (shifted_box[2], y_max-j*y_min), (0, 255, 0), 3)
                        #YOLO information for the translated bounding box
                        x_center,y_center,bbox_width,bbox_height = bbox_2_yolo(shifted_box[0],shifted_box[1],shifted_box[2],y_max-j*y_min,width,height)

                    #Left of annotation box out of bounds
                    elif box[0] <= x_min and box[2] <= x_max and box[1] >= y_min and box[3] <= y_max:
                        #To visualise the tiled image and annotation boxes
                        if mark_boxes:
                            cv2.putText(cropped_image, str(shifted_box), (x_min-i*x_min, shifted_box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
                            cv2.rectangle(cropped_image, (x_min-i*x_min, shifted_box[1]), (shifted_box[2], shifted_box[3]), (0, 255, 0), 3)
                        #YOLO information for the translated bounding box
                        x_center,y_center,bbox_width,bbox_height = bbox_2_yolo(x_min-i*x_min,shifted_box[1],shifted_box[2],shifted_box[3],width,height)

                    #Right of annotation box out of bounds
                    elif box[0] >= x_min and box[2] >= x_max and box[1] >= y_min and box[3] <= y_max:
                        #To visualise the tiled image and annotation boxes
                        if mark_boxes:
                            cv2.putText(cropped_image, str(shifted_box), (shifted_box[0], shifted_box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
                            cv2.rectangle(cropped_image, (shifted_box[0], shifted_box[1]), (x_max-i*x_min, shifted_box[3]), (0, 255, 0), 3)
                        #YOLO information for the translated bounding box
                        x_center,y_center,bbox_width,bbox_height = bbox_2_yolo(shifted_box[0],shifted_box[1],x_max-i*x_min,shifted_box[3],width,height)

                    #Top/Left of annotation box out of bounds
                    elif box[0] <= x_min and box[2] <= x_max and box[1] <= y_min and box[3] <= y_max:
                        #To visualise the tiled image and annotation boxes
                        if mark_boxes:
                            cv2.putText(cropped_image, str(shifted_box), (x_min-i*x_min, y_min-j*y_min), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
                            cv2.rectangle(cropped_image, (x_min-i*x_min, y_min-j*y_min), (shifted_box[2], shifted_box[3]), (0, 255, 0), 3)
                        #YOLO information for the translated bounding box
                        x_center,y_center,bbox_width,bbox_height = bbox_2_yolo(x_min-i*x_min, y_min-j*y_min,shifted_box[2],shifted_box[3],width,height)

                    #Top/Right of annotation box out of bounds
                    elif box[0] >= x_min and box[2] >= x_max and box[1] <= y_min and box[3] <= y_max:
                        #To visualise the tiled image and annotation boxes
                        if mark_boxes:
                            cv2.putText(cropped_image, str(shifted_box), (shifted_box[0], y_min-j*y_min), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
                            cv2.rectangle(cropped_image, (shifted_box[0], y_min-j*y_min), (x_max-i*x_min, shifted_box[3]), (0, 255, 0), 3)
                        #YOLO information for the translated bounding box
                        x_center,y_center,bbox_width,bbox_height = bbox_2_yolo(shifted_box[0],y_min-j*y_min,x_max-i*x_min,shifted_box[3],width,height)

                    #Bottom/Left of annotation box out of bounds
                    elif box[0] <= x_min and box[2] <= x_max and box[1] >= y_min and box[3] >= y_max:
                        #To visualise the tiled image and annotation boxes
                        if mark_boxes:
                            cv2.putText(cropped_image, str(shifted_box), (x_min-i*x_min, shifted_box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
                            cv2.rectangle(cropped_image, (x_min-i*x_min, shifted_box[1]), (shifted_box[2], y_max-j*y_min), (0, 255, 0), 3)
                        #YOLO information for the translated bounding box
                        x_center,y_center,bbox_width,bbox_height = bbox_2_yolo(x_min-i*x_min,shifted_box[1],shifted_box[2],y_max-j*y_min,width,height)
                        
                    #Bottom/Right of annotation box out of bounds
                    elif box[0] >= x_min and box[2] >= x_max and box[1] >= y_min and box[3] >= y_max:
                        #To visualise the tiled image and annotation boxes
                        if mark_boxes:
                            cv2.putText(cropped_image, str(shifted_box), (shifted_box[0], shifted_box[1]), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3, cv2.LINE_AA)
                            cv2.rectangle(cropped_image, (shifted_box[0], shifted_box[1]), (x_max-i*x_min, y_max-j*y_min), (0, 255, 0), 3) 
                        #YOLO information for the translated bounding box
                        x_center,y_center,bbox_width,bbox_height = bbox_2_yolo(shifted_box[0],shifted_box[1],x_max-i*x_min, y_max-j*y_min,width,height)

                    #Adding the next annotation to be saved
                    annotation_string = annotation_string + '{} {} {} {} {}\n'.format(box[4],x_center,y_center,bbox_width,bbox_height)        
                    
                # Saving the tiled images and their annotations
                file.write(annotation_string)
                cv2.imwrite(tiled_image_path + 'img ({}).jpg'.format(img_num), cropped_image)
                img_num += 1