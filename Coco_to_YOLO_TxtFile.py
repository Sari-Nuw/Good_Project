import numpy as np

# Convert Coco bounding_box to Yolo
def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

#Size of images being converted
width = 640
height = 640

with open(r'C:\Users\nuway\OneDrive\Desktop\Good_Project\Image Annotations\instances_default.json', 'r') as openfile:

	# Reading from json file
	object = openfile.read()

#To store the image_id of the bounding boxes
image_id = []

#Extracting image id's from the json file
index = 0
while index < len(object):
	index = object.find('"image_id":', index)
	if index == -1:
		break
	#print('found at: ', index)
	index = index + 11
	split_index = object.find(',',index)
	#print('split at: ',split_index)
	id = object[index:split_index]
	#print(id)
	image_id.append(id)

#To store the bouding box values
boxes = []

#Getting the bounding box values and splitting each number
index = 0
while index < len(object):
	index = object.find('"bbox":[', index)
	if index == -1:
		break
	#print('found at: ', index)
	index = index + 8
	split_index = object.find(']',index)
	#print('split at: ',split_index)
	bounding = object[index:split_index]
	#print(bounding)
	bounding = bounding.split(',')
	boxes.append(bounding)

#Converting from string to float for mathematical manipulation 
for i in range(len(boxes)):
	for j in range(4):
		boxes[i][j] = boxes[i][j].replace(',','')
		boxes[i][j] = float(boxes[i][j])

#Storing the boundning box values in YOLO format
YOLO_bbox = []

#Converting the bounding boxes to YOLO format
for i in range(len(boxes)):
		YOLO_bbox.append(coco_to_yolo(boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3],width,height))

#Storing all the boxes for each image in its respective text file
total_boxes = 0
for i in range (97):
	with open (r'C:\Users\nuway\OneDrive\Desktop\Good_Project\Image Annotations\train_{}.txt'.format(i+1),'w') as file:
		while int(image_id[total_boxes]) == i+1:
			file.write(('0 '+str(YOLO_bbox[total_boxes][0])+' '+str(YOLO_bbox[total_boxes][1])+' '+str(YOLO_bbox[total_boxes][2])+' '+str(YOLO_bbox[total_boxes][3])+'\n'))
			total_boxes += 1