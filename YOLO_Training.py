from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolov8l.pt") 
#Load a saved model

# Use the model
if __name__ == '__main__':
    # Train the model
    results = model.train(data=r"C:\Users\nuway\OneDrive\Desktop\Realsense Project\GOOD Project\config.yml", epochs=10, patience = 0)
