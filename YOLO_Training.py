from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolov8m.pt") 
#Load a saved model

# Use the model
if __name__ == '__main__':
    # Train the model
    results = model.train(data=r"C:\Users\nuway\OneDrive\Desktop\GOOD_Project\config.yml", epochs=30, patience = 0)
