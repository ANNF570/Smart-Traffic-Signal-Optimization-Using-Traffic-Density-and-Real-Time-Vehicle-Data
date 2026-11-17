from roboflow import Roboflow

rf = Roboflow(api_key="rf_N47E2cazJLQtCAFzHnbEVBzOpZm1")  # paste your key from roboflow.com

# Indian Traffic Dataset with multiple vehicle classes
project = rf.workspace("indian-traffic").project("indian-traffic-detection")
dataset = project.version(1).download("yolov8", location="./dataset")

print("Dataset downloaded!")
print("Location: ./dataset")