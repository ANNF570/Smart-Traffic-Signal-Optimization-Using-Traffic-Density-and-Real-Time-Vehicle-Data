from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_REAL_API_KEY")
project = rf.workspace("YOUR_REAL_WORKSPACE").project("YOUR_PROJECT_NAME")
dataset = project.version(1).download("yolov8")