# Smart Traffic Signal Optimization Using Traffic Density and Real-Time Vehicle Data

An AI-powered smart traffic management project that uses computer vision, vehicle detection, object tracking, and traffic density analysis to support intelligent traffic signal optimization.

## Project Overview

Traditional traffic signal systems often operate using fixed time intervals, which may not reflect real-time traffic conditions. This project explores an intelligent approach using YOLOv8 and DeepSORT to detect, track, and count vehicles from traffic video streams.

The system analyzes traffic conditions and provides a foundation for more adaptive traffic management.

## Key Features

- **Vehicle Detection:** Detects vehicles using the YOLOv8 object detection model.
- **Object Tracking:** Uses DeepSORT to track vehicles across video frames.
- **Vehicle Counting:** Counts detected and tracked vehicles.
- **Traffic Density Analysis:** Analyzes vehicle density to understand traffic conditions.
- **Emergency Vehicle Detection:** Includes emergency vehicle detection functionality.
- **Siren Detection:** Includes audio-based siren detection functionality.
- **Interactive Dashboard:** Uses Streamlit to provide an interface for traffic monitoring and analysis.
- **Traffic Data Visualization:** Displays traffic-related information through visualizations.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8 | Vehicle detection |
| DeepSORT | Object tracking |
| OpenCV | Video processing |
| Streamlit | Interactive dashboard |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Plotly | Data visualization |
| EasyOCR | Optical character recognition |
| Librosa | Audio feature extraction |

## Project Structure

```text
Smart-Traffic-Signal-Optimization/
│
├── YOLOv8-DeepSORT-Streamlit/
│   ├── app.py
│   ├── requirements.txt
│   ├── SmartTraffic/
│   ├── traffic_detection/
│   ├── ultralytics_local/
│   ├── siren_model.pkl
│   └── label_encoder.pkl
│
├── G1-REPORT(TRAFFIC -SYSTEM).docx
├── Smart Traffic Signal Optimization Using Traffic Density and Real-Time Vehicle Data.pdf
├── Smart Traffic Signal Optimization Using Traffic Density and Real-Time Vehicle Data (IJRCAIT).pdf
└── yolov8n.pt
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/ANNF570/Smart-Traffic-Signal-Optimization-Using-Traffic-Density-and-Real-Time-Vehicle-Data.git
```

### 2. Navigate to the application directory

```bash
cd Smart-Traffic-Signal-Optimization-Using-Traffic-Density-and-Real-Time-Vehicle-Data/YOLOv8-DeepSORT-Streamlit
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to access the dashboard.

## Research and Documentation

This repository includes the project report and research publication:

- **Project Report:** `G1-REPORT(TRAFFIC -SYSTEM).docx`
- **Research Paper:** Smart Traffic Signal Optimization Using Traffic Density and Real-Time Vehicle Data
- **Publication:** IJRCAIT paper PDF included in the repository.

## Project Contributions

This is an academic project developed as part of the B.Tech Computer Science and Engineering program.

The project involves working with computer vision, machine learning, object tracking, traffic data analysis, and interactive dashboard development.

## Future Enhancements

- Integration with real-time traffic camera feeds.
- Improved adaptive traffic signal timing.
- Priority-based emergency vehicle routing.
- Real-time traffic monitoring and reporting.
- Deployment of the system in a cloud environment.

## Author

**Akif Nehal**

B.Tech – Computer Science and Engineering

- GitHub: [ANNF570](https://
