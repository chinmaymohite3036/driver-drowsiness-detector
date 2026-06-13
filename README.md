# Neuromorphic Driver Drowsiness Detection

A beginner-friendly Python application featuring a futuristic "Neuromorphic Event Monitor" that detects driver drowsiness using your computer's webcam. It leverages OpenCV and MediaPipe Face Mesh to calculate both Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) in real-time.

## Features
- **Neuromorphic Event Panel:** A futuristic HUD that displays live event tracking and real-time logs.
- **Eye Closure Detection:** Calculates Eye Aspect Ratio (EAR) to determine if your eyes have been closed beyond the safety threshold.
- **Yawn Detection:** Calculates Mouth Aspect Ratio (MAR) to monitor yawning events.
- **Neural Spikes:** Instead of spamming continuous alarms, the system intelligently logs singular "spikes" in a chronological Event Log whenever an event happens.

## Prerequisites
- Python 3.13 (or older)
- Windows OS (uses `winsound` for the native system event chimes)
- A connected webcam

## Setup Instructions

1. **Open a terminal** and navigate to this project's directory.
2. (Optional but recommended) **Create a virtual environment**:
   ```bash
   python -m venv venv
   # Activate the virtual environment on Windows:
   venv\Scripts\activate
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. Ensure your webcam is available and not being used by another application.
2. **Run the main script**:
   ```bash
   python main.py
   ```
   *(Note: The first time you run this, it will automatically download the required MediaPipe model file.)*
3. A window will open showing your live webcam feed with the Neuromorphic HUD. 
4. Try closing your eyes for more than 1.5 seconds, or opening your mouth wide (yawning) for 1 second. You will hear an event chime, and the Event Log on the right side will update with a timestamped "Spike".
5. **Press `q`** while the webcam window is active to exit the application gracefully.
