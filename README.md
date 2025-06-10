# MENTAL-HEALTH-CARE-USING-FACE-RECOGINITION
How to Run the Project: Mental Health Tracker Using Face Emotion Detection

Step 1: Set Up Project Folder
Ensure your project files are organized in the following structure:
cpp
CopyEdit
mental_health_tracker/
├── app.py
├── emotion_model.py
├── model/
│   ├── emotiondetector.json
│   └── emotiondetector.h5
├── emotion_history/       (auto-created at runtime)
├── habit_data/            (auto-created at runtime)
•	app.py: Main application script.
•	emotion_model.py: Contains face detection and emotion prediction logic.
•	model/: Folder containing the trained emotion detection model files.

Step 2: Install Python
Ensure you have Python 3.8 or above installed.
You can check using:
bash
CopyEdit
python --version
If not installed, download it from: https://www.python.org/downloads/

Step 3: Install Required Python Libraries
Open your terminal or command prompt and run:
bash
CopyEdit
pip install streamlit streamlit-webrtc streamlit-authenticator opencv-python-headless tensorflow keras pandas numpy
This installs all required libraries for the application to run.

Step 4: Navigate to the Project Directory
Use the terminal or command prompt to go to the project folder:
bash
CopyEdit
cd path/to/your/mental_health_tracker
Replace path/to/your/ with the actual location on your computer.

Step 5: Run the Streamlit Application
Execute the following command to launch the app:
bash
CopyEdit
streamlit run app.py
This will start a local server.

Step 6: Open the App in a Web Browser
Once Streamlit runs, it will open your browser or show a link such as:
nginx
CopyEdit
Local URL: http://localhost:8501
Click the link or copy and paste it into your browser to access the application.

Step 7: Log In to the System
Use one of the pre-configured credentials found in app.py. For example:
•	Username: alice
•	Password: 123
You can customize these credentials in the login section of app.py.

Step 8: Use the Application Features
After logging in, you will have access to:
•	Live face emotion detection using webcam
•	Emotion-based wellness tips, music, and videos
•	Psychological First Aid (PFA) chatbot
•	Daily habit tracker
•	Emotion history log
Make sure your webcam is enabled and allowed in the browser.

Troubleshooting Tips
•	If the webcam does not start, ensure no other application is using it.
•	Use a modern browser like Chrome or Edge for best compatibility.
•	If modules fail to import, double-check the library installation in Step 3.

