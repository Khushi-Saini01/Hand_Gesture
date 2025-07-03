# Hand_Gesture
🖐️ Hand Gesture Recognition using CNN & Streamlit
A deep learning-based web application built with TensorFlow and Streamlit that classifies hand gestures (0–19) using a trained Convolutional Neural Network (CNN). Upload an image with a black background and get the predicted hand gesture instantly! 🧠📸
#📸 Demo
https://your-deployment-link.streamlit.app/ (Replace this with your actual Streamlit Cloud link)

#🚀 Features
✅ Upload hand gesture images

🧠 Predict gesture using a trained CNN model

📊 Displays predicted class and confidence score

🖼️ Accepts .jpg, .png, .jpeg formats

⚡ Fast and lightweight — works locally and online

#🗂️ Dataset
This model was trained on a custom hand gesture dataset containing 20 classes (0 to 19) with a black background.

Each image was resized to 64x64 pixels and normalized before training.

If you're looking for a similar dataset, check Kaggle Hand Gesture Datasets or build your own using webcam captures.

#🏗️ Tech Stack
Frontend: Streamlit 🟩

Backend: TensorFlow (Keras API) 🧠

Libraries: OpenCV, NumPy, Pillow

Language: Python 🐍

#📁 Project Structure
Hand_Gesture_Recognition/
├── app.py                     # Streamlit web app
├── hand_gesture_cnn_model.h5 # Trained CNN model
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
#⚙️ Setup Instructions
🔽 1. Clone the Repository
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
🛠 2. Install Dependencies
pip install -r requirements.txt
▶️ 3. Run the Streamlit App
streamlit run app.py
Then open http://localhost:8501 in your browser.

#📦 Model Training Summary
CNN model trained with 2 convolutional layers, MaxPooling, and Dropout

Optimizer: Adam

Loss Function: categorical_crossentropy

Accuracy Achieved: ~95%+ on validation data

Trained using ImageDataGenerator with rotation and flip augmentation

📷 Image Upload Guidelines
Upload images with a black background

Use one hand per image

Accepts .jpg, .jpeg, .png

#🌍 Deployment
You can deploy the app to Streamlit Cloud:

Push the code to GitHub

Go to streamlit.io/cloud

Connect your repo and deploy

#💡 Future Ideas
Add real-time prediction via webcam (OpenCV)

Convert model to .tflite for mobile deployment

Add gesture-to-text feature

Train with larger, diverse datasets

#🙋‍♀️ Author
Khushi Saini
🧠 AI & ML Enthusiast | 👩‍💻 CSE (AIML) @ Chandigarh University
#📫 LinkedIn : https://www.linkedin.com/in/khushi-saini-b99724285/
