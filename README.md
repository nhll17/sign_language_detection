


# 🤟 Sign Language Detection Web App

A real-time sign language recognition system built with Flask, OpenCV, and machine learning. This application captures sign gestures via webcam, predicts the corresponding alphabet using a trained ML model, and speaks the result using browser-based speech synthesis.

## 🌟 Features

- 🎥 Real-time webcam-based sign detection
- 🧠 Trained machine learning model (Random Forest)
- 💬 Automatic speech output of predicted sign
- 💻 Beautiful glassmorphic UI with Bootstrap 5
- 🧪 Speech synthesis integration (browser-based)
- 🛠️ Flask backend for model integration and video streaming

---

## 📁 Project Structure

```

sign-language-flask-app/
│
├── static/                    # (Optional for custom CSS/JS)
├── templates/
│   └── index.html             # Frontend HTML file
│
├── model/
│   └── model.pkl              # Trained ML model
│
├── sign\_data\_combined.csv     # Dataset used for training
├── app.py                     # Flask server logic
├── video.py                   # Webcam + Prediction logic
├── requirements.txt           # Python dependencies
└── README.md                  # You're here!

````

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Devatheertha05/sign_language_detection.git
cd sign_language_detection
````

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
python app.py
```

Then open your browser and go to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 Model Training (Optional)

You can retrain the model using `sign_data_combined.csv` in a Jupyter Notebook or a Python script:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd, pickle

df = pd.read_csv("sign_data_combined.csv")
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
```

---

## 🧪 Tech Stack

* **Frontend:** HTML, Bootstrap 5, JavaScript (SpeechSynthesis API)
* **Backend:** Python, Flask
* **ML Model:** RandomForest (scikit-learn)
* **Others:** OpenCV, Pandas, Pickle

---

## 🔊 Demo

* 📷 Real-time webcam feed
* ✋ Predicts hand sign as alphabet
* 🗣️ Automatically speaks the predicted result

---

## 📦 Dependencies

Install via:

```bash
pip install -r requirements.txt
```

Typical libraries used:

* flask
* opencv-python
* scikit-learn
* pandas
* numpy

---

## 📜 License

This project is open-source and free to use.

---

## 🤝 Contributions

Feel free to fork this repository and open a pull request. Suggestions and improvements are welcome!

---

## 🧑‍💻 Author

**Devatheertha**
[GitHub Profile](https://github.com/Devatheertha05)

```

