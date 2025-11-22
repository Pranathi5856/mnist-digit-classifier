Handwritten Digit Recognition (MNIST – CNN)

This project is a complete handwritten digit recognition system built using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

It contains:

🧪 Training & evaluation scripts

🧠 CNN model saved as .h5

⚙️ Prediction pipeline

🎨 Streamlit app for drawing and predicting digits

📁 Well-organized modular code structure

📁 Project Structure
mnist-cnn/
│
├── app/
│   └── streamlit_app.py        # Streamlit UI to draw digit & get prediction
│
├── artifacts/
│   └── mnist_cnn.h5            # Trained MNIST CNN model
│
├── debug_images/               # (Optional) Saved intermediate images
│
├── src/
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Model evaluation script
│   ├── predict.py              # Prediction script
│   └── model.py                # CNN model architecture
│
├── venv/                       # Virtual environment (ignored in Git)
│
└── requirements.txt            # Project dependencies

🔧 Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/mnist-cnn.git
cd mnist-cnn

2️⃣ Create a virtual environment
python -m venv venv


Activate it:

Windows:

venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

🚀 Usage
▶️ Train the Model
python src/train.py


This will train the CNN and save the model to artifacts/mnist_cnn.h5.

▶️ Evaluate the Model
python src/evaluate.py


Displays accuracy metrics, plots, etc.

▶️ Test Prediction
python src/predict.py


Uses the saved CNN model to predict digits from test images.

▶️ Run Streamlit App (Draw & Predict)
streamlit run app/streamlit_app.py


The app will launch at:

👉 http://localhost:8501

You can draw any digit (0–9) and get instant predictions.

🧠 Model Details

Dataset: MNIST (60,000 training + 10,000 testing images)

Input: 28×28 grayscale digit

Architecture:

Conv2D → MaxPooling

Conv2D → MaxPooling

Flatten

Dense → Output (10 classes)

Accuracy: ~99% on MNIST

📜 Requirements

Installed via requirements.txt:

TensorFlow / Keras

NumPy

Matplotlib

OpenCV

Streamlit

Pillow
