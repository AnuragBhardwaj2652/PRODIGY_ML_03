# PRODIGY_ML_03
Cats vs Dogs Classification using Support Vector Machine (SVM)

Welcome to Prodigy-ML-03, a project implemented in Visual Studio Code (VS Code) that classifies images of cats and dogs using Support Vector Machine (SVM). The model is trained on the popular Dogs vs Cats dataset from Kaggle, and uses OpenCV for image preprocessing and scikit-learn for model training and evaluation.


# 📌 Libraries used:

numpy : Numerical operations  

opencv-python: Image reading and preprocessing  

matplotlib.pyplot : Visualization of predictions  

scikit-learn : Model training and evaluation  

joblib : Saving/loading trained model  


# 🗂️ Dataset

Source:https://www.kaggle.com/c/dogs-vs-cats/data

Structure: Images named like cat0.jpg, dog1.jpg, etc.

All images are converted to grayscale and resized to 64×64 pixels.


## 🧠 Features

Data preprocessing (grayscale conversion, resizing, flattening)

Label extraction from filenames

Training an SVM classifier (with linear kernel)

Evaluation using accuracy, precision, recall, and F1-score

Visualization of predictions

# 🚀 Usage 

1. Setup & Installation

#Create virtual environment 

python -m venv venv
source venv/bin/activate 

#Install dependencies
pip install -r requirements.txt

2. Run Scripts
   
To train model:
 python train.py

To make predictions:
  python predict.py

# 📈 Model Evaluation

Accuracy Score

Classification Report

Visual Comparison of predicted vs actual labels (matplotlib)

# 🔧 Future Enhancements

Use MobileNetV2 for feature extraction

Try non-linear kernels (RBF, poly) for better generalization

Deploy as a web app using Flask/Streamlit

# 🙏 Acknowledgments

Kaggle community for the Dogs vs Cats dataset  

Developers of scikit-learn, OpenCV, and matplotlib
  





   
