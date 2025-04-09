# Brain-Tumor-Project
# Brain Tumor Detection using Machine Learning

## Overview
This project aims to detect brain tumors from MRI images using machine learning. The model utilizes deep learning techniques, particularly Convolutional Neural Networks (CNNs), to classify MRI scans as having a tumor or being tumor-free.

## Dataset
The dataset used for this project consists of MRI brain images obtained from publicly available sources such as:
- [BRATS Dataset](https://www.med.upenn.edu/cbica/brats2018.html)
- [Figshare Brain Tumor Dataset](https://figshare.com/articles/brain_tumor_dataset/1512427)
- [Kaggle Brain MRI Dataset](https://www.kaggle.com/datasets)

The dataset is split into training, validation, and test sets.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** TensorFlow/Keras, OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn
- **Frameworks:** TensorFlow, Keras, PyTorch (optional)
- **Tools:** Jupyter Notebook, Google Colab

## Project Structure
```
brain-tumor-detection/
│── dataset/                 # Contains MRI images
│── models/                  # Trained models
│── notebooks/               # Jupyter Notebooks
│── src/
│   ├── preprocess.py        # Image preprocessing functions
│   ├── train.py             # Model training script
│   ├── predict.py           # Prediction script
│── app.py                   # Flask app for deployment
│── requirements.txt         # Dependencies
│── README.md                # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection.git
   cd brain-tumor-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download and place the dataset in the `dataset/` directory.

## Data Preprocessing
- Convert images to grayscale (if needed).
- Resize images to 224x224 pixels.
- Normalize pixel values (0-1 range).
- Augment data using rotation, flipping, and zooming.

## Model Training
Run the training script to train the CNN model:
```bash
python train.py
```
The model is trained using **categorical cross-entropy loss** and **Adam optimizer**.

## Model Evaluation
Evaluate the trained model using:
```bash
python evaluate.py
```
This will output accuracy, precision, recall, and F1-score.

## Prediction
To make a prediction on a new MRI scan:
```bash
python predict.py --image path/to/image.jpg
```

## Deployment
The model can be deployed using Flask:
```bash
python app.py
```
Access the web app at `http://localhost:5000`.

## Future Improvements
- Improve accuracy using advanced architectures like ResNet50, EfficientNet.
- Implement Grad-CAM for explainability.
- Deploy using cloud platforms like AWS/GCP.

## Contributors
- **Your Name** – Ritesh Bhaskar

## License
This project is licensed under the MIT License.

