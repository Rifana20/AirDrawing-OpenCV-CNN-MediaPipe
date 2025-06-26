
# 🖐️ Air Drawing Recognition System using CNN, OpenCV & MediaPipe

This project brings AI and creativity together by allowing users to draw in the air using their **index finger**, with the sketch being classified in real-time by a custom-trained CNN model.

Instead of using a large pre-made dataset, the model was trained on a **small hand-drawn dataset** with just **10 samples per class** across 10 categories (e.g., star, fish, apple, heart, etc.), and extended via **image augmentation**. The complete pipeline includes data generation, CNN training, and a real-time hand gesture interface using **MediaPipe + OpenCV**.

---

## 🚀 Features

- Real-time air drawing using webcam and index finger tracking
- Custom CNN trained on augmented grayscale images (64x64)
- Live sketch classification and prediction feedback
- Built using TensorFlow, OpenCV, and MediaPipe

---

## 📁 Project Structure

```

AirDrawing-OpenCV-CNN-MediaPipe/
├── images/                  # Original hand-drawn sketches (10 per class)
├── augmented\_images/        # Augmented dataset for training
├── class\_names.txt          # Stores class labels in order
├── sketch\_model.keras       # Trained CNN model
├── augment.py               # Script to augment image dataset
├── train.py                 # Script to train CNN model
├── canvas.py                # Real-time air drawing and prediction
└── README.md                # Project documentation

````

---

## 🧠 Technologies Used

- **Python**  
- **TensorFlow / Keras** – for building and training the CNN  
- **OpenCV** – for webcam access and drawing canvas  
- **MediaPipe** – for index finger tracking and gesture recognition  
- **ImageDataGenerator** – for data augmentation  
- **NumPy, tqdm** – for preprocessing and progress tracking  

---

## 🛠️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Rifana20/AirDrawing-OpenCV-CNN-MediaPipe.git
cd AirDrawing-OpenCV-CNN-MediaPipe
````

### 2. Install the dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install manually:

```bash
pip install tensorflow opencv-python mediapipe numpy tqdm
```

### 3. Prepare the Dataset

* Put 10 grayscale images (64x64) for each class in the `images/` folder under subfolders like `images/apple/`, `images/star/`, etc.

### 4. Augment the Images

```bash
python augment.py
```

### 5. Train the CNN Model

```bash
python train.py
```

This saves the model as `sketch_model.keras` and the class names in `class_names.txt`.

### 6. Launch the Air Drawing App

```bash
python canvas.py
```

### Controls:

* **Draw**: Move your index finger while keeping thumb apart (to simulate drawing)
* **Predict**: Press `s`
* **Clear canvas**: Press `c`
* **Quit**: Press `q`

---

## 🖼️ Example Output


---

## ✨ What I Learned

* Creating an end-to-end pipeline from data collection to real-time AI deployment
* Handling small datasets with augmentation
* Building gesture-based interfaces using MediaPipe
* Integrating computer vision with deep learning models in real-time

---

## 📌 Future Improvements

* Increase dataset size with more classes and samples
* Improve model robustness and accuracy
* Deploy as a web app using Streamlit or Flask

---

## 🔖 License

This project is for educational and personal learning purposes.

