# Live Object Detection

A Python-based computer vision project that allows training custom object detection models and performing real-time detection using trained weights. The project supports single, dual, and triple dataset training with validation and inference support.

---

## 🚀 Features

- 📸 Real-time object detection  
- 🧠 Custom model training support  
- 🔁 Dual dataset training  
- 🔂 Triple dataset training  
- ✅ Model validation  
- 🎯 Accurate bounding box detection  
- 📊 Performance evaluation  

---

## 🛠 Tech Stack

- Programming Language: Python  
- Libraries Used:
  - OpenCV  
  - PyTorch  
  - NumPy  
  - Torchvision  
- Framework:
  - YOLO / Deep Learning Detection Framework  
- Tools:
  - VS Code  
  - Git & GitHub  

---

## 📂 Project Structure

```
Object-Detection-Project/
│
├── detect.py          # Detection script
├── train.py           # Single dataset training
├── train_dual.py      # Dual dataset training
├── train_triple.py    # Triple dataset training
├── val.py             # Model validation
└── README.md
```

---

## ⚙ Installation Steps

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

---

### 2️⃣ Navigate to Project Folder

```bash
cd your-repository-name
```

---

### 3️⃣ Install Required Packages

```bash
pip install opencv-python torch torchvision numpy
```

---

## ▶ Running The Project

---

### 🔹 Train Model (Single Dataset)

```bash
python train.py
```

---

### 🔹 Train Using Dual Dataset

```bash
python train_dual.py
```

---

### 🔹 Train Using Triple Dataset

```bash
python train_triple.py
```

---

### 🔹 Validate Model

```bash
python val.py
```

---

### 🔹 Run Detection

```bash
python detect.py
```

---

## 🧠 How It Works

1. Training scripts load datasets and preprocess data  
2. Deep learning model is trained on labeled images  
3. Validation script evaluates accuracy and performance  
4. Detection script uses trained weights  
5. Input image/video/webcam feed is processed  
6. Objects are detected with bounding boxes  

---

## 📌 Learning Outcomes

- Deep Learning fundamentals  
- Computer Vision concepts  
- Object detection pipelines  
- Model training and validation  
- Dataset handling  
- Real-time video processing  
- PyTorch framework usage  

---

## ⚠ Requirements

- Python 3.8 or above  
- GPU (Recommended for faster training)  
- Webcam (For real-time detection)  
- Proper dataset with annotations  

---

## 📸 Screenshots / Results

(Add detection output screenshots here)

Example:

```markdown
![Detection Output](screenshots/output.png)
```

---

## 👤 Author

**Karnam Nikhilesh**  
GitHub: https://github.com/Karnam-Nikhilesh 

---

## ⭐ Support

If you like this project, please give it a ⭐ on GitHub!

---

## 📄 License

This project is developed for educational and learning purposes.
