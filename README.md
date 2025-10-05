
# 🧠 Brain Tumor Detection using Deep Learning

This project is a simple **Brain Tumor Prediction system** built with **PyTorch + FastAPI**.  
It allows you to upload an MRI scan image and get predictions whether a tumor is present or not.  

---

## 📌 Features
- Deep Learning model (CNN) trained on MRI images.  
- REST API built with **FastAPI** for predictions.  
- Easy to run locally (CPU-based).  

---

## ⚙️ Requirements

Clone the repository:
```bash
git clone https://github.com/asqar268008/brain-tumor-prediction.git
```
```bash
cd brain-tumor-prediction
```

Create a virtual environment
```bash
python -m venv venv
```

Activate the environment:
Windows
```bash
venv\Scripts\activate
```

Mac/Linux
```bash
source venv/bin/activate
```


## Install dependencies 
```bash
pip install -r requirements.txt
```

## 🚀 Running the Project
Run the FastAPI app:
```bash
python prediction.py
```

By default, the server will start at:
```bash
http://127.0.0.1:8000/docs
```
1. Open the above link in your browser.

2. Click on /predict → Try it out

3. Upload your MRI scan image and get the prediction! 🩺
 ---

## 📝 Example API response

{

   "prediction": "notumor",
  
   "confidence": 1,
  
   "is_tumor": false
  
}
