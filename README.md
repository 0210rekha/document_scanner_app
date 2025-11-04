# 📄 Smart Document Scanner using OpenCV & Streamlit

This project is a **Document Scanner App** built using **Computer Vision** techniques in Python.  
It detects the edges of a document, finds its boundary, and applies a **perspective transformation** to create a flattened, scanned-like output — all directly in your browser via **Streamlit**.

---

## 🚀 Features
- Detects document edges using Canny edge detection.
- Finds the document contour using contour approximation.
- Applies perspective transform to get a top-down scanned view.
- Provides clean black-and-white scanned output.

---

## 🧠 Tech Stack
- Python 🐍  
- OpenCV  
- NumPy  
- Streamlit  
- Pillow (PIL)

---

## 💻 Run Locally
1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/document-scanner-app.git
   cd document-scanner-app
