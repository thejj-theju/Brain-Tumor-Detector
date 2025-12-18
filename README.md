# 🧠 Brain Tumor MRI Image Classification -- CNN

A deep learning project to classify brain MRI images using Convolutional Neural Networks (CNNs).  
The goal is to automatically detect and classify brain tumors using MRI scans with high accuracy.

---

## 🔍 Project Flow

The main steps included in this project:

1. **Importing Libraries**  
2. **Mount Drive** (Google Colab)  
3. **Data Preprocessing & Augmentation**  
   - Resize images  
   - Normalize pixel values  
   - Augment data (rotate, flip, etc.)  
4. **Build Custom CNN Model**  
5. **Use Transfer Learning**  
   - Leverage pre-trained models (e.g.,VGG16,ResNet50, MobileNetV2,InceptionV3,EfficientNetB0)  
6. **Train Model**  
7. **Evaluate Model**  
8. **Save Trained Model**

---

## 📊 Model Performance

| Model          | Accuracy | Precision | Recall   |
|----------------|----------|-----------|----------|
| VGG16          | 0.760163 | 0.589431  | 0.895062 |
| ResNet50       | 0.735772 | 0.823204  | 0.605691 |
| MobileNetV2    | 0.849594 | 0.876596  | 0.837398 |
| InceptionV3    | 0.821138 | 0.859729  | 0.772358 |
| EfficientNetB0 | 0.325203 | 0.000000  | 0.000000 |

---

## 🛠 Technologies Used

- Python  
- TensorFlow & Keras  
- Convolutional Neural Network (CNN)  
- Transfer Learning  
- Google Colab (Drive Mount)  
- NumPy, Matplotlib

---

## 📌 To Run

Use Terminal (or Colab / local environment):
1. In Jupyter,
-- streamlit run streamlit_project_5.py

2. In colab,
%%writefile application.py
" write the streamlit code "
Run using cloudfare tunnel:

## Streamlit Application:

<img width="1919" height="815" alt="Screenshot 2025-12-07 191222" src="https://github.com/user-attachments/assets/a9fee486-3dec-45af-a96f-efac07c3021b" />

