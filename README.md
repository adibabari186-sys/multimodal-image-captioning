# 🤖 Multimodal Image Captioning System

An advanced Deep Learning application that bridges **Computer Vision (CV)** and **Natural Language Processing (NLP)** into a singular multimodal architecture. This project leverages a **Vision Transformer (ViT)** encoder to extract spatial visual features from images and feeds them into an auto-regressive **GPT-2 Decoder** to generate highly accurate, context-aware textual captions. 

The entire framework is wrapped in a high-end, custom-engineered **Neon Cyberpunk Dark-Theme UI** built using Streamlit and custom CSS injection.

---

## 🚀 Key Features

*   **Multimodal AI Pipeline:** Seamlessly unifies advanced computer vision feature mapping with language generation models.
*   **State-of-the-Art Transformers:** Utilizes Hugging Face's `ViT-GPT2` cross-attention architecture for high-fidelity text-from-image generation.
*   **Stunning Cyber-UI:** Completely customized Streamlit interface using raw HTML/CSS styling, featuring vibrant glowing containers, neon headers, and coordinated matrix loaders.
*   **Smart Resource Caching:** Deep learning models are cached natively inside the application runtime for instant performance and zero-latency subsequent inference.
*   **Real-time Image Workspace:** Live image preview arrays that match the dark mode ecosystem dynamically.

---

## 🛠️ Deep Learning Architecture

1.  **Vision Encoder:** The pre-trained `ViTImageProcessor` tokenizes incoming multi-channel visual matrices (images) into fixed-size spatial feature patches.
2.  **Transformer Decoder:** An auto-regressive language head (`GPT2LMHeadModel`) processes these visual tokens via cross-attention layers using **Beam Search Decoding** to generate grammatically accurate sentences describing the image.

---

## 📦 Tech Stack & Frameworks

*   **Core Backend:** Python 3.x
*   **Deep Learning Backend:** PyTorch (`torch`, `torchvision`)
*   **Transformer Architecture:** Hugging Face `transformers`
*   **Web Framework:** Streamlit (Custom theme-engine overrides)
*   **Image Handling:** Pillow (`PIL`)

---

## 💻 Installation & Local Setup

Follow these steps to spin up the cyber-workspace environment on your local system:

### 1. Clone the repository
```bash
git clone [https://github.com/adibabari186-sys/multimodal-image-captioning.git](https://github.com/adibabari186-sys/multimodal-image-captioning.git)
cd multimodal-image-captioning
