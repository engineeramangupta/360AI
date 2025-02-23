# 360AI Application 
### (https://huggingface.co/spaces/engineeramangupta/360AI.com)

## Table of Contents
- [360AI Application](#360ai-application)
  - [Table of Contents](#table-of-contents)
  - [General Information](#general-information)
  - [Approach Used](#approach-used)
  - [Conclusions](#conclusions)
  - [Technologies Used](#technologies-used)
  - [Acknowledgements](#acknowledgements)
  - [Contact](#contact)

---

## General Information
**What is 360AI?**  
360AI is a unified, user-friendly platform that harnesses the power of **Generative AI** to deliver multiple services under one roof:
- **GeminiPro GenAI**: A text-based chatbot interface for asking questions and receiving intelligent responses.  
- **Image Analysis**: Upload images and get AI-driven insights.  
- **PDF Analysis**: Process and query PDF documents using advanced embedding and retrieval methods.  

**Key Features**  
- **Signup & Login**: Secure user authentication with a streamlined flow.  
- **Real-time Chat**: ChatGPT-like experience for all features (text, image, PDF).  
- **Interactive Interface**: Intuitive UI built with Streamlit for seamless user engagement.  
- **Mobile-Friendly**: Optimized layouts and styling for different screen sizes.  

**Business Context**  
360AI aims to make AI accessible to everyone—be it students, researchers, or enterprises. By integrating various AI functionalities, 360AI streamlines tasks such as document summarization, image-based insights, and conversational AI into one cohesive application.

---

## Approach Used
1. **User Authentication**  
   - Signup/Login system built with Streamlit session states, ensuring secure credential storage and easy session handling.

2. **Generative AI Integration**  
   - Utilizes **Google Generative AI** (Gemini models) for text, image, and PDF queries, providing accurate and context-aware responses.

3. **Conversational Flow**  
   - Chat-style interfaces for each feature, allowing users to ask follow-up questions and receive answers in real time.

4. **PDF Embedding & Retrieval**  
   - **FAISS** vector store for storing and retrieving PDF chunks.  
   - **LangChain** integration for advanced question-answering on documents.

5. **Image Analysis**  
   - Upload an image and receive AI-generated insights based on the **Gemini** image model.

6. **Responsive UI**  
   - **Streamlit** forms and custom CSS ensure the interface is mobile-friendly and visually appealing.

---

## Conclusions
- **All-in-One AI**: 360AI consolidates text queries, image analysis, and PDF analysis into a single platform, saving time and effort.  
- **High Accuracy**: By leveraging advanced Gemini models, the application delivers reliable responses across diverse queries.  
- **User-Centric Design**: Chat-style layouts and streamlined authentication ensure a smooth user experience.  
- **Scalable & Extendable**: The modular code structure allows easy addition of new features or AI models in the future.

---

## Technologies Used
- **Python 3.9+**  
- **Streamlit** for the interactive web interface  
- **Google Generative AI** (Gemini) for text, image, and PDF analysis  
- **LangChain** for PDF embedding and question-answering  
- **FAISS** for vector storage and retrieval  
- **PIL** for image handling  
- **PyPDF2** for PDF reading and extraction  

---

## Acknowledgements
- Special thanks to **Aman Gupta** for spearheading the development of 360AI.  
- Inspired by cutting-edge research in AI and the evolving capabilities of large language models.

---

## Contact
Created by [@engineeramangupta](https://github.com/engineeramangupta) — feel free to reach out with questions or suggestions!
