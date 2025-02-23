import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# ----------------- 1. Environment & Basic Setup ----------------- #
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
st.set_page_config(page_title="360AI", page_icon="✨", layout="wide")

# Meta viewport for mobile
st.markdown('<meta name="viewport" content="width=device-width, initial-scale=1">', unsafe_allow_html=True)

# ----------------- 2. Animated, Colorful CSS ----------------- #
st.markdown("""
<style>
/* Turn off browser autofill attempts (may still be overridden by some browsers) */
input {
  autocomplete: off !important;
}

/* Animated gradient background */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
body {
    background: linear-gradient(-45deg, #ffe57f, #ffd180, #ffd54f, #fff176);
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
    font-family: 'Helvetica Neue', sans-serif;
    margin:0; padding:0;
}

/* Main Headers */
.main-header {
    text-align: center;
    color: #4e342e;
    margin-top: 10px;
    margin-bottom: 5px;
    font-size: 2.5rem;
    font-weight: bold;
}
.subheader {
    color: #3e2723;
    text-align: center;
    margin-bottom: 10px;
    font-size: 1.4rem;
    font-weight: 500;
}

/* Auth container */
.auth-box {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
}

/* Chat container & bubbles */
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    padding: 10px;
    border: 3px solid #ccc;
    border-radius: 10px;
    background-color: rgba(255, 255, 255, 0.7);
    margin-bottom: 10px;
}
.chat-user {
    background-color: #c5e1a5; /* pastel green */
    color: #212121;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    max-width: 80%;
    transition: transform 0.3s ease;
    font-size: 1rem;
}
.chat-bot {
    background-color: #ffd54f; /* pastel yellow */
    color: #424242;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
    max-width: 80%;
    margin-left: auto;
    transition: transform 0.3s ease;
    font-size: 1rem;
}
.chat-user:hover, .chat-bot:hover {
    transform: scale(1.02);
}

/* Buttons with transition, each a unique pastel color */
.button-gemini {
    background-color: #80deea !important; /* pastel cyan */
    color: #004d40 !important;           /* dark teal */
    font-weight: bold;
}
.button-image {
    background-color: #f48fb1 !important; /* pastel pink */
    color: #880e4f !important;           /* dark pink */
    font-weight: bold;
}
.button-pdf {
    background-color: #b39ddb !important; /* pastel purple */
    color: #311b92 !important;           /* deep purple */
    font-weight: bold;
}
.button-about {
    background-color: #c8e6c9 !important; /* pastel green */
    color: #1b5e20 !important;           /* deep green */
    font-weight: bold;
}

/* General stButton styling for all states */
.stButton>button {
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s ease;
    margin-top: 10px;
}
.stButton>button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ----------------- 3. Headers & Logo ----------------- #
st.markdown('<h1 class="main-header">Welcome to 360AI Application ✨</h1>', unsafe_allow_html=True)
st.write("_________________________")
st.image("https://huggingface.co/spaces/engineeramangupta/360AI.com/resolve/main/logo.jpeg", width=400)
st.write("___________________________")

# ----------------- 4. Session State ----------------- #
if 'data' not in st.session_state:
    st.session_state.data = {}  # {username: password}
if 'signed_up' not in st.session_state:
    st.session_state.signed_up = False
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False

# Toggles for features
if 'geminipro_on' not in st.session_state:
    st.session_state.geminipro_on = False
if 'imageAnalysis_on' not in st.session_state:
    st.session_state.imageAnalysis_on = False
if 'pdfAnalysis_on' not in st.session_state:
    st.session_state.pdfAnalysis_on = False
if 'aboutus_on' not in st.session_state:
    st.session_state.aboutus_on = False

# Chat histories
if "chat_history_gemini" not in st.session_state:
    st.session_state.chat_history_gemini = []
if "chat_history_image" not in st.session_state:
    st.session_state.chat_history_image = []
if "chat_history_pdf" not in st.session_state:
    st.session_state.chat_history_pdf = []

# ----------------- 5. Auth Functions ----------------- #
def check_signup(new_username, new_password):
    if new_username not in st.session_state.data:
        st.session_state.data[new_username] = new_password
        return True
    return False

def check_login(username, password):
    return username in st.session_state.data and st.session_state.data[username] == password

# ----------------- 6. AI Calls Setup ----------------- #
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# GeminiPro text model
text_model = genai.GenerativeModel("gemini-pro")
def get_gemini_response(query: str) -> str:
    with st.spinner("Generating response..."):
        response = text_model.generate_content(query)
    return response.text

# Image analysis
image_model = genai.GenerativeModel("gemini-1.5-flash")
def gemini_response(input_text: str, image) -> str:
    with st.spinner("Analyzing image..."):
        if input_text.strip():
            response = image_model.generate_content([input_text, image])
        else:
            response = image_model.generate_content(image)
    return response.text

# PDF analysis
pdf_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def get_pdf_texts(pdf_docs) -> str:
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text: str):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    vs = FAISS.from_texts(chunks, embedding=pdf_embeddings)
    vs.save_local("faiss_index")

def load_faiss_index():
    return FAISS.load_local("faiss_index", pdf_embeddings, allow_dangerous_deserialization=True)

def pdf_chat_response(user_question: str) -> str:
    prompt_template = """
    Answer the questions to the best of your abilities from the provided content, making sure to provide details.
    If the answer is not available, simply say "Sorry, Please Try Again :) " without guessing.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """
    with st.spinner("Processing PDFs..."):
        db = load_faiss_index()
        docs = db.similarity_search(user_question)
        pdf_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(pdf_model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# ----------------- 7. Contact Us Section ----------------- #
def contact_us():
    st.header("About Us 💡")
    st.write("""
        **Welcome to 360AI**, your digital companion in the world of artificial intelligence!
        Created with cutting‐edge technology, our platform offers advanced solutions for your various needs,
        from text analysis and image recognition to PDF processing.
    """)
    st.header("Our Vision 🌈")
    st.write("""
        At 360AI, we strive to make AI accessible and valuable for everyone.
        Whether you're looking for insights from your documents, analyzing images, or seeking answers
        from advanced AI models, we are here to assist you with precision and ease.
    """)
    st.header("Meet the Creator 👨‍💻")
    st.write("""
        Our platform is brought to you by **Aman Gupta**, a dedicated technologist with a passion for integrating AI
        into practical solutions. Aman ensures that 360AI delivers a seamless experience.
    """)
    st.header("Contact Us 📞")
    st.write("""**Phone**: +91-7906116356  
**Email**: amu.aman19@gmail.com  
**LinkedIn**: https://www.linkedin.com/in/amanamu/

**Address**: Pune, India""")
    st.markdown("""
    <iframe 
        src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3782.9626577475274!2d73.85674311438226!3d18.520430687413616!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3bc2c1a0bb97b7ef%3A0x9938e6d6bda2951b!2sPune%2C%20Maharashtra%2C%20India!5e0!3m2!1sen!2sus!4v1623148577984!5m2!1sen!2sus" 
        width="300" 
        height="200" 
        style="border:0;" 
        allowfullscreen="" 
        loading="lazy">
    </iframe>
    """, unsafe_allow_html=True)

# ----------------- 8. Authentication Flow ----------------- #
if not st.session_state.signed_up:
    # -- Signup --
    with st.container():
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        st.header("Signup 📝")
        new_user = st.text_input("Choose your username", value="", autocomplete="off")
        new_pass = st.text_input("Enter your password", type="password", value="", autocomplete="off")
        signup_btn = st.button("Signup")
        if signup_btn:
            if new_user and new_pass:
                if check_signup(new_user, new_pass):
                    st.success("Account created! Switching to login...")
                    st.balloons()  # fun effect
                    st.session_state.signed_up = True
                    st.rerun()     # immediately show login
                else:
                    st.error("Username already exists. Choose a different username.")
            else:
                st.error("Please provide both username and password.")
        st.markdown('</div>', unsafe_allow_html=True)

elif not st.session_state.is_logged_in:
    # -- Login --
    with st.container():
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)
        st.header("Login 🔐")
        login_user = st.text_input("Enter your username", value="", autocomplete="off")
        login_pass = st.text_input("Enter your password", type="password", value="", autocomplete="off")
        login_btn = st.button("Login")
        if login_btn:
            if login_user and login_pass:
                if check_login(login_user, login_pass):
                    st.success("Logged in successfully!")
                    st.snow()  # fun effect
                    st.session_state.is_logged_in = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
            else:
                st.error("Please enter both username and password.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # -- Main App --
    st.markdown('<h2 class="subheader">Welcome to 360AI!</h2>', unsafe_allow_html=True)
    
    # Top-level feature selection
    c1, c2, c3, c4 = st.columns(4)

    # Button with custom CSS classes for unique colors
    with c1:
        if st.button("⚡ GeminiPro GenAI", key="btn_gemini", help="Chat with GeminiPro for text-based queries", 
                     use_container_width=True):
            st.session_state.geminipro_on = True
            st.session_state.imageAnalysis_on = False
            st.session_state.pdfAnalysis_on = False
            st.session_state.aboutus_on = False
    with c2:
        if st.button("🖼 Image Analysis", key="btn_image", help="Analyze images with Gemini", 
                     use_container_width=True):
            st.session_state.geminipro_on = False
            st.session_state.imageAnalysis_on = True
            st.session_state.pdfAnalysis_on = False
            st.session_state.aboutus_on = False
    with c3:
        if st.button("📄 PDF Analysis", key="btn_pdf", help="Ask questions about uploaded PDFs", 
                     use_container_width=True):
            st.session_state.geminipro_on = False
            st.session_state.imageAnalysis_on = False
            st.session_state.pdfAnalysis_on = True
            st.session_state.aboutus_on = False
    with c4:
        if st.button("ℹ️ About Us", key="btn_about", help="Learn more about 360AI", 
                     use_container_width=True):
            st.session_state.geminipro_on = False
            st.session_state.imageAnalysis_on = False
            st.session_state.pdfAnalysis_on = False
            st.session_state.aboutus_on = True

    # Assign classes to the buttons via JS hack (since Streamlit doesn't directly allow it)
    st.markdown("""
    <script>
    // Assign custom classes to the four main buttons by their text
    const btns = window.parent.document.getElementsByTagName('button');
    for (let b of btns) {
      if (b.innerText.includes('GeminiPro GenAI')) { b.classList.add('button-gemini'); }
      else if (b.innerText.includes('Image Analysis')) { b.classList.add('button-image'); }
      else if (b.innerText.includes('PDF Analysis')) { b.classList.add('button-pdf'); }
      else if (b.innerText.includes('About Us')) { b.classList.add('button-about'); }
    }
    </script>
    """, unsafe_allow_html=True)

    # --------------- GeminiPro Chat --------------- #
    if st.session_state.geminipro_on:
        st.subheader("Chat with GeminiPro ⚡")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history_gemini:
            bubble_class = "chat-user" if msg["role"] == "user" else "chat-bot"
            st.markdown(f"<div class='{bubble_class}'>{msg['message']}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Chat input with no persistent key
        user_query = st.text_input("Type your text query...", value="", autocomplete="off")
        if st.button("Send Query"):
            if user_query.strip():
                st.session_state.chat_history_gemini.append({"role": "user", "message": user_query})
                response_text = get_gemini_response(user_query)
                st.session_state.chat_history_gemini.append({"role": "bot", "message": response_text})
            st.rerun()  # clears the input box

    # --------------- Image Analysis Chat --------------- #
    if st.session_state.imageAnalysis_on:
        st.subheader("Chat with Gemini Image Analysis 🖼")
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_img:
            img_obj = Image.open(uploaded_img)
            st.image(img_obj, caption="Uploaded Image", use_column_width=True)

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history_image:
            bubble_class = "chat-user" if msg["role"] == "user" else "chat-bot"
            st.markdown(f"<div class='{bubble_class}'>{msg['message']}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        user_query = st.text_input("Ask something about this image...", value="", autocomplete="off")
        if st.button("Send Image Query"):
            if not uploaded_img:
                st.error("Please upload an image first.")
            elif user_query.strip():
                st.session_state.chat_history_image.append({"role": "user", "message": user_query})
                answer = gemini_response(user_query, img_obj)
                st.session_state.chat_history_image.append({"role": "bot", "message": answer})
            st.rerun()  # clears the input box

    # --------------- PDF Analysis Chat --------------- #
    if st.session_state.pdfAnalysis_on:
        st.subheader("Chat with Gemini PDF Analysis 📄")
        st.write("Upload PDF file(s) and click 'Process PDFs' first, then ask your questions.")

        pdf_docs = st.file_uploader("Upload PDF(s)", accept_multiple_files=True)
        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Reading & Embedding PDFs..."):
                    raw_text = get_pdf_texts(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")
            else:
                st.error("Please upload at least one PDF file before processing.")

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history_pdf:
            bubble_class = "chat-user" if msg["role"] == "user" else "chat-bot"
            st.markdown(f"<div class='{bubble_class}'>{msg['message']}</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        user_query = st.text_input("Type your PDF question...", value="", autocomplete="off")
        if st.button("Send PDF Query"):
            if user_query.strip():
                st.session_state.chat_history_pdf.append({"role": "user", "message": user_query})
                pdf_answer = pdf_chat_response(user_query)
                st.session_state.chat_history_pdf.append({"role": "bot", "message": pdf_answer})
            st.rerun()  # clears the input box

    # --------------- About Us --------------- #
    if st.session_state.aboutus_on:
        contact_us()

