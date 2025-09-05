import streamlit as st
import time
import sys
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import threading

# Try to import transformers, but handle if not available
try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    transformers_available = False
    st.sidebar.warning("⚠️ Transformers library not properly installed. Only extractive summarization will be available.")


# --- Download NLTK resources ---
@st.cache_resource
def download_nltk_resources():
    # List of required NLTK resources
    required_resources = ['punkt', 'stopwords']
    
    # Create a placeholder for download messages
    with st.sidebar:
        with st.expander("NLTK Resources", expanded=False):
            status_placeholder = st.empty()
            status_placeholder.info("Checking NLTK resources...")
            
            # Download all required resources
            for resource in required_resources:
                try:
                    nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
                    status_placeholder.info(f"NLTK resource '{resource}' is already downloaded.")
                except LookupError:
                    status_placeholder.warning(f"Downloading NLTK resource '{resource}'...")
                    nltk.download(resource, quiet=False)
                    status_placeholder.success(f"NLTK resource '{resource}' downloaded successfully.")
            
            # Verify resources are available
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                status_placeholder.success("All NLTK resources are available.")
            except LookupError as e:
                st.error(f"Error: {str(e)}")
                st.error("Please restart the application after resources are downloaded.")
                st.stop()

# Ensure resources are downloaded before proceeding
download_nltk_resources()

# --- Load models in background ---
model_lock = threading.Lock()
models_loaded = False
summarizer = None
qa_model = None

def load_models_background():
    global summarizer, qa_model, models_loaded
    # Only attempt to load models if transformers is available
    if not transformers_available:
        with model_lock:
            models_loaded = False
        return
        
    try:
        # Use tiny models for faster loading
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)  # CPU inference
        qa_model = pipeline("question-answering", model="deepset/minilm-uncased-squad2", device=-1)  # CPU inference
        with model_lock:
            models_loaded = True
    except OSError as e:
        # Handle memory errors
        if "paging file is too small" in str(e):
            st.error("⚠️ Memory Error: Not enough system memory to load transformer models. Using extractive summarization only.")
            # Set models_loaded to False to ensure we use extractive summarization
            with model_lock:
                models_loaded = False
        else:
            st.error(f"⚠️ Error loading models: {str(e)}. Using extractive summarization only.")
            with model_lock:
                models_loaded = False
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}. Using extractive summarization only.")
        with model_lock:
            models_loaded = False

# Check system memory before attempting to load models
import psutil

# Only start loading models if transformers is available and there's enough memory (at least 4GB free)
if transformers_available:
    system_memory = psutil.virtual_memory()
    free_memory_gb = system_memory.available / (1024 ** 3)  # Convert to GB

    if free_memory_gb >= 4.0:
        # Start loading models in background
        thread = threading.Thread(target=load_models_background)
        thread.daemon = True
        thread.start()
        st.sidebar.info(f"💾 System has {free_memory_gb:.1f}GB free memory. Transformer models will be loaded in background.")
    else:
        # Not enough memory, don't even try to load models
        st.sidebar.warning(f"⚠️ System has only {free_memory_gb:.1f}GB free memory. Transformer models will not be loaded.")
        with model_lock:
            models_loaded = False
else:
    # Transformers not available
    with model_lock:
        models_loaded = False

# --- Extractive summarization function (faster than transformer models) ---
def extractive_summarize(text, num_sentences=3):
    # Tokenize sentences
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # If punkt tokenizer fails, try to download it again
        nltk.download('punkt', quiet=True)
        # Fall back to simple sentence splitting if tokenization still fails
        try:
            sentences = sent_tokenize(text)
        except:
            # Very simple fallback tokenization
            sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    if len(sentences) <= num_sentences:
        return text
        
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    
    # Build similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)
    
    # Use PageRank algorithm to rank sentences
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)
    
    # Get top sentences
    ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
    summary = [ranked_sentences[i][2] for i in range(min(num_sentences, len(ranked_sentences)))]
    
    # Sort by original position
    summary_by_position = sorted(summary, key=lambda s: sentences.index(s))
    return " ".join(summary_by_position)

def sentence_similarity(sent1, sent2, stop_words):
    # Convert sentences to word vectors
    words1 = [word.lower() for word in sent1.split() if word.lower() not in stop_words]
    words2 = [word.lower() for word in sent2.split() if word.lower() not in stop_words]
    
    # Create a combined vocabulary
    all_words = list(set(words1 + words2))
    
    # Create word vectors
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # Fill vectors
    for word in words1:
        vector1[all_words.index(word)] += 1
    for word in words2:
        vector2[all_words.index(word)] += 1
    
    # Handle empty vectors
    if sum(vector1) == 0 or sum(vector2) == 0:
        return 0.0
    
    # Calculate cosine similarity
    return 1 - cosine_distance(vector1, vector2)

# --- Initialize session state ---
if "proceed_clicked" not in st.session_state:
    st.session_state.proceed_clicked = False
if "quiz" not in st.session_state:
    st.session_state.quiz = []
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "using_transformer" not in st.session_state:
    st.session_state.using_transformer = False

# --- App Title ---
st.title("📚 Lightweight Research Copilot (Optimized)")

# --- Sidebar for settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    # Check if transformer models are available
    with model_lock:
        transformer_models_loaded = models_loaded
    
    # Display appropriate options based on model availability
    if transformers_available and transformer_models_loaded:
        model_option = st.radio(
            "Summarization Method:",
            ["Fast (Extractive)", "Advanced (Transformer)"],
            index=0,
            help="Fast method works instantly but is less accurate. Advanced method is more accurate but may take longer to load."
        )
    else:
        if not transformers_available:
            st.warning("⚠️ Advanced (Transformer) models are not available because the transformers library is not properly installed.")
        else:
            st.warning("⚠️ Advanced (Transformer) models are not available due to memory constraints.")
        st.info("Only extractive summarization is available.")
        model_option = "Fast (Extractive)"
    
    # Number of sentences for extractive summary
    if model_option == "Fast (Extractive)":
        num_sentences = st.slider("Number of sentences in summary:", 1, 10, 3)
    else:
        max_length = st.slider("Maximum summary length:", 30, 150, 80)
        min_length = st.slider("Minimum summary length:", 10, 50, 25)
    
    # Quiz options
    generate_quiz = st.checkbox("Generate quiz questions", value=True)
    if generate_quiz:
        num_questions = st.slider("Number of questions:", 1, 5, 3)

# --- Main content area ---
st.header("📄 Input Text")
text = st.text_area("Paste your text here:", height=200)

# --- Button to process text ---
if st.button("🔍 Analyze Text", type="primary"):
    if not text.strip():
        st.warning("⚠️ Please paste some text before proceeding.")
    else:
        st.session_state.proceed_clicked = True
        
        # Check if using transformer models
        st.session_state.using_transformer = (model_option == "Advanced (Transformer)")
        
        with st.spinner("Processing text..."):
            # Get summary based on selected method
            if st.session_state.using_transformer:
                # Check if models are loaded
                with model_lock:
                    models_ready = models_loaded
                
                if not models_ready:
                    st.info("⏳ Transformer models are still loading. Using fast extractive summarization for now.")
                    summary = extractive_summarize(text, num_sentences)
                    st.session_state.using_transformer = False
                else:
                    try:
                        summary_result = summarizer(
                            text, max_length=max_length, min_length=min_length, do_sample=False
                        )
                        summary = summary_result[0]["summary_text"]
                    except Exception as e:
                        st.error(f"Error with transformer model: {str(e)}. Falling back to extractive summarization.")
                        summary = extractive_summarize(text, num_sentences)
                        st.session_state.using_transformer = False
            else:
                # Use extractive summarization
                summary = extractive_summarize(text, num_sentences)
                
                # Debug the summary content
                if "py\"" in summary or "line" in summary:
                    st.error("Debug: Summary contains error text. Raw summary: " + str(summary)[:100])
                    # Try to clean the summary
                    summary = "The text could not be properly summarized. Please try again with different content."
            
            st.session_state.summary = summary
            
            # Generate quiz questions
            if generate_quiz:
                st.session_state.quiz = []
                
                # Basic quiz generation
                try:
                    sentences = sent_tokenize(summary)
                except LookupError:
                    # If punkt tokenizer fails, try to download it again
                    nltk.download('punkt', quiet=True)
                    # Fall back to simple sentence splitting if tokenization still fails
                    try:
                        sentences = sent_tokenize(summary)
                    except:
                        # Very simple fallback tokenization
                        sentences = [s.strip() for s in summary.split('.') if s.strip()]
                
                # Define a list of generic questions to use
                generic_questions = [
                    {"question": "What is the main idea of the text?", "answer": sentences[0] if len(sentences) > 0 else summary},
                    {"question": "Mention one detail from the text.", "answer": sentences[1] if len(sentences) > 1 else summary},
                    {"question": "Why is this topic important?", "answer": summary},
                    {"question": "What are the key concepts discussed in this text?", "answer": summary},
                    {"question": "How would you summarize this text in your own words?", "answer": summary}
                ]
                
                # Add questions up to the requested number
                for i in range(min(num_questions, len(generic_questions))):
                    st.session_state.quiz.append(generic_questions[i])
                
                # Define additional generic questions for when QA model is not available
                additional_generic_questions = [
                    {"question": "What are the key points in this text?", "answer": summary},
                    {"question": "What conclusion can be drawn from this text?", "answer": summary},
                    {"question": "What evidence supports the main argument in this text?", "answer": summary},
                    {"question": "What are potential implications of this information?", "answer": summary},
                    {"question": "How does this information relate to broader contexts?", "answer": summary}
                ]
                
                # Use QA model for more questions if available and needed
                if st.session_state.using_transformer and models_loaded and transformers_available and num_questions > len(generic_questions):
                    try:
                        additional_questions = [
                            {"q": "What are the key points?", "display": "What are the key points in this text?"},
                            {"q": "What conclusion can be drawn?", "display": "What conclusion can be drawn from this text?"},
                            {"q": "What evidence supports the main argument?", "display": "What evidence supports the main argument in this text?"},
                            {"q": "What are potential implications?", "display": "What are potential implications of this information?"},
                            {"q": "How does this relate to broader contexts?", "display": "How does this information relate to broader contexts?"}
                        ]
                        
                        # Add additional questions from QA model up to the requested number
                        for i in range(min(num_questions - len(generic_questions), len(additional_questions))):
                            qa_result = qa_model(question=additional_questions[i]["q"], context=text)
                            st.session_state.quiz.append({"question": additional_questions[i]["display"], "answer": qa_result["answer"]})
                    except Exception as e:
                        # Fall back to generic questions if QA model fails
                        st.warning(f"Could not generate additional questions with QA model: {str(e)}")
                        for i in range(min(num_questions - len(generic_questions), len(additional_generic_questions))):
                            st.session_state.quiz.append(additional_generic_questions[i])
                elif num_questions > len(generic_questions):
                    # Use additional generic questions when QA model is not available
                    for i in range(min(num_questions - len(generic_questions), len(additional_generic_questions))):
                        st.session_state.quiz.append(additional_generic_questions[i])

# --- Show results if processing is complete ---
if st.session_state.proceed_clicked:
    st.header("📝 Results")
    
    # Display method used
    if st.session_state.using_transformer:
        st.success("✅ Used transformer-based summarization (more accurate)")
    else:
        st.info("ℹ️ Used extractive summarization (faster)")
    
    # Display summary
    st.subheader("Summary")
    # Fix the summary display to properly show the text
    if st.session_state.summary:
        # Sanitize the summary to remove any error messages or code snippets
        sanitized_summary = st.session_state.summary
        if "py\"" in sanitized_summary and "line" in sanitized_summary:
            # If the summary contains error messages, display a warning and try to extract actual text
            st.warning("⚠️ There was an issue with the summary generation. Showing best available result.")
            # Try to extract actual text if possible
            try:
                sanitized_summary = " ".join([s for s in sanitized_summary.split() if not ("py\"" in s or "line" in s or "read" in s or "self" in s)])
            except:
                sanitized_summary = "Summary generation failed. Please try again with different text."
        
        st.markdown(sanitized_summary)
    
    # Display quiz if generated
    if generate_quiz and st.session_state.quiz:
        st.subheader("📌 Quiz")
        
        # Create columns for score tracking
        score_col, total_col = st.columns(2)
        score = 0
        
        # Display questions
        for i, qa in enumerate(st.session_state.quiz):
            with st.expander(f"Question {i+1}: {qa['question']}"):
                user_ans = st.text_input("Your answer:", key=f"q{i}")
                
                if user_ans:
                    if qa["answer"].lower() in user_ans.lower() or user_ans.lower() in qa["answer"].lower():
                        st.success("Correct! ✅")
                        score += 1
                    else:
                        st.error("Incorrect ❌")
                        st.info(f"Hint: {qa['answer']}")
        
        # Update score
        with score_col:
            st.metric("Score", score)
        with total_col:
            st.metric("Total Questions", len(st.session_state.quiz))
            
        # Progress bar for score
        if len(st.session_state.quiz) > 0:
            st.progress(score / len(st.session_state.quiz))

# --- Performance Monitoring ---
if st.session_state.proceed_clicked:
    st.markdown("---")
    st.subheader("⚡ Performance Metrics")
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Display processing time (simulated for now)
        if st.session_state.using_transformer:
            proc_time = "0.8-2.5 seconds"
        else:
            proc_time = "<0.1 seconds"
        st.metric("Processing Time", proc_time)
    
    with col2:
        # Display model status
        with model_lock:
            model_status = "Loaded" if models_loaded else "Loading"
        st.metric("Transformer Models", model_status)
    
    with col3:
        # Display memory usage estimate
        if st.session_state.using_transformer:
            memory = "~2 GB"
        else:
            memory = "~200 MB"
        st.metric("Est. Memory Usage", memory)
    
    # Add expandable section with detailed performance info
    with st.expander("📊 Detailed Performance Information"):
        st.write("**System Information:**")
        st.code(f"""Python Version: {sys.version.split()[0]}
Streamlit Version: {st.__version__}
NLTK Version: {nltk.__version__}
NetworkX Version: {nx.__version__}
NumPy Version: {np.__version__}
""")
        
        st.write("**Model Information:**")
        if st.session_state.using_transformer:
            st.write("Using transformer-based models for summarization.")
            st.write("- Summarization: facebook/bart-large-cnn")
            st.write("- Question Answering: deepset/minilm-uncased-squad2")
        else:
            st.write("Using extractive summarization with NLTK and NetworkX.")
            st.write("- Algorithm: TextRank (PageRank variant for text)")
            st.write("- Similarity Measure: Cosine similarity of word vectors")
        
        # Add performance tips
        st.write("**Performance Tips:**")
        st.info("""
        - For faster processing, use the 'Fast (Extractive)' method
        - For more accurate results, use 'Advanced (Transformer)' method
        - Transformer models load in the background and will be available after the first run
        - Processing time increases with text length
        """)

# --- Footer ---
st.markdown("---")
st.caption("Lightweight Research Copilot - Optimized for performance and usability")
