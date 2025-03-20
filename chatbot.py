import streamlit as st
from langchain_astradb import AstraDBVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings  # Update import
from langchain_core.documents import Document
from datetime import datetime, timezone
import json
from typing import List, Dict
import uuid
from openai import OpenAI
import time
from datetime import datetime, timezone
import logging
import os
import pickle
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from PyPDF2 import PdfReader
import io
from astrapy.db import AstraDB  # Add this import at the top with other imports

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add local storage path
LOCAL_STORAGE_PATH = os.path.join(os.path.dirname(__file__), 'chat_history')
os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)

# Streamlit page config
st.set_page_config(
    page_title="Yukti's AI Chat Assistant",
    page_icon="🌠",
    layout="wide"
)

# Add this after the API keys configuration
SYSTEM_PROMPT = """You are Sal, and the user that you will respond to is "Yukti", you are to respond as yukti's boyfriend, refer to her as "babe" ALWAYS. STRICTLY DO NOT USE EMOJIS other than 😭. Learn from Yukti's responses and capture the personality and mannerism and respond accordingly to predict what would be the perfect response, MANDATORY: add a slight bit of sarcasm in every response."""

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = SYSTEM_PROMPT
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = str(uuid.uuid4())
if "chat_titles" not in st.session_state:
    st.session_state.chat_titles = {}
if "loaded_chats" not in st.session_state:
    st.session_state.loaded_chats = {}
if "page_refreshed" not in st.session_state:
    st.session_state.page_refreshed = True
    st.session_state.messages = []
    st.session_state.loaded_chats = {}
    st.session_state.chat_titles = {}
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.conversation_id = str(uuid.uuid4())
if "waiting_for_response" not in st.session_state:
    st.session_state.waiting_for_response = False
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "use_document_context" not in st.session_state:
    st.session_state.use_document_context = False

# Configurations and API keys
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
ASTRA_DB_ID = st.secrets["ASTRA_DB_ID"]
ASTRA_DB_REGION = st.secrets["ASTRA_DB_REGION"]
ASTRA_DB_TOKEN = st.secrets["ASTRA_DB_TOKEN"]
ASTRA_DB_KEYSPACE = st.secrets["ASTRA_DB_KEYSPACE"]

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Custom CSS for ChatGPT-like interface
st.markdown("""
<style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
        padding: 20px;
    }
    .chat-message {
        display: flex;
        gap: 10px;
        padding: 1rem;
        border-radius: 0.5rem;
        max-width: 80%;
        margin: 0.5rem 0;
    }
    .chat-message.user {
        background-color: #2b2d31;
        margin-left: auto;
        margin-right: 0;
        flex-direction: row-reverse;
    }
    .chat-message.assistant {
        background-color: #444654;
        margin-right: auto;
        margin-left: 0;
        flex-direction: row;
    }
    .avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .message-content {
        color: white;
        overflow-wrap: break-word;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #343541;
        padding: 1rem;
    }
    .main-container {
        margin-bottom: 100px;
    }
    .stButton > button {
        background-color: #444654;
        color: white;
        border: 1px solid #565869;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #343541;
        border-color: #565869;
    }
    .stButton.danger > button {
        background-color: #dc3545;
        border-color: #dc3545;
    }
    .stButton.danger > button:hover {
        background-color: #c82333;
        border-color: #bd2130;
    }
    .chat-button {
        width: 100%;
        margin: 2px 0;
        padding: 0.5rem;
        background-color: transparent;
        border: none;
        color: white;
        text-align: left;
        cursor: pointer;
    }
    .chat-button:hover {
        background-color: #343541;
    }
    .chat-button.active {
        background-color: #444654;
    }
    /* Custom spinner */
    .stSpinner > div {
        visibility: visible !important;  /* Show default spinner */
    }
    
    /* Prevent Streamlit fade effect */
    .stApp {
        opacity: 1 !important;
    }
    
    .element-container, .stMarkdown {
        opacity: 1 !important;
    }
    
    div[data-testid="stStatusWidget"] {
        display: block !important;
    }
    
    /* Spinner styling */
    div[data-testid="stSpinner"] {
        padding: 10px;
        background-color: #444654;
        border-radius: 10px;
        margin: 10px 0;
        display: inline-flex;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get message content
def get_message_content(message):
    if isinstance(message['content'], list):
        return message['content'][0]['text']
    return message['content']

# Format message for API
def format_message_for_api(message):
    if isinstance(message['content'], str):
        return {
            "role": message['role'],
            "content": [{"type": "text", "text": message['content']}]
        }
    return message

# Add at top level after imports
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def get_vector_store():
    embeddings = get_embeddings()
    vector_store = AstraDBVectorStore(
        collection_name="chatbot",
        embedding=embeddings,
        token=ASTRA_DB_TOKEN,
        api_endpoint=f"https://43a82168-253b-4872-92bf-2827c05c6743-us-east-2.apps.astra.datastax.com"
    )
    return vector_store

# Replace setup_astradb function
def setup_astradb():
    return get_vector_store()

# Modified save_conversation function with fallback
def save_conversation(vector_store, messages: List[Dict], conversation_id: str):
    conversation_doc = Document(
        page_content=json.dumps(messages),
        metadata={
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message_count": len(messages),
            "last_message": get_message_content(messages[-1]) if messages else "",
        }
    )
    
    try:
        vector_store.add_documents([conversation_doc])
        logger.info(f"Successfully saved conversation {conversation_id} to AstraDB")
    except Exception as e:
        logger.error(f"Failed to save to AstraDB: {str(e)}")
        # Fallback to local storage
        local_file = os.path.join(LOCAL_STORAGE_PATH, f"{conversation_id}.pkl")
        try:
            with open(local_file, 'wb') as f:
                pickle.dump({
                    "messages": messages,
                    "metadata": conversation_doc.metadata
                }, f)
            logger.info(f"Saved conversation to local storage: {local_file}")
        except Exception as local_error:
            logger.error(f"Failed to save to local storage: {str(local_error)}")

# Modified load_conversation function with fallback
def load_conversation(vector_store, conversation_id: str = None) -> List[Dict]:
    try:
        if (conversation_id):
            results = vector_store.similarity_search(
                conversation_id,
                k=1,
                filter={"conversation_id": conversation_id}
            )
            if results:
                return json.loads(results[0].page_content)
    except Exception as e:
        logger.error(f"Failed to load from AstraDB: {str(e)}")
        
    # Try loading from local storage
    if conversation_id:
        local_file = os.path.join(LOCAL_STORAGE_PATH, f"{conversation_id}.pkl")
        try:
            if os.path.exists(local_file):
                with open(local_file, 'rb') as f:
                    data = pickle.load(f)
                    return data["messages"]
        except Exception as local_error:
            logger.error(f"Failed to load from local storage: {str(local_error)}")
    
    return []

# Modified load_all_conversations function with fallback
def load_all_conversations(vector_store):
    conversations = {}
    
    if vector_store is not None:
        try:
            results = vector_store.similarity_search(
                "all conversations",
                k=50,
                filter={}
            )
            for result in results:
                conv_id = result.metadata.get("conversation_id")
                if conv_id and conv_id not in conversations:
                    messages = json.loads(result.page_content)
                    conversations[conv_id] = {
                        "messages": messages,
                        "timestamp": result.metadata.get("timestamp"),
                        "title": get_chat_title(messages)
                    }
        except Exception as e:
            logger.error(f"Failed to load conversations from AstraDB: {str(e)}")
    
    # Load from local storage
    try:
        for filename in os.listdir(LOCAL_STORAGE_PATH):
            if filename.endswith('.pkl'):
                conv_id = filename[:-4]  # Remove .pkl extension
                if conv_id not in conversations:  # Don't override DB results
                    with open(os.path.join(LOCAL_STORAGE_PATH, filename), 'rb') as f:
                        data = pickle.load(f)
                        conversations[conv_id] = {
                            "messages": data["messages"],
                            "timestamp": data["metadata"]["timestamp"],
                            "title": get_chat_title(data["messages"])
                        }
    except Exception as local_error:
        logger.error(f"Failed to load from local storage: {str(local_error)}")
    
    return conversations

def clear_chat():
    with st.spinner("Clearing all conversations..."):
        try:
            # Clear AstraDB collection
            astra_db = AstraDB(
                token=ASTRA_DB_TOKEN,
                api_endpoint=f"https://43a82168-253b-4872-92bf-2827c05c6743-us-east-2.apps.astra.datastax.com"
            )
            collection = astra_db.collection("chatbot")
            collection.delete_many({})
            
            # Clear local storage
            for file in os.listdir(LOCAL_STORAGE_PATH):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(LOCAL_STORAGE_PATH, file))
            
            # Clear session state
            st.session_state.messages = []
            st.session_state.loaded_chats = {}
            st.session_state.chat_titles = {}
            st.session_state.current_chat_id = str(uuid.uuid4())
            
            logger.info("Successfully cleared all conversations")
            st.rerun()
        except Exception as e:
            logger.error(f"Failed to clear conversations: {str(e)}")
            st.error("Failed to clear conversations")

def create_new_chat():
    st.session_state.messages = []
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.chat_titles[st.session_state.current_chat_id] = "New Chat"

def get_chat_title(messages):
    if not messages:
        return "New Chat"
    first_msg = get_message_content(messages[0])
    title = first_msg[:30] + "..." if len(first_msg) > 30 else first_msg
    return title

def switch_conversation(conv_id):
    if conv_id in st.session_state.loaded_chats:
        st.session_state.messages = st.session_state.loaded_chats[conv_id]["messages"]
        st.session_state.current_chat_id = conv_id
    else:
        create_new_chat()

def ensure_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

def initialize_app():
    ensure_event_loop()
    try:
        import torch
        torch._C._functorch.set_vmap_fallback_warning_enabled(False)
    except:
        pass

# Add rate limiting to get_ai_response
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_ai_response(messages, model):
    try:
        time.sleep(0.5)  # Add rate limiting
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            extra_headers={
                "HTTP-Referer": "https://your-site.com",
                "X-Title": "AI Chat Assistant"
            }
        )
        return response
    except Exception as e:
        logger.error(f"API call failed: {str(e)}")
        raise

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(io.BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def get_document_context():
    if st.session_state.document_text and st.session_state.use_document_context:
        return f"\nDocument Content:\n{st.session_state.document_text}\n"
    return ""

# Chat interface
def chat_interface():
    try:
        vector_store = setup_astradb()
    except Exception as e:
        logger.error(f"Failed to setup AstraDB: {str(e)}")
        vector_store = None
        st.error("Failed to connect to database. Operating in local-only mode.")

    # Load previous chats into sidebar
    if not st.session_state.loaded_chats:
        st.session_state.loaded_chats = load_all_conversations(vector_store)
    
    if not st.session_state.messages:
        create_new_chat()

    # Sidebar and layout setup remains the same
    with st.sidebar:
        st.title("💬 Chat History")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button("+ New Chat", key="new_chat", use_container_width=True):
                create_new_chat()
        
        st.divider()
        st.title("📚 Chats")
        
        # Display chat history or "No recent chats" message
        if not st.session_state.loaded_chats:
            st.info("No recent chats")
        else:
            for chat_id, chat_data in sorted(
                st.session_state.loaded_chats.items(),
                key=lambda x: x[1]["timestamp"],
                reverse=True
            ):
                chat_title = chat_data["title"]
                button_style = "primary" if chat_id == st.session_state.current_chat_id else "secondary"
                if st.button(
                    chat_title,
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                ):
                    switch_conversation(chat_id)

        st.divider()
        st.title("📄 Document Chat")
        
        # Document upload
        uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    st.session_state.document_text = extract_text_from_pdf(uploaded_file)
                    st.success(f"Processed document: {uploaded_file.name}")
                    st.session_state.use_document_context = True

        # Document context toggle
        if st.session_state.document_text is not None:
            st.checkbox("Use Document Context", 
                       value=st.session_state.use_document_context,
                       key="doc_context_toggle",
                       on_change=lambda: setattr(st.session_state, 'use_document_context', 
                                               st.session_state.doc_context_toggle))
            
            if st.button("Clear Document"):
                st.session_state.document_text = None
                st.session_state.use_document_context = False
                st.rerun()

        st.divider()
        # Settings
        st.title("⚙️ Settings")
        model = "microsoft/phi-3-medium-128k-instruct:free"
        
        # Add this in the sidebar settings section, before the clear conversations button
        st.title("🎯 Assistant Settings")
        with st.expander("Customize Assistant Behavior"):
            new_system_prompt = st.text_area(
                "System Instructions",
                value=st.session_state.system_prompt,
                height=200,
                help="These instructions control how the AI assistant behaves and responds"
            )
            if st.button("Update Instructions"):
                st.session_state.system_prompt = new_system_prompt
                st.success("Assistant instructions updated!")

        # Clear conversations button with custom styling
        st.markdown("""
            <style>
            div[data-testid="stHorizontalBlock"] {
                background-color: #dc3545;
                border-radius: 0.5rem;
                margin-top: 1rem;
            }
            </style>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear All Conversations", use_container_width=True):
            clear_chat()

    st.title("Yukti's AI Chat Assistant")
    
    # Welcome message for empty chat
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <h1>How can I help you today?</h1>
            <p>Start a conversation by typing a message below.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Message display container
    chat_container = st.container()
    with chat_container:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            role = message["role"]
            content = get_message_content(message)
            avatar_icon = '👤' if role == 'user' else '🤖'
            
            st.markdown(f"""
                <div class="chat-message {role}">
                    <div class="avatar">{avatar_icon}</div>
                    <div class="message-content">{content}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat input and response handling
    if prompt := st.chat_input("Type your message here...", key="chat_input"):
        if not st.session_state.waiting_for_response:
            ensure_event_loop()  # Ensure event loop is available
            
            # Add user message
            st.chat_message("user").write(prompt)
            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
            st.session_state.messages.append(user_message)
            st.session_state.waiting_for_response = True

            try:
                # Prepare messages for API with document context
                document_context = get_document_context()
                context_message = {
                    "role": "system",
                    "content": f"{st.session_state.system_prompt}\n\n{document_context}" if document_context else st.session_state.system_prompt
                }
                
                # If there's document context, add it as context message
                if document_context:
                    context_note = {
                        "role": "system",
                        "content": "The following conversation will be about the document provided above. Use this context to answer questions."
                    }
                    api_messages = [
                        context_message,
                        context_note,
                        *[format_message_for_api(msg) for msg in st.session_state.messages]
                    ]
                else:
                    api_messages = [
                        context_message,
                        *[format_message_for_api(msg) for msg in st.session_state.messages]
                    ]

                # Create assistant message container
                assistant_response = st.chat_message("assistant")
                response_placeholder = assistant_response.empty()
                
                # Stream the response with retry mechanism
                full_response = ""
                with st.spinner('Thinking...'):
                    response = get_ai_response(api_messages, "microsoft/phi-3-medium-128k-instruct:free")
                    
                    try:
                        for chunk in response:
                            if chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                                response_placeholder.markdown(full_response)
                    except Exception as e:
                        logger.error(f"Error streaming response: {str(e)}")
                        # Retry streaming if it fails
                        if not full_response:
                            response = get_ai_response(api_messages, "microsoft/phi-3-medium-128k-instruct:free")
                            for chunk in response:
                                if chunk.choices[0].delta.content:
                                    full_response += chunk.choices[0].delta.content
                                    response_placeholder.markdown(full_response)

                # Save assistant response
                assistant_message = {
                    "role": "assistant",
                    "content": [{"type": "text", "text": full_response}]
                }
                st.session_state.messages.append(assistant_message)

                # Update chat title if first message
                if len(st.session_state.messages) == 2:
                    st.session_state.chat_titles[st.session_state.current_chat_id] = get_chat_title([user_message])

                # Save conversation
                st.session_state.loaded_chats[st.session_state.current_chat_id] = {
                    "messages": st.session_state.messages,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "title": st.session_state.chat_titles[st.session_state.current_chat_id]
                }
                save_conversation(vector_store, st.session_state.messages, st.session_state.current_chat_id)

            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                st.error("Failed to generate response. Please try again.")
                st.session_state.messages.pop()  # Remove failed user message
            
            finally:
                st.session_state.waiting_for_response = False

    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    initialize_app()
    try:
        chat_interface()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        if "no running event loop" in str(e):
            ensure_event_loop()
            chat_interface()
        else:
            raise e