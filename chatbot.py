import streamlit as st
from langchain_astradb import AstraDBVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from datetime import datetime
import json
from typing import List, Dict
import uuid
from openai import OpenAI
import time
from datetime import datetime, timezone
import logging
import os
import pickle

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add local storage path
LOCAL_STORAGE_PATH = os.path.join(os.path.dirname(__file__), 'chat_history')
os.makedirs(LOCAL_STORAGE_PATH, exist_ok=True)

# Streamlit page config
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🌠",
    layout="wide"
)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
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
        align-items: flex-start;
        gap: 10px;
        padding: 1rem;
        border-radius: 0.5rem;
        max-width: 80%;
        margin: 0.5rem 0;
    }
    .chat-message.user {
        background-color: #2b2d31;
        margin-left: auto;
        flex-direction: row-reverse;
    }
    .chat-message.assistant {
        background-color: #444654;
        margin-right: auto;
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

# AstraDB setup
def setup_astradb():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_store = AstraDBVectorStore(
        collection_name="chatbot",
        embedding=embeddings,
        token=ASTRA_DB_TOKEN,
        api_endpoint=f"https://67582ca9-0ffe-4d01-92e4-f79922d4e517-us-east-2.apps.astra.datastax.com"
    )
    return vector_store

# Modified save_conversation function with fallback
def save_conversation(vector_store, messages: List[Dict], conversation_id: str):
    conversation_doc = Document(
        page_content=json.dumps(messages),
        metadata={
            "conversation_id": conversation_id,
            "timestamp": datetime.utcnow().isoformat(),
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
    
    # Try loading from AstraDB
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
    st.session_state.messages = []
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.chat_titles = {}

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

def load_all_conversations(vector_store):
    results = vector_store.similarity_search(
        "all conversations",
        k=50,
        filter={}
    )
    conversations = {}
    for result in results:
        conv_id = result.metadata.get("conversation_id")
        if conv_id and conv_id not in conversations:
            messages = json.loads(result.page_content)
            conversations[conv_id] = {
                "messages": messages,
                "timestamp": result.metadata.get("timestamp"),
                "title": get_chat_title(messages)
            }
    return conversations

def switch_conversation(conv_id):
    if conv_id in st.session_state.loaded_chats:
        st.session_state.messages = st.session_state.loaded_chats[conv_id]["messages"]
        st.session_state.current_chat_id = conv_id
    else:
        create_new_chat()

# Chat interface
def chat_interface():
    try:
        vector_store = setup_astradb()
    except Exception as e:
        logger.error(f"Failed to setup AstraDB: {str(e)}")
        vector_store = None
        st.error("Failed to connect to database. Operating in local-only mode.")

    # Reset messages on page refresh
    if st.session_state.page_refreshed:
        st.session_state.messages = []
        st.session_state.page_refreshed = False

    # Load all conversations if not loaded
    if not st.session_state.loaded_chats:
        st.session_state.loaded_chats = load_all_conversations(vector_store)
        if st.session_state.loaded_chats:
            latest_chat = max(st.session_state.loaded_chats.items(), 
                            key=lambda x: x[1]["timestamp"])
            st.session_state.current_chat_id = latest_chat[0]
            st.session_state.messages = latest_chat[1]["messages"]

    # Enhanced sidebar with chat history
    with st.sidebar:
        st.title("💬 Chat History")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button("+ New Chat", key="new_chat", use_container_width=True):
                create_new_chat()
        
        st.divider()
        st.title("📚 Chats")
        # Display chat history
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
        # Settings
        st.title("⚙️ Settings")
        model = "google/gemini-2.0-flash-thinking-exp:free"
        
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

    st.title("AI Chat Assistant")
    
    # Display welcome message if no messages
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align: center; margin-top: 50px;">
            <h1>How can I help you today?</h1>
            <p>Start a conversation by typing a message below.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create a container for the main chat area
    main_container = st.container()
    with main_container:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        for message in st.session_state.messages:
            message_content = get_message_content(message)
            avatar = "👤" if message['role'] == 'user' else "🤖"
            st.markdown(f"""
            <div class="chat-message {message['role']}">
                <div class="avatar">{avatar}</div>
                <div class="message-content">{message_content}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Create a container for the input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    if prompt := st.chat_input("Type your message here..."):
        # Add and display user message immediately
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
        st.session_state.messages.append(user_message)
        
        # Display user message
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar">👤</div>
            <div class="message-content">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)

        # Update chat title if first message
        if len(st.session_state.messages) == 1:
            st.session_state.chat_titles[st.session_state.current_chat_id] = get_chat_title([user_message])

        # Format all messages for API to maintain context
        api_messages = [format_message_for_api(msg) for msg in st.session_state.messages]

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model=model,
                messages=api_messages,
                stream=True,
                extra_headers={
                    "HTTP-Referer": "https://your-site.com",
                    "X-Title": "AI Chat Assistant"
                }
            )

            # Stream the response
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(f"""
                    <div class="chat-message assistant">
                        <div class="avatar">🤖</div>
                        <div class="message-content">{full_response}</div>
                    </div>
                    """, unsafe_allow_html=True)
                time.sleep(0.01)

        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": full_response}]
        }
        st.session_state.messages.append(assistant_message)

        # Update loaded chats and save to DB
        st.session_state.loaded_chats[st.session_state.current_chat_id] = {
            "messages": st.session_state.messages,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "title": st.session_state.chat_titles[st.session_state.current_chat_id]
        }
        save_conversation(vector_store, st.session_state.messages, st.session_state.current_chat_id)
        
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    chat_interface()