<img width="1437" alt="Screenshot 2025-04-09 at 12 27 10 AM" src="https://github.com/user-attachments/assets/8b847233-50a2-465d-914f-53c9e66c5614" />



# Customizable Streamlit Chatbot

A versatile chatbot application built with Streamlit that allows users to customize the chatbot's personality and query uploaded documents.

## Features

- **Personality Customization**: Adjust the chatbot's tone, style, and behavior to suit your preferences
- **Document Upload**: Support for PDF, DOCX, TXT, and CSV files
- **Context-Aware Responses**: Ask questions about your uploaded documents and receive relevant answers
- **User-Friendly Interface**: Clean, intuitive design for seamless interaction

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to the provided URL (typically http://localhost:8501)
3. Upload documents using the file uploader
4. Customize the chatbot's personality using the sidebar options
5. Start chatting!

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- OpenAI API key (or alternative LLM provider)
- PyPDF2 (for PDF processing)
- python-docx (for DOCX processing)
- pandas (for CSV processing)

## Configuration

Set your API key in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## How It Works

The application uses a combination of document processing technologies and language models to analyze uploaded files and generate responses. The personality customization leverages prompt engineering to modify the chatbot's behavior according to user preferences.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
