# MultiPDF Chat App

## Introduction
------------
The MultiPDF Chat App is a Python application that allows you to chat with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.

## How It Works
------------

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

The application follows these steps to provide responses to your questions:

1. PDF Loading: The app reads multiple PDF documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

## Dependencies and Installation
----------------------------
To install the MultiPDF Chat App, please follow these steps:
1. Download [Ollama library](https://github.com/jmorganca/ollama)
   ```
   curl https://ollama.ai/install.sh | sh
   ```
2. pull the chat model we will use [LLAMA2](https://ollama.ai/library/llama2)
   ```
   ollama pull llama2:70b-chat
   ```
3. Create new enviroment with python 3.9 and activate it, in this case we will use conda
   ```
   conda create -n chat-with-pdf python=3.9
   ```
   ```
   conda activate chat-with-pdf
   ```


4. Clone the repository to your local machine.
   ```
   git clone https://github.com/jorge-armando-navarro-flores/chat-with-multiple-PDFs-LLAMA2.git
   ```
   ```
   cd chat-with-multiple-PDFs-LLAMA2
   ```

5. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```


## Usage
-----
To use the MultiPDF Chat App, follow these steps:

1. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

2. The application will launch in your default web browser, displaying the user interface.
![Upload PDF](./docs/interface.png)

3. Load multiple PDF documents into the app by following the provided instructions.
![Upload PDF](./docs/upload_PDF.png)

4. Ask questions in natural language about the loaded PDFs using the chat interface.
![Ask to PDF](./docs/ask_to_PDF.png)
