# Lokale Agentic RAG Pipeline in einem Chatbot

## Schnellstes setup

1. VS Code nutzen
2. dieses Projekt im devcontainer öffnen
3. ollama serve
4. ollama run llama3.1:8b
5. ollama run mxbai-embed-large
1.
1.  `python ./2_chunking_embedding_ingestion.py`
7. Starte den chatbot mit `streamlit run ./3_chatbot.py`
8. Öffne den Bot im Browser unter http://localhost:8501

![](chatbot.png)

## Scraping nur mit eigenem API Key von https://brightdata.de

Ich hab bereits einige Dateien von wikipedia ausgelesen und die data.txt eingecheckt