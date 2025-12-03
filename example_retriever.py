#################################################################################################################################################################
###############################   1.  IMPORTING MODULES AND INITIALIZING VARIABLES   ############################################################################
#################################################################################################################################################################

import os

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from discover_ollama import discover_ollama

pd.options.mode.chained_assignment = None

load_dotenv()

###############################   INITIALIZE EMBEDDINGS MODEL  #################################################################################################

embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL"),
    base_url=discover_ollama()
)

###############################   INITIALIZE CHROMA VECTOR STORE   #############################################################################################

vector_store = Chroma(
    collection_name=os.getenv("COLLECTION_NAME"),
    embedding_function=embeddings,
    persist_directory=os.getenv("DATABASE_LOCATION"),
)

results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("what is langchain"), k=5
)

for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")