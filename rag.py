# 📦 Importa le librerie necessarie
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# 📄 STEP 1: Carica il PDF
loader = PyPDFLoader("CV_Massimiliano_Armato.pdf")
documents = loader.load()

# ✂️ STEP 2: Spezza il contenuto in chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# 🧬 STEP 3: Crea gli embedding con Ollama
embedding_model = OllamaEmbeddings(model="mistral")

# 🗃️ STEP 4: Salva i chunk in un vector store
vectorstore = Chroma.from_documents(chunks, embedding_model)

# 🔍 STEP 5: Crea il retriever
retriever = vectorstore.as_retriever()

# 🧠 STEP 6: Inizializza il modello LLM
llm = OllamaLLM(model="mistral")

# 🔗 STEP 7: Costruisci la catena RAG
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ❓ STEP 8: Fai la domanda
query = "Qual è l'occupazione desiderata?"
risposta = rag_chain.run(query)

# 🖨️ STEP 9: Stampa la risposta
print("Domanda:", query)
print("Risposta:", risposta)