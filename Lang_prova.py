from langchain_ollama import OllamaLLM

# Inizializza il modello locale
llm = OllamaLLM(model="mistral")

# Fai una domanda
domanda = "Di che colore è la fragola?"
risposta = llm.invoke(domanda)

# Stampa la risposta
print("Domanda:", domanda)
print("Risposta:", risposta)