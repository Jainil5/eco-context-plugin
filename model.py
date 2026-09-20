from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="gpt-oss:20b-cloud", temperature=0.2)

embedding = OllamaEmbeddings(model="nomic-embed-text")
