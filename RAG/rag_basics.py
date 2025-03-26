"""
RAG basics --> TextSplitter,Embedding,Chroma Vector DB, Retrival and Gemini 

"""
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

###Filepaths##
current_dir = os.path.dirname(os.path.abspath('/home/shivargha/langchain_projects/RAG/'))+"/RAG/"
print(current_dir)
filepath = os.path.join(current_dir,"books","odyssey.txt")
persistent_dir = os.path.join(current_dir,"db","chroma_db")

#############
###Huggingface Embeddings###
embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-mpnet-base-v2')
##############

##check for persistent dir/chroma db##
if not os.path.exists(persistent_dir):
    print("Vector store initialisated")

    ##Read the text from the file##
    loader = TextLoader(filepath)
    documents = loader.load()

    ##Split the document into chunks##
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    #Print Doc information##
    #print("\n----Doc Chunks Info----")
    print("Number of chunks: ",len(docs))
    #print("Sample: ",docs[0].page_content)

    
    ###creating the vector store###
    chroma_db = Chroma.from_documents(docs,
                                    embeddings,
                                    persist_directory = persistent_dir)
    print("Persistent directory created,embeddings created")

else:
    print("Vector Store already exists")
    chroma_db = Chroma(persist_directory = persistent_dir,
                        embedding_function = embeddings)
 ###Retriver section###
query = "Who is Odysseus?"
retriever = chroma_db.as_retriever(
                search_type = "similarity_score_threshold",
                search_kwargs = {"k":3,"score_threshold":0.4}
                ##k value is for k nearest/similar documents
                ##score_threshold : documents with less than 0.4 value,will be filtered out
)
nearest_docs = retriever.invoke(query)
print("Nearest/Similar Documents: \n")
print(nearest_docs)