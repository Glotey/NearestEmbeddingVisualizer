import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

#Load the Model and Vector Store
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import pickle

#THING YOU WANT TO ASK:
question = "Hello"
print("Query to be matched -- ",question)
'''
## UNCOMMENT BELOW CODE IF YOU WANT TO RUN THE CODE USING ONLINE FILES,
## I HAVE RUN IT ONCE AND STORED DATA OFFLINE BELOW IN PIKCLE OBJECT

# url = "https://www.freenews.fr/"
# loader = RecursiveUrlLoader(
#     url=url, max_depth=5, extractor=lambda x: Soup(x, "lxml").text
# )
# documents = loader.load()

#Temp code to save loaded data:
# # Open a file and use dump() 
# with open('file.pkl', 'wb') as file: 
#     # A new file will be created 
#     pickle.dump(documents, file)
'''

# Open the file in binary mode 
with open('file.pkl', 'rb') as file:       
    # Call load method to deserialze 
    documents = pickle.load(file)

print("Document Loaded . . .")
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0
        )
texts = text_splitter.split_documents(documents)

modelPath = "sentence-transformers/distiluse-base-multilingual-cased-v1"
model_kwargs = {'device':'cpu'}
#encode_kwargs = {'normalize_embeddings': False}
'''
## I HAVE STORED EMBEDDINGS OFFLINE TOO, UNCOMMENT TO AGAIN DOWNLOAD THEM
# embeddings = HuggingFaceEmbeddings(model_name=modelPath,model_kwargs=model_kwargs)
# with open('embeddings.pkl', 'wb') as file: 
#     # A new file will be created 
#     pickle.dump(embeddings, file)
'''
# Open the file in binary mode 
with open('embeddings.pkl', 'rb') as file:   
    # Call load method to deserialze 
    embeddings = pickle.load(file)
print("Embedding Generated . . .")
# SAVE (USE THIS ONLY FOR THE FIRST TIME TO STORE DATA in DB)
# docs_vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db_multilingual")
print("Embedding stored in ChromaDB . . .")
# Providing link to CromaDB
modelPath = "sentence-transformers/distiluse-base-multilingual-cased-v1"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings_model = HuggingFaceEmbeddings(model_name=modelPath, model_kwargs=model_kwargs)
docs_vectorstore = Chroma(persist_directory="./chroma_db_multilingual", embedding_function=embeddings_model)

# Fetching and Preparing data
response = docs_vectorstore.get(include=["metadatas", "documents", "embeddings"])
df = pd.DataFrame({
 "id": response["ids"],
 "source": [metadata.get("source") for metadata in response["metadatas"]],
 "page": [metadata.get("page", -1) for metadata in response["metadatas"]],
 "document": response["documents"],
 "embedding": response["embeddings"],
})
df["contains_answer"] = df["document"].apply(lambda x: "Nœud Répartition Optique)" in x)

#Calculating Vector match score
# Here we use Euclidean match
question_embedding = embeddings_model.embed_query(question)
df["dist"] = df.apply(
    lambda row: np.linalg.norm(
        np.array(row["embedding"]) - question_embedding
    ),
    axis=1,
)

## UNCOMMENT THIS TO TRY SPOTLIGHT . . . 
#Visualize using spotlight
# from renumics import spotlight
# spotlight.show(df)


#Vizualize using UMAP
import umap
import matplotlib.pyplot as plt
# Find the  5 closest vectors
closest_vectors_indices = df.nsmallest(5, 'dist')['id'].values

# Prepare the embeddings for UMAP
embeddings = np.array([np.array(x) for x in df["embedding"]])

# Reduce dimensionality with UMAP
reducer = umap.UMAP()
embedding_reduced = reducer.fit_transform(embeddings)

# Plot the reduced embeddings
plt.scatter(embedding_reduced[:,  0], embedding_reduced[:,  1], c='gray', alpha=0.2)

# Highlight the question embedding and the  5 closest vectors
plt.scatter(embedding_reduced[df["id"].isin(closest_vectors_indices),  0], embedding_reduced[df["id"].isin(closest_vectors_indices),  1], c='red', alpha=1)
plt.scatter(embedding_reduced[df["id"] == df[df["dist"] == df["dist"].min()]["id"].values[0],  0], embedding_reduced[df["id"] == df[df["dist"] == df["dist"].min()]["id"].values[0],  1], c='blue', alpha=1, marker='*')

# Add labels and title
plt.title("UMAP Visualization of Text Embeddings with Question Highlighted")
plt.xlabel("UMAP  1")
plt.ylabel("UMAP  2")
plt.show()
print("Add a debugger on this line to see results from above file . . . .")