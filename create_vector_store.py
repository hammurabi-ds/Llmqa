from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pickle



def get_data():
    """
    
    """

    raise NotImplementedError

def split_text(data, sources):
    """
    Here we split the documents into smaller chunks.
    We do this due to the context limits of the LLMs.
    """

    print('2: splitting text..')

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs = []
    metadatas = []
    for i, d in enumerate(data):
        splits = text_splitter.split_text(d)
        docs.extend(splits)
        metadatas.extend([{"source": sources[i]}] * len(splits))
    print('2: done')
    return docs, metadatas

def make_vectorstore(docs, metadatas):
    """
    Here we create a vector store from the documents and save it to disk.
    """
    print('3: creating vectorstore..')
    embeddings = HuggingFaceEmbeddings()

    store = FAISS.from_texts(docs, embeddings, metadatas=metadatas)
    faiss.write_index(store.index, "storage/docs.index")
    store.index = None
    with open("storage/faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)
    print('3: done')

if __name__=='__main__':
    d, s = get_data()
    ds, ms = split_text(d,s)
    make_vectorstore(ds,ms)


