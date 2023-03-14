import faiss
from langchain import HuggingFaceHub
from langchain import OpenAI
from langchain import VectorDBQA
from langchain.chains import VectorDBQAWithSourcesChain
import pickle
import argparse
from langchain import PromptTemplate
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def get_data():
    """
    Get the vector store data
    """
    index = faiss.read_index("storage/docs.index")

    with open("storage/faiss_store.pkl", "rb") as f:
        store = pickle.load(f)

    store.index = index
    return store

def _get_prompt_template():
    """
    Creates a custom LLM prompt template
    """
    temp = """
    Given the following context and a question, create a 1 to 2 sentences answer. 
    
    Try to extract the answer from the context.
    
    return a "Source" part in your answer if possible.

    If you don't know the answer, just say that you don't know. 
    
    Context: {context} 
    
    Question: {question}
    """

    prompt = PromptTemplate(input_variables=["question", "context"], 
                            template=temp)
    chain_type_kwargs = {"prompt": prompt}
    return chain_type_kwargs

def get_chain(vector_store, 
              model_kwargs=None):
    """
    Initialize the language chain

    :param vector_store: your vectorstore for knowledge database
    :param model_kwargs: arguments for the huggingface models
    """

    chain_type_kwargs = _get_prompt_template()

    chain = VectorDBQA.from_chain_type(llm=OpenAI(model_name='text-davinci-003',
                                                        temperature=0, 
                                                        max_tokens=256), 
                                        chain_type="stuff", 
                                        vectorstore=vector_store, 
                                        chain_type_kwargs=chain_type_kwargs,
                                        k=3)

    return chain

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Ask a question')
    parser.add_argument('--question', type=str, help='The question to ask')

    args = parser.parse_args()

    vector_store = get_data()

    language_chain = get_chain(args.type,
                               vector_store,
                               args.model_name,)

    result = language_chain.run(args.question)
    print("_______________________________________")
    print(f"Answer: {result}")
