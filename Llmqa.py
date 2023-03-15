from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import OpenAI
from langchain import VectorDBQA
from langchain import PromptTemplate
from pathlib import Path

class Llmqa(object):

    def __init__(self, text_path):

        self.text_path = text_path
        self.data = []
        self.docs = []

    def _get_data(self):
        """
        load text data in markdown format
        """
        print('1: Getting data..')
        ps = list(Path(self.text_path).glob("**/*.md"))
        for p in ps:
            with open(p) as f:
                self.data.append(f.read())
        print('1: done')

    def _split_text(self):
        """
        Here we split the documents into smaller chunks.
        We do this due to the context limits of the LLMs.
        """

        print('2: splitting text..')

        text_splitter = CharacterTextSplitter(chunk_size=200, separator="\n")
        for i, d in enumerate(self.data):
            splits = text_splitter.split_text(d)
            self.docs.extend(splits)

        print('2: done')

    def make_vectorstore(self):
        """
        Here we create a vector store from the documents.
        """
        self._get_data()
        self._split_text()
 
        print('3: creating vectorstore..')
        embeddings = HuggingFaceEmbeddings()
        self.vectordb = FAISS.from_texts(self.docs, 
                                    embeddings)
        
    def _get_prompt_template(self):
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
        self.chain_type_kwargs = {"prompt": prompt}

    
    def _get_chain(self):
        """
        Initialize the language chain
        """

        self._get_prompt_template()

        self.chain = VectorDBQA.from_chain_type(llm=OpenAI(model_name='text-davinci-003',
                                                            temperature=0, 
                                                            max_tokens=256), 
                                            chain_type="stuff", 
                                            vectorstore=self.vectordb, 
                                            chain_type_kwargs=self.chain_type_kwargs,
                                            k=3)
    
    def ask_question(self, question):
        self._get_chain()
        result = self.chain.run(question)
        print("_______________________________________")
        print(f"Answer: {result}")


        


