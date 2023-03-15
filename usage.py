from Llmqa import Llmqa

qa = Llmqa('example_data/')

qa.make_vectorstore()

qa.ask_question('hello there')