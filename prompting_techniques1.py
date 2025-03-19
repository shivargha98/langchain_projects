'''
Components of Prompting:
1. Instructions
2. Example Input
3. Example Output
4. Query
'''
from langchain_google_genai import ChatGoogleGenerativeAI
from env_variables import *
from langchain.prompts import ChatPromptTemplate
import os


os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
llm_model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash',
                                    temperature = 0,
                                    max_tokens=None,
                                    timeout=None,
                                    max_retries=2)

### Prompting with Multiple Placeholders ###
template_multiple = """You are a helpful AI assistant, you write 2-line emails on a particular topic to a particular person.
Human: Write an email for {topic} to {person_name}.
Assistant:"""
prompt_mul = ChatPromptTemplate.from_template(template_multiple)
query_topic = input("Email Topic?: ")
query_person_name = input("Write Email to which person?: ")
prompt = prompt_mul.invoke({"topic":query_topic,"person_name":query_person_name})

result = llm_model.invoke(prompt)
print(result.content)

#### Prompting using tuples ####
messages = [
('system','You are an AI assistant, that converts one currency to another'),
('human','Convert {value},{currency1} to {currency2}')
]
value = input("Enter the value you want to convert?")
currency1 = input("Enter the currency:")
currency2 = input("Enter the currency to convert to:")

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"value":value,"currency1":currency1,"currency2":currency2})
result = llm_model.invoke(prompt)
print(result.content)