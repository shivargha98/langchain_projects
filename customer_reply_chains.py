##Chaining strategy : branching###
import os
from env_variables import *
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
llm_mode = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash',
                                    temperature = 0,
                                    max_tokens=None,
                                    timeout=None,
                                    max_retries=2)

