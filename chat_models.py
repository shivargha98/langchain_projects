from langchain_google_genai import ChatGoogleGenerativeAI
from env_variables import *
import os
from langchain.schema import AIMessage,HumanMessage,SystemMessage

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
llm_model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash',
                                    temperature = 0,
                                    max_tokens=None,
                                    timeout=None,
                                    max_retries=2)
 
### model with chat history ###
chat_history = []
system_message = SystemMessage(content="You are a helpful AI assistant, which returns the capital or currency of the country that I ask for.")
chat_history.append(system_message)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query)) ###User message###


    result = llm_model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content = response))

    print("AI: ",response)

print("Message History: ")
print(chat_history)