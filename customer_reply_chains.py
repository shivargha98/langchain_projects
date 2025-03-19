##Chaining strategy : branching###
### Classifying the customer feedbacks, replying back in 4 different cases####

import os
from env_variables import *
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
llm_model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash',
                                    temperature = 0,
                                    max_tokens=None,
                                    timeout=None,
                                    max_retries=2)

### classify the customer feedback ###
classification_template = ChatPromptTemplate.from_messages(
    [
        ('system','You are an AI assistant that classifies the sentiment of the text or feedback'),
        ('human','Classify the {feedback} as positive,negative,neutral or escalate')

    ]
)

## positive feedback reply ##
positive_reply = ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful AI assistant, which helps in writing 2-line replies'),
        ('human','Generate a 2 line reply for a positive {feedback} from a customer')
    ]
)

### negative feedback reply ###
negative_reply = ChatPromptTemplate.from_messages(
[

    ('system','You are a helpful AI assistant, which helps in writing 2-line replies'),
    ('human','Generate a 2 line reply for a negative {feedback} from a customer')
])

### neutral feedback reply ####
neutral_reply = ChatPromptTemplate.from_messages(
[

    ('system','You are a helpful AI assistant, which helps in writing 2-line replies'),
    ('human','Generate a 2 line reply for a neutral {feedback} from a customer')
]
)


escalate_reply = ChatPromptTemplate.from_messages(
[
    ('system','You are a helpful AI assistant, which helps in writing 2-line replies'),
    ('human','Generate a 2 line reply saying escalated to customer support for this {feedback}')
])

###### creating the branches ####
branches = RunnableBranch(
    (
        lambda x : "positive" in x.lower(),
        positive_reply | llm_model | StrOutputParser()
    ),
    (
        lambda x : "negative" in x.lower(),
        negative_reply | llm_model | StrOutputParser()
    ),
    (
        lambda x : "neutral" in x.lower(),
        neutral_reply | llm_model | StrOutputParser()
    ),
    
    escalate_reply | llm_model | StrOutputParser()
    
)

### Feedback classification chain ###
classification_chain = classification_template | llm_model | StrOutputParser()

### Chain for reply #########
reply_chain = classification_chain | branches


feedback = input('Write a feedback for a response: ')
classification = classification_chain.invoke({'feedback':feedback})
print("Review Classification: ",classification)
reply = reply_chain.invoke({'feedback':feedback})
print(reply)