###using few shot prompting to create more JEE questions###
import os
from env_variables import *
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,FewShotPromptTemplate

os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
llm_model = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash',
                                    temperature = 0,
                                    max_tokens=None,
                                    timeout=None,
                                    max_retries=2)

##setting an example prompt template###
mcq_formatter_template = """MCQ_context:{mcq_context}
"subject":{subject}
"level":{level}
"aIanswer":{ai_answer}
"""

example_mcq_prompt = PromptTemplate.from_template(mcq_formatter_template)

example_fewshot = [{
    "mcq_context":"Generate Easy-Medium JEE-Mains level Question for students in 12th Standard.",
    "subject":"Chemistry",
    "level": "Easy",
    "ai_answer": "What is the atomic number of oxygen? A. 5 B. 8 C. 12 D. 15 Answer: B"

},
{
    "mcq_context":"Generate Easy-Medium JEE-Mains level Question for students in 12th Standard.",
    "subject":"Maths",
    "level": "Medium",
    "ai_answer": "If the vectors a = i + j and b = i - j are perpendicular, then the dot product a and b is? A. 0 B. 1 C. 2 D. -1 Answer: A"

},
{
    "mcq_context":"Generate Easy-Medium JEE-Mains level Question for students in 12th Standard.",
    "subject":"Physics",
    "level": "Medium",
    "ai_answer": "A lens of focal length 20 cm forms an image at 30 cm from the lens. The object distance is:? A. 12cm B. 15cm C. 60cm D. 90cm Answer: C"

}]

print('Test the formatting of the few shot examples:')
print(example_mcq_prompt.invoke(example_fewshot[0]).to_string())

fewshot_prompt = FewShotPromptTemplate(

    examples = example_fewshot,
    example_prompt = example_mcq_prompt,
    prefix='You are an AI assistant, that generates Easy-Medium JEE level questions',
    suffix="MCQ_context:{mcq_context} subject:{subject} level:{level} ai_answer:",
    input_variables = ['mcq_context','subject','level'],
    example_separator="\n"
)

# print(
#     fewshot_prompt.invoke({'mcq_context':'Generate Easy-Medium JEE-Mains level Question for students in 12th Standard',
#                             'subject':'Chemistry',
#                             'level':'easy',
#                             }).to_string()
# )
subject_ip = input("Enter the Subject Required: ")
level_ip = input("Enter the level required: ")
prompt = fewshot_prompt.invoke({'mcq_context':'Generate Easy-Medium JEE-Mains level Question for students in 12th Standard',
                             'subject':subject_ip,
                             'level':level_ip,
                             })
result = llm_model.invoke(prompt)
print('New Question Generated:')
print(result.content)