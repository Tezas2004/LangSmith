import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith settings
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Sequential_LLM_Chain"

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)

# Chat model
model = ChatHuggingFace(llm=llm)

# Prompt 1
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2
prompt2 = PromptTemplate(
    template="Generate a 5 point summary from the following text:\n{text}",
    input_variables=["text"]
)

# Output parser
parser = StrOutputParser()

# Sequential chain
chain = prompt1 | model | parser | prompt2 | model | parser

config={'tags': ['sequential-chain','report_generation','summarization'],
        'metadata': {'author': 'Tejas Dwivedi', 'description': 'A sequential chain that generates a report and then summarizes it.'},
        'run_name': 'sequential_chain_run'}

# Invoke chain
result = chain.invoke({
    "topic": "Unemployment in India"}, config=config)

print(result)