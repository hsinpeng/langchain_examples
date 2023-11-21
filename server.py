#!/usr/bin/env python
import sys
import json
from typing import List
import uvicorn
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import BaseOutputParser
from langserve import add_routes

###### Azure OpenAI Settings #######
with open('param.json', 'r', encoding='utf-8') as param_file:
    param_data = json.load(param_file)
    azure_apikey = param_data["azure_apikey"]
    azure_apibase  = param_data["azure_apibase"]
    azure_apitype = param_data["azure_apitype"]
    azure_apiversion = param_data["azure_apiversion"]
    azure_gptx_deployment = param_data["azure_gptx_deployment"]
    azure_embd_deployment = param_data["azure_embd_deployment"]
param_file.close()
####################################

class CommaSeparatedListOutputParser(BaseOutputParser[List[str]]):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")
    
def main():
    try:
        print(f"LangServe")
        
        # 1. Chain definition
        template = """You are a helpful assistant who generates comma separated lists.
        A user will pass in a category, and you should generate 5 objects in that category in a comma separated list.
        ONLY return a comma separated list, and nothing more."""
        human_template = "{text}"

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            ("human", human_template),
        ])

        chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, azure_endpoint=azure_apibase, temperature=0.9)

        category_chain = chat_prompt | chatllm | CommaSeparatedListOutputParser()

        # 2. App definition
        app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="A simple api server using Langchain's Runnable interfaces",
        )

        # 3. Adding chain route
        add_routes(
            app,
            category_chain,
            path="/category_chain",
        )

        uvicorn.run(app, host="localhost", port=8000)

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())