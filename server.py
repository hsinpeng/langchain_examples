#!/usr/bin/env python
import os
import sys
import json
from operator import itemgetter
from typing import List, Optional
import uvicorn
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

###### Azure OpenAI Settings #######
with open('param.json', 'r', encoding='utf-8') as param_file:
    param_data = json.load(param_file)
    hostname = param_data["hostname"]
    hostport  = param_data["hostport"]
    azure_apikey = param_data["azure_apikey"]
    azure_apibase  = param_data["azure_apibase"]
    azure_apitype = param_data["azure_apitype"]
    azure_apiversion = param_data["azure_apiversion"]
    azure_gptx_deployment = param_data["azure_gptx_deployment"]
    azure_embd_deployment = param_data["azure_embd_deployment"]
    redis_url = param_data["redis_url"]
param_file.close()
####################################

os.environ['NO_PROXY'] = 'localhost, 127.0.0.1'

def get_history(id):
    ret = RedisChatMessageHistory(session_id=id, url=redis_url).messages
    #ret = ''.join(str(e.content) for e in (RedisChatMessageHistory(session_id=id, url=REDIS_URL).messages))
    #print(ret)
    return ret

def _save_history(id, question, answer):
    try:
        history = RedisChatMessageHistory(id, url=redis_url)
        history.add_user_message(question)
        history.add_ai_message(answer)
        return True
    except ValueError as ve:
        return False

def save_history(_dict):
    return _save_history(_dict["param1"], _dict["param2"], _dict["param3"])

def run_server(llm_chain_list, api_name_list):
    try:
        ### App definition ###
        app = FastAPI(
        title="CHT-TL E21 LLM Back-end Server",
        version="1.0",
        description="A LLM API Server using Langchain's Runnable interfaces",
        )

        ### Adding chain route ###
        for chain, name in zip(llm_chain_list, api_name_list):
            add_routes(
                app,
                chain,
                path=name,
            )

        ### Run server ###
        uvicorn.run(app, host=hostname, port=hostport)
    
    except ValueError as ve:
        return str(ve)


def main():
    try:
        ##### Models #####
        chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, azure_endpoint=azure_apibase, temperature=0)
        #embdllm = AzureOpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, azure_endpoint=azure_apibase)
        output_parser = StrOutputParser()

        # API 1: General query (without chat history)
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful chatbot. Try to answer in {language}"),
            ("human", "{question}"),
        ])

        query_chain = query_prompt | chatllm | output_parser
        
        # API 2: General chat (with history)
        chat_history_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful chatbot."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        chat_history_chain = chat_history_prompt | chatllm | output_parser
       
        chat_chain = RunnableWithMessageHistory(
            chat_history_chain,
            lambda session_id: RedisChatMessageHistory(session_id, url=redis_url),
            input_messages_key="question",
            history_messages_key="history",
        )

        # API 3: Cypher Generator
        CYPHER_GENERATION_TEMPLATE = """
        Task:Generate Cypher statement to query a graph database.
        Instructions:
        Use only the provided relationship types and properties in the schema.
        Do not use any other relationship types or properties that are not provided.
        Schema:
        {db_schema}
        Cypher examples:
        # How many streamers are from Norway?
        MATCH (s:Stream)-[:HAS_LANGUAGE]->(:Language {{name: 'no'}})
        RETURN count(s) AS streamers

        Note: Do not include any explanations or apologies in your responses.
        Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
        Do not include any text except the generated Cypher statement.

        The question is:
        {question}"""
        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["db_schema", "question"], template=CYPHER_GENERATION_TEMPLATE
        )

        cypher_chain = (
            {
                "db_schema": itemgetter("db_schema"),
                "question": itemgetter("question"),
            }
            | CYPHER_GENERATION_PROMPT | chatllm | output_parser
        )

        # API 4: History Retrieve
        history_retrieve_chain = itemgetter("session_id") | RunnableLambda(get_history)

        # API 5: History Store
        history_store_chain = {"param1": itemgetter("session_id"), "param2": itemgetter("question"), "param3": itemgetter("answer")} | RunnableLambda(save_history)

        # API 6: Customized Query 1 (with history and "obliged" knowledge)
        history_obliged_knowledge_query_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful chatbot"),
                MessagesPlaceholder(variable_name="history"),
                ("human", """Task: Answer the question based on the knowledge.
                Instructions:
                please answer the question in {language}.
                you should answer the question based on the knowledge only.
                Use only the provided facts and information in the knowledge.
                Do not use any other facts or information that are not provided.
                Knowledge:
                {knowledge}

                The question is:
                {question}"""),
            ]
        )

        custom_obliged_query_chain = history_obliged_knowledge_query_prompt | chatllm | output_parser
        
        # API 7: Customized Query 2 (with history and "referential" knowledge)
        history_referential_knowledge_query_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful chatbot"),
                MessagesPlaceholder(variable_name="history"),
                ("human", """Task: Answer the question as the best you can do.
                Instructions:
                please answer the question in {language}.
                The knowledge is only referential, you should answer as the best you can do.
                Knowledge:
                {knowledge}

                The question is:
                {question}"""),
            ]
        )

        custom_referential_query_chain = history_referential_knowledge_query_prompt | chatllm | output_parser

        # run_server
        run_server(
            [query_chain, chat_chain, cypher_chain, history_retrieve_chain, history_store_chain, custom_obliged_query_chain, custom_referential_query_chain], 
            ["/query", "/chat", "/cypher", "/get_history", "/save_history", "/custom_obliged_query", "/custom_referential_query"]
        )

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())