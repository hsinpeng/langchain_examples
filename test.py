import sys
import json
import logging
import uuid
from datetime import datetime, timedelta
import faiss
from test_classes import Actor, Action, LineList, LineListOutputParser
from test_utilities import _get_datetime, _pretty_print_docs

import openai
import langchain
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.example_selector import (
    SemanticSimilarityExampleSelector,
    MaxMarginalRelevanceExampleSelector,
    LengthBasedExampleSelector,
)
from langchain.prompts.example_selector.ngram_overlap import NGramOverlapExampleSelector
from langchain.vectorstores import Chroma, FAISS
from langchain.storage import InMemoryStore
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
    ChatMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    load_prompt,
)
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.chains import LLMChain
from langchain.cache import InMemoryCache
from langchain.output_parsers import (
    PydanticOutputParser,
    OutputFixingParser,
    RetryWithErrorOutputParser,
    StructuredOutputParser, 
    ResponseSchema,
    XMLOutputParser,
)
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationChain
from langchain.retrievers import (
    ContextualCompressionRetriever,
    BM25Retriever,
    EnsembleRetriever,
    ParentDocumentRetriever,
    TimeWeightedVectorStoreRetriever,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter, DocumentCompressorPipeline
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.docstore import InMemoryDocstore
from langchain.utils import mock_now
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains.router.embedding_router import EmbeddingRouterChain

###### User Define Parameters ######
test_option = 0
test_phrase = "Who is the director of dark knight?"
####################################
        
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

def main():
    try:
        print(f"TestOption #{test_option}")

        match test_option:
            case 0:
                ##### Just Test #####
                print("Hello LangChain!")
                print(azure_apikey, azure_apibase, azure_apitype, azure_apiversion, azure_gptx_deployment, azure_embd_deployment)
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase, temperature=0.9)
                #prompt = PromptTemplate(
                #    input_variables=["product"],
                #    template="What is a good name for a company that makes {product}?",
                #)
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a helpful assistant to give five examples for the following question."), 
                        ("human", "What is a good name for a company that makes {product}?"),
                    ]
                )
                chain = LLMChain(llm=chatllm, prompt=prompt)
                resp = chain.run(product="Cycling")
                print(resp)
            case 1|2:
                ##### Azure Settings #####
                openai.api_key = azure_apikey
                openai.api_base = azure_apibase
                openai.api_type = azure_apitype
                openai.api_version = azure_apiversion        
                if test_option == 1:
                    ##### Test #1: Azure Completion #####
                    response = openai.Completion.create(engine=azure_gptx_deployment, prompt=test_phrase, max_tokens=10) #, temperature=5.0, frequency_penalty=0, presence_penalty=0, stop=None)
                    text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
                    print(f"INPUT: {test_phrase}")
                    print("OUTPUT: ", text)        
                elif test_option == 2:
                    ##### Test #2: Azure ChatCompletion #####
                    chatmsg = [
                            #{"role": "system", "content": "You are a helpful assistant."},
                            #{"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
                            #{"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
                            {"role": "user", "content": test_phrase}
                    ]
                    response = openai.ChatCompletion.create(engine=azure_gptx_deployment, messages=chatmsg, temperature=0.5, frequency_penalty=0, presence_penalty=0)
                    print(f"INPUT: {test_phrase}")
                    print("OUTPUT: ", response['choices'][0]['message']['content'])
                else:
                    print("Error: Mission Impossible!")
            case 3:
                ##### Test #3: LangChain LLMs #####
                llm = AzureOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase)
                print(f"INPUT: {test_phrase}")
                print("OUTPUT: ", llm.predict(test_phrase))
            case 4:
                ##### Test #4: LangChain ChatModels #####
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase)
                msg = HumanMessage(content=test_phrase)
                #print(chatllm(messages=[msg]))
                print(f"INPUT: {test_phrase}")
                print("OUTPUT: ", chatllm.predict_messages(messages=[msg]).content)
            case 5:
                ##### Test #5: Azure Embedding Models #####
                with open('./data/examples01.json', 'r', encoding='utf-8') as f:
                    examples_01 = json.load(f)
                f.close()
                #print(examples_01)
                
                embdllm = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)
                print(f"INPUT: {examples_01[0]}")
                print("OUTPUT: ", embdllm.embed_documents(examples_01[0]))
            case 6:
                ##### Test #6: Few-shot prompt templates with Azure Embedding #####
                with open('./data/examples01.json', 'r', encoding='utf-8') as f:
                    examples_01 = json.load(f)
                f.close()
                #print(examples_01)

                example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")
                #print(example_prompt.format(**examples[0]))

                example_selector = SemanticSimilarityExampleSelector.from_examples(
                    # This is the list of examples available to select from.
                    examples_01,
                    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
                    OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase),
                    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
                    Chroma,
                    # This is the number of examples to produce.
                    k=1
                )

                # Select the most similar example to the input.
                question = test_phrase
                #selected_examples = example_selector.select_examples({"question": question})
                #print(f"Examples most similar to the input: {question}")
                #for example in selected_examples:
                #    print("\n")
                #    for k, v in example.items():
                #        print(f"{k}: {v}")
                    
                prompt = FewShotPromptTemplate(
                    example_selector=example_selector, 
                    example_prompt=example_prompt, 
                    suffix="Question: {input}", # The real input format 
                    input_variables=["input"]
                )
                print(f"INPUT: {question}")
                print(prompt.format(input=question))
            case 7:
                ##### Test #7: Few-shot examples for chat models #####
                with open('./data/examples02.json', 'r', encoding='utf-8') as f:
                    examples_02 = json.load(f)
                f.close()

                to_vectorize = [" ".join(example.values()) for example in examples_02]
                #print(to_vectorize)
                embeddings = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)
                vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples_02)
                example_selector = SemanticSimilarityExampleSelector(
                    vectorstore=vectorstore,
                    k=2,
                )
                # The prompt template will load examples by passing the input do the `select_examples` method
                #similar_examples = example_selector.select_examples({"input": "horse"})
                #print(similar_examples)

                # Define the few-shot prompt.
                few_shot_prompt = FewShotChatMessagePromptTemplate(
                    # The input variables select the values to pass to the example_selector
                    input_variables=["input"],
                    example_selector=example_selector,
                    # Define how each example will be formatted.
                    # In this case, each example will become 2 messages:
                    # 1 human, and 1 AI
                    example_prompt=ChatPromptTemplate.from_messages(
                        [("human", "{input}"), ("ai", "{output}")]
                    ),
                )
                #print(few_shot_prompt.format(input="What's 3+3?"))
                
                final_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a wondrous wizard of math."),
                        few_shot_prompt,
                        ("human", "{input}"),
                    ]
                )

                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase)
                msg = final_prompt.format_messages(input=test_phrase)
                print("INPIUT: ", msg)
                print("OUTPUT: ", chatllm.predict_messages(messages=msg).content)
            case 8:
                ##### Test #8: Prompt templates #####
                # Type 0 Full prompt template
                # BJ4

                # Type 1 Partial prompt template with strings
                # 1-1
                prompt = PromptTemplate(template="{foo}{bar}", input_variables=["foo", "bar"])
                partial_prompt = prompt.partial(foo="foo");
                print("(1-1) Partial prompt template with strings 1 OUTPUT: ", partial_prompt.format(bar="baz"))
                # 1-2
                prompt = PromptTemplate(template="{foo}{bar}", input_variables=["bar"], partial_variables={"foo": "foo"})
                print("(1-2) Partial prompt template with strings 2 OUTPUT: ", prompt.format(bar="baz"))

                # Type 2 Partial prompt template with functions
                prompt = PromptTemplate(
                    template="Tell me a {adjective} joke about the day {date}", 
                    input_variables=["adjective"],
                    partial_variables={"date": _get_datetime}
                )
                print("(2) Partial prompt template with functions OUTPUT: ", prompt.format(adjective="funny"))

                # Type 3 Pipeline prompts
                full_template = """{introduction}

                {example}

                {start}"""
                full_prompt = PromptTemplate.from_template(full_template)

                introduction_template = """You are impersonating {person}."""
                introduction_prompt = PromptTemplate.from_template(introduction_template)

                example_template = """Here's an example of an interaction: 
                Q: {example_q}
                A: {example_a}"""
                example_prompt = PromptTemplate.from_template(example_template)

                start_template = """Now, do this for real!
                Q: {input}
                A:"""
                start_prompt = PromptTemplate.from_template(start_template)

                input_prompts = [
                    ("introduction", introduction_prompt),
                    ("example", example_prompt),
                    ("start", start_prompt)
                ]
                pipeline_prompt = PipelinePromptTemplate(final_prompt=full_prompt, pipeline_prompts=input_prompts)
                #print(f"{pipeline_prompt.input_variables}")
                msg = pipeline_prompt.format(
                    person="Elon Musk",
                    example_q="What's your favorite car?",
                    example_a="Tesla",
                    input="What's your favorite social media site?"
                )
                print(f"(3) Pipeline prompts OUTPUT: {msg}")
            case 9:
                ##### Test #9: Types of MessagePromptTemplate #####
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase)

                # Type 0 The most commonly used are AIMessagePromptTemplate, SystemMessagePromptTemplate and HumanMessagePromptTemplate, 
                # which create an AI message, system message and human message respectively.
                chat_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."), 
                        ("human", "{text}"),
                    ]
                )
                print("(0) Full ChatPromptTemplate OUTPUT: ", chat_prompt.format_messages(input_language="English", output_language="Chinese", text=test_phrase))

                # Type 1 ChatMessagePromptTemplate: Allows user to specify the role name
                prompt = "May the {subject} be with you"
                chat_message_prompt = ChatMessagePromptTemplate.from_template(role="user", template=prompt) # "Jedi" is ilegal to LLM
                print("(1-1) ChatMessagePromptTemplate OUTPUT: ", chat_message_prompt.format(subject="force"))
                msg = chat_message_prompt.format_messages(subject="force")
                print("(1-2) ChatModel OUTPUT: ", chatllm.predict_messages(messages=msg).content)

                # Type 2 MessagesPlaceholder: Gives you full control of what messages to be rendered during formatting.
                human_prompt = "Summarize our conversation so far in {word_count} words."
                human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

                chat_prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="conversation"), human_message_template])
                human_message = HumanMessage(content="What is the best way to learn programming?")
                ai_message = AIMessage(content="""\
                1. Choose a programming language: Decide on a programming language that you want to learn.

                2. Start with the basics: Familiarize yourself with the basic programming concepts such as variables, data types and control structures.

                3. Practice, practice, practice: The best way to learn programming is through hands-on experience\
                """)
                msg = chat_prompt.format_prompt(conversation=[human_message, ai_message], word_count="10").to_messages()
                print(f"(2-1) MessagesPlaceholder OUTPUT: {msg}")                
                print("(2-2) ChatModel OUTPUT: ", chatllm.predict_messages(messages=msg).content)
            case 10:
                ##### Test #10: Serialization, store prompts as files and load them on-demand #####
                # PromptTemplate Type 1 Loading from YAML
                prompt = load_prompt("./data/simple_prompt.yaml")
                print("\nPromptTemplate(1) From YAML OUTPUT: ", prompt.format(adjective="funny", content="chickens"))

                # PromptTemplate Type 2 Loading from JSON
                prompt = load_prompt("./data/simple_prompt.json")
                print("\nPromptTemplate(2) From JSON OUTPUT: ", prompt.format(adjective="funny", content="chickens"))

                # PromptTemplate Type 3 Loading from JSON and it loads template from a file
                prompt = load_prompt("./data/simple_prompt_with_template_file.json")
                print("\nPromptTemplate(3) From template file OUTPUT: ", prompt.format(adjective="funny", content="chickens"))

                # FewShotPromptTemplate Type 1 Loading from YAML
                prompt = load_prompt("./data/few_shot_prompt.yaml") # loads JSON exmaples
                print("\nFewShotPromptTemplate(1-1) From YAML (with JSON exmaples) OUTPUT: ", prompt.format(adjective="funny"))
                prompt = load_prompt("./data/few_shot_prompt_yaml_examples.yaml") # loads YAML exmaples
                print("\nFewShotPromptTemplate(1-2) From YAML (with YAML exmaples) OUTPUT: ", prompt.format(adjective="funny"))

                # FewShotPromptTemplate Type 2 Loading from JSON
                prompt = load_prompt("./data/few_shot_prompt.json")
                print("\nFewShotPromptTemplate(2-1) From JSON (with JSON exmaples) OUTPUT: ", prompt.format(adjective="funny"))
                prompt = load_prompt("./data/few_shot_prompt_examples_in.json")
                print("\nFewShotPromptTemplate(2-2) From JSON (infile exmaples) OUTPUT: ", prompt.format(adjective="funny"))
                prompt = load_prompt("./data/few_shot_prompt_example_prompt.json")
                print("\nFewShotPromptTemplate(2-3) From JSON (with JSON exmaples and prompt) OUTPUT: ", prompt.format(adjective="funny"))

                # PromptTemplate with OutputParser
                prompt = load_prompt("./data/prompt_with_output_parser.json")
                msg = prompt.output_parser.parse(
                    "George Washington was born in 1732 and died in 1799.\nScore: 1/2"
                )
                print("\nPromptTemplate with OutputParser OUTPUT: ", msg)
            case 11:
                ##### Test #11: Prompt pipelining for composing different parts of prompts together #####
                # Type 1 String prompt pipelining
                prompt = (
                    PromptTemplate.from_template("Tell me a joke about {topic}")
                    + ", make it funny"
                    + "\n\nand in {language}"
                )
                print("\n(1) String prompt pipelining OUTPUT: ", prompt.format(topic="sports", language="spanish"))

                # Type 2 Chat prompt pipelining
                prompt = SystemMessage(content="You are a nice pirate")
                new_prompt = (
                    prompt
                    + HumanMessage(content="hi")
                    + AIMessage(content="what?")
                    + "{input}"
                )
                print("\n(2-1) Chat prompt pipelining OUTPUT: ", new_prompt.format_messages(input="i said hi"))
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase)
                chain = LLMChain(llm=chatllm, prompt=new_prompt)
                print("\n(2-2) Chat model OUTPUT: ", chain.run("i said hi"))
            case 12:
                ##### Test #12: Example selector #####
                with open('./data/examples03.json', 'r', encoding='utf-8') as f:
                    examples_03 = json.load(f) # Examples of a pretend task of creating antonyms.
                f.close()

                # Type 1 Select by length
                example_prompt = PromptTemplate(
                    input_variables=["input", "output"],
                    template="Input: {input}\nOutput: {output}",
                )
                example_selector = LengthBasedExampleSelector(
                    # The examples it has available to choose from.
                    examples=examples_03, 
                    # The PromptTemplate being used to format the examples.
                    example_prompt=example_prompt, 
                    # The maximum length that the formatted examples should be.
                    # Length is measured by the get_text_length function below.
                    max_length=25,
                    # The function used to get the length of a string, which is used
                    # to determine which examples to include. It is commented out because
                    # it is provided as a default value if none is specified.
                    # get_text_length: Callable[[str], int] = lambda x: len(re.split("\n| ", x))
                )
                dynamic_prompt = FewShotPromptTemplate(
                    # We provide an ExampleSelector instead of examples.
                    example_selector=example_selector,
                    example_prompt=example_prompt,
                    prefix="Give the antonym of every input",
                    suffix="Input: {adjective}\nOutput:", 
                    input_variables=["adjective"],
                )
                # An example with small input, so it selects all examples.
                print("\n(1-2) Select by length OUTPUT(small input): ", dynamic_prompt.format(adjective="big"))
                # An example with long input, so it selects only one example.
                long_string = "big and huge and massive and large and gigantic and tall and much much much much much bigger than everything else"
                print("\n(1-2) Select by length OUTPUT(long input): ", dynamic_prompt.format(adjective=long_string))
                # You can add an example to an example selector as well.
                new_example = {"input": "big", "output": "small"}
                dynamic_prompt.example_selector.add_example(new_example)
                print("\n(1-3) Select by length OUTPUT(add examples): ", dynamic_prompt.format(adjective="enthusiastic"))

                # Type 2 Select by maximal marginal relevance (MMR)
                example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
                    # The list of examples available to select from.
                    examples_03,
                    # The embedding class used to produce embeddings which are used to measure semantic similarity.
                    OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase),
                    # The VectorStore class that is used to store the embeddings and do a similarity search over.
                    FAISS,
                    # The number of examples to produce.
                    k=2,
                )
                mmr_prompt = FewShotPromptTemplate(
                    # We provide an ExampleSelector instead of examples.
                    example_selector=example_selector,
                    example_prompt=example_prompt,
                    prefix="Give the antonym of every input",
                    suffix="Input: {adjective}\nOutput:",
                    input_variables=["adjective"],
                )
                # Input is a feeling, so should select the happy/sad example as the first one
                print("\n(2-1) Select by MMR OUTPUT: ", mmr_prompt.format(adjective="worried"))
                # Let's compare this to what we would just get if we went solely off of similarity,
                # by using SemanticSimilarityExampleSelector instead of MaxMarginalRelevanceExampleSelector.
                example_selector = SemanticSimilarityExampleSelector.from_examples(
                    # The list of examples available to select from.
                    examples_03,
                    # The embedding class used to produce embeddings which are used to measure semantic similarity.
                    OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase),
                    # The VectorStore class that is used to store the embeddings and do a similarity search over.
                    FAISS,
                    # The number of examples to produce.
                    k=2,
                )
                similar_prompt = FewShotPromptTemplate(
                    # We provide an ExampleSelector instead of examples.
                    example_selector=example_selector,
                    example_prompt=example_prompt,
                    prefix="Give the antonym of every input",
                    suffix="Input: {adjective}\nOutput:",
                    input_variables=["adjective"],
                )
                print("\n(2-2) Select by similarity OUTPUT: ", similar_prompt.format(adjective="worried"))

                # Type 3 Select by n-gram overlap
                with open('./data/examples04.json', 'r', encoding='utf-8') as f:
                    examples_04 = json.load(f) # Examples of a fictional translation task.
                f.close()

                example_selector = NGramOverlapExampleSelector(
                    # The examples it has available to choose from.
                    examples=examples_04,
                    # The PromptTemplate being used to format the examples.
                    example_prompt=example_prompt,
                    # The threshold, at which selector stops.
                    # It is set to -1.0 by default.
                    threshold=-1.0,
                    # For negative threshold:
                    # Selector sorts examples by ngram overlap score, and excludes none.
                    # For threshold greater than 1.0:
                    # Selector excludes all examples, and returns an empty list.
                    # For threshold equal to 0.0:
                    # Selector sorts examples by ngram overlap score,
                    # and excludes those with no ngram overlap with input.
                )
                dynamic_prompt = FewShotPromptTemplate(
                    # We provide an ExampleSelector instead of examples.
                    example_selector=example_selector,
                    example_prompt=example_prompt,
                    prefix="Give the Spanish translation of every input",
                    suffix="Input: {sentence}\nOutput:",
                    input_variables=["sentence"],
                )
                # An example input with large ngram overlap with "Spot can run."
                # and no overlap with "My dog barks."
                print("\n(3-1) Select by n-gram overlap (threshold=-1.0) OUTPUT: ", dynamic_prompt.format(sentence="Spot can run fast."))

                # You can add examples to NGramOverlapExampleSelector as well.
                new_example = {"input": "Spot plays fetch.", "output": "Spot juega a buscar."}
                example_selector.add_example(new_example)
                print("\n(3-2) Select by n-gram overlap (add example) OUTPUT: ", dynamic_prompt.format(sentence="Spot can run fast."))

                # You can set a threshold at which examples are excluded.
                # For example, setting threshold equal to 0.0
                # excludes examples with no ngram overlaps with input.
                # Since "My dog barks." has no ngram overlaps with "Spot can run fast."
                # it is excluded.
                example_selector.threshold = 0.0
                print("\n(3-3) Select by n-gram overlap (threshold=0.0) OUTPUT: ", dynamic_prompt.format(sentence="Spot can run fast."))
                
                # Setting small nonzero threshold
                example_selector.threshold = 0.09
                print("\n(3-4) Select by n-gram overlap (threshold=0.09) OUTPUT: ", dynamic_prompt.format(sentence="Spot can play fetch."))

                # Setting threshold greater than 1.0
                example_selector.threshold = 1.0 + 1e-9
                print("\n(3-5) Select by n-gram overlap (threshold>1.0) OUTPUT: ", dynamic_prompt.format(sentence="Spot can play fetch."))
            case 13:
                ##### Test #13: Language models #####
                # Type 1 LLMs (Cannot work with GPT-4)
                # Async API, Custom LLM, Fake LLM, Human input LLM, Caching, Serialization, Streaming, Tracking token usage
                print('\n(1) LLMs OUTPUT: Cannot work with GPT-4!')

                # Type 2 Chat Models: "chat messages" are the inputs and outputs.
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase) #, cache=False)
                messages = [
                    SystemMessage(content="You are a helpful assistant that translates English to Chinese."),
                    HumanMessage(content=test_phrase)
                ]
                print("\n(2-1) Chat_model OUTPUT: ", chatllm(messages).content)
                
                # Token usage
                result = chatllm.generate([messages])
                print("\n(2-2) Chat_model (generate) OUTPUT: ", result.llm_output['token_usage'])
                
                # Caching
                langchain.llm_cache = InMemoryCache()
                #langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
                print("\n(2-3) Chat_model (Caching) \nOUTPUT 1 (First): ", chatllm.predict(test_phrase))
                print("\nOUTPUT 2 (Cached): ", chatllm.predict(test_phrase))
                
                # Disable cache
                #chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase, cache=False)
                
                # Human input chat model
                #tools = load_tools(["wikipedia"])
                #llm = HumanInputChatModel()
                #agent = initialize_agent(
                #    tools, llm, agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True
                #)
                #print(agent("What is Bocchi the Rock?"))

                # LLM Chain
                chat_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."), 
                        ("human", "{text}"),
                    ]
                )
                chain = LLMChain(llm=chatllm, prompt=chat_prompt)
                print("\n(3) LLM Chain\nOUTPUT: ", chain.run(input_language="English", output_language="Chinese", text=test_phrase))
            case 14:
                ##### Test #14: Output parsers #####
                # List parser, Datetime parser, Enum parser, Auto-fixing parser, Pydantic (JSON) parser, Retry parser, Structured output parser, XML parser
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase, temperature=0) #, cache=False)
                # Type 1  Auto-fixing parser
                #actor_query = "Generate the filmography for a random actor."
                parser = PydanticOutputParser(pydantic_object=Actor)
                misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"
                #print("\n(1-1) Auto-fixing parser OUTPUT: ", parser.parse(misformatted))
                new_parser = OutputFixingParser.from_llm(parser=parser, llm=chatllm) # Takes another output parser and an LLM with which to try to correct any formatting mistakes.
                print("\n(1) Auto-fixing parser OUTPUT: ", new_parser.parse(misformatted))

                # Type 2 Retry parser
                parser = PydanticOutputParser(pydantic_object=Action)
                prompt = PromptTemplate(
                    template="Answer the user query.\n{format_instructions}\n{query}\n",
                    input_variables=["query"],
                    partial_variables={"format_instructions": parser.get_format_instructions()},
                )
                prompt_value = prompt.format_prompt(query="who is leo di caprios gf?")
                bad_response = '{"action": "search"}'
                fix_parser = OutputFixingParser.from_llm(parser=parser, llm=chatllm)
                print("\n(2-1) Fix parser OUTPUT: ", fix_parser.parse(bad_response))
                retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=chatllm)
                print("\n(2-2) Retry parser OUTPUT: ", retry_parser.parse_with_prompt(bad_response, prompt_value))

                # Type 3 Structured output parser
                response_schemas = [
                    ResponseSchema(name="answer", description="answer to the user's question"),
                    ResponseSchema(name="source", description="source used to answer the user's question, should be a website.")
                ]
                output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
                format_instructions = output_parser.get_format_instructions()
                #print(format_instructions)
                prompt = PromptTemplate(
                    template="answer the users question as best as possible.\n{format_instructions}\n{question}",
                    input_variables=["question"],
                    partial_variables={"format_instructions": format_instructions}
                )
                _input = prompt.format_prompt(question="what's the capital of france?")
                output = chatllm.predict(_input.to_string())
                print("\n(3-1) Structured output parser OUTPUT: ", output_parser.parse(output))
                prompt = ChatPromptTemplate(
                    messages=[
                        HumanMessagePromptTemplate.from_template("answer the users question as best as possible.\n{format_instructions}\n{question}")  
                    ],
                    input_variables=["question"],
                    partial_variables={"format_instructions": format_instructions}
                )
                _input = prompt.format_prompt(question="what's the capital of france?")
                output = chatllm(_input.to_messages())
                print("\n(3-2) Structured output parser OUTPUT: ", output_parser.parse(output.content))

                # Type 4 XML parser
                actor_query = "Generate the shortened filmography for Tom Hanks."
                messages = [
                    HumanMessage(content=f'{actor_query}\n Please enclose the movies in <movie></movie> tags')  
                ]
                output = chatllm(messages).content
                #print(output)
                parser = XMLOutputParser()
                prompt = PromptTemplate(
                    template="""
                    
                    Human:
                    {query}
                    {format_instructions}
                    Assistant:""",
                    input_variables=["query"],
                    partial_variables={"format_instructions": parser.get_format_instructions()},
                )

                chain = prompt | chatllm | parser
                output = chain.invoke({"query": actor_query})
                print("\n(4) XML parser OUTPUT: ", output)
            case 15:
                ##### Test #15: Retrieval (Document loaders, Document transformers, Text embedding models, Vector stores, Retrievers) #####
                # Type 1 Document loaders: Text, CSV, File Directory, HTML, JSON, Markdown, PDF
                
                # Type 2 Document transformers: Text splitters, Split by character, Split code, MarkdownHeaderTextSplitter, Recursively split by character, Split by tokens
                # Note: When models must access relevant information in the middle of long contexts, they tend to ignore the provided documents. To avoid this issue you can re-order documents after retrieval to avoid performance degradation.
                
                # Type 3 Text embedding models
                embeddings_model = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)             
                embeddings = embeddings_model.embed_documents(
                    [
                        "Hi there!",
                        "Oh, hello!",
                        "What's your name?",
                        "My friends call me World",
                        "Hello World!"
                    ]
                )
                print("\n(1-1) Text embedding OUTPUT: ", len(embeddings), len(embeddings[0]))
                embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
                print("\n(1-2) Embedding query OUTPUT: ", embedded_query[:5])

                # Type 4 Vector stores: Chroma, FAISS, Lance
                # Load the document, split it into chunks, embed each chunk and load it into the vector store.
                raw_documents = TextLoader("./data/state_of_the_union.txt", encoding="utf-8").load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                documents = text_splitter.split_documents(raw_documents)
                db = Chroma.from_documents(documents, embeddings_model)
                query = "What did the president say about Ketanji Brown Jackson"
                docs = db.similarity_search(query)
                print("\n(2-1) DB similarity search OUTPUT[0]:\n", docs[0].page_content)
                print(f"Length of docs: {len(docs)}")
                embedding_vector = embeddings_model.embed_query(query)
                docs = db.similarity_search_by_vector(embedding_vector)
                print("\n(2-2) DB similarity search by vector OUTPUT[0]:\n", docs[0].page_content)
                print(f"Length of docs: {len(docs)}")

                # Type 5 Retrievers: A retriever is an interface that returns documents given an unstructured query. It is more general than a vector store.
                retriever = db.as_retriever() # Expose this index in a retriever interface.
                docs = retriever.get_relevant_documents(query)
                print("\n(3-1) Similarity retriever OUTPUT[0]:\n", docs[0].page_content)
                print(f"Length of docs: {len(docs)}")

                # Maximum marginal relevance retrieval
                retriever = db.as_retriever(search_type="mmr")
                docs = retriever.get_relevant_documents(query)
                print("\n(3-2) MMR retriever OUTPUT[0]:\n", docs[0].page_content)
                print(f"Length of docs: {len(docs)}")

                # Similarity score threshold retrieval
                retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
                docs = retriever.get_relevant_documents(query)
                print("\n(3-3) Similarity score threshold retriever OUTPUT[0]:\n", docs[0].page_content)
                print(f"Length of docs: {len(docs)}")

                # Specifying top k
                retriever = db.as_retriever(search_kwargs={"k": 1})
                docs = retriever.get_relevant_documents(query)
                print("\n(3-4) Specifying top k OUTPUT[0]:\n", docs[0].page_content)
                print(f"Length of docs: {len(docs)}")

            case 16:
                ##### Test #16: Question Answering Retriever #####
                embeddings_model = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)             
                loader = TextLoader('./data/state_of_the_union.txt', encoding='utf8')
                documents = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(documents) # Split the documents into chunks
                db = Chroma.from_documents(texts, embeddings_model) # Create the vector store to use as the index.                
                retriever = db.as_retriever() # Expose this index in a retriever interface.                
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase)
                qa = RetrievalQA.from_chain_type(llm=chatllm, chain_type="stuff", retriever=retriever)
                query = "What did the president say about Ketanji Brown Jackson"
                print("\nRetrievalQA OUTPUT: ", qa.run(query))
            case 17|18:
                ##### Test #17: MultiQueryRetriever (Simple usage) #####
                # The automatic prompt tuning by using an LLM to generate multiple queries from different perspectives for a given user input query.
                # It might be able to overcome some of the limitations of the distance-based retrieval and get a richer set of results.
                
                ##### Test #18: MultiQueryRetriever (Supplying customized prompt) ##### 
                # Supply a prompt along with an output parser to split the results into a list of queries.
                
                # Load blog post
                loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
                data = loader.load() 
                # Split
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
                splits = text_splitter.split_documents(data)
                # VectorDB
                embeddings_model = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)             
                vectordb = Chroma.from_documents(documents=splits, embedding=embeddings_model)
                # Chat model
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase, temperature=0)
                question = "What are the approaches to Task Decomposition?"
                # Set logging for the queries
                logging.basicConfig()
                logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
                if test_option == 17:
                    # Simple usage
                    retriever_from_llm = MultiQueryRetriever.from_llm(
                        retriever=vectordb.as_retriever(), llm=chatllm
                    )
                    print("\nMultiQueryRetriever (Simple usage) OUTPUT")
                    unique_docs = retriever_from_llm.get_relevant_documents(query=question)
                    print("\nLength of unique_docs: ", len(unique_docs))
                    i_dx = 0
                    for doc in unique_docs: 
                        print(f"\nunique_docs[{i_dx}]: ", doc)
                        i_dx = i_dx + 1
                elif test_option == 18:
                    # Supplying customized prompt
                    output_parser = LineListOutputParser()
                    QUERY_PROMPT = PromptTemplate(
                        input_variables=["question"],
                        template="""You are an AI language model assistant. Your task is to generate five 
                        different versions of the given user question to retrieve relevant documents from a vector 
                        database. By generating multiple perspectives on the user question, your goal is to help
                        the user overcome some of the limitations of the distance-based similarity search. 
                        Provide these alternative questions seperated by newlines.
                        Original question: {question}""",
                    )
                    # Chain
                    llm_chain = LLMChain(llm=chatllm, prompt=QUERY_PROMPT, output_parser=output_parser)
                    print("\nMultiQueryRetriever (Supplying customized prompt) OUTPUT")
                    # Run
                    retriever = MultiQueryRetriever(
                        retriever=vectordb.as_retriever(), llm_chain=llm_chain, parser_key="lines"
                    )  # "lines" is the key (attribute name) of the parsed output
                    # Results
                    unique_docs = retriever.get_relevant_documents(query=question)
                    print("\nLength of unique_docs: ", len(unique_docs))
                    i_dx = 0
                    for doc in unique_docs: 
                        print(f"\nunique_docs[{i_dx}]: ", doc)
                        i_dx = i_dx + 1
                else:
                    print("Error: Mission Impossible!")
            case 19:
                ##### Test #19: Contextual compression #####
                # Compressing retrieved documents by using the context of the given query. (Needs a base retriever and a document compressor)
                embeddings_model = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)             
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase, temperature=0)
                question = "What did the president say about Ketanji Jackson Brown"
                # Retrieve documents
                documents = TextLoader('./data/state_of_the_union.txt', encoding='utf8').load()
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(documents)
                retriever = FAISS.from_documents(texts, embeddings_model).as_retriever() # Base Retriever
                
                #  No Compression
                docs = retriever.get_relevant_documents(question)
                print("\nContextual compression OUTPUT")
                print("\n(1) No Compression:")
                _pretty_print_docs(docs)
                
                # Built-in compressor: LLMChainExtractor
                # Returned documents and extract from each only the content that is relevant to the query
                Extractor_compressor = LLMChainExtractor.from_llm(chatllm) # Document Compressor
                compression_retriever1 = ContextualCompressionRetriever(base_compressor=Extractor_compressor, base_retriever=retriever)
                compressed_docs = compression_retriever1.get_relevant_documents(question)
                print("\n(2) Compression with LLMChainExtractor:")
                _pretty_print_docs(compressed_docs)
                
                # Built-in compressor: LLMChainFilter
                # Filter out and which ones to return, without manipulating the document content
                Filter_compressor = LLMChainFilter.from_llm(chatllm)
                compression_retriever2 = ContextualCompressionRetriever(base_compressor=Filter_compressor, base_retriever=retriever)
                compressed_docs = compression_retriever2.get_relevant_documents(question)
                print("\n(3) Compression with LLMChainFilter:")
                _pretty_print_docs(compressed_docs)
            
                ##### Stringing compressors and document transformers together (DocumentCompressorPipeline) #####
                splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
                redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings_model)
                relevant_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.76)
                pipeline_compressor = DocumentCompressorPipeline(
                    transformers=[splitter, redundant_filter, relevant_filter]
                )
                compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)
                compressed_docs = compression_retriever.get_relevant_documents(question)
                print("\n(4) Compression with DocumentCompressorPipeline:")
                _pretty_print_docs(compressed_docs)
            case 20:
                ##### Test #20: Ensemble Retriever #####
                # The EnsembleRetriever takes a list of retrievers as input and ensemble the results of their get_relevant_documents() methods,
                # then rerank the results based on the Reciprocal Rank Fusion algorithm.
                # The most common pattern is to combine a sparse retriever (like BM25) with a dense retriever (like embedding similarity), 
                # because their strengths are complementary.
                doc_list = [
                    "I like apples",
                    "I like oranges",
                    "Apples and oranges are fruits",
                ]

                # initialize the bm25 retriever and faiss retriever
                bm25_retriever = BM25Retriever.from_texts(doc_list)
                bm25_retriever.k = 2
                embeddings_model = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)             
                faiss_vectorstore = FAISS.from_texts(doc_list, embeddings_model)
                faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

                # initialize the ensemble retriever
                ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
                docs = ensemble_retriever.get_relevant_documents("apples")
                print("Ensemble Retriever OUTPUT: ", docs)
            case 21|22|23:
                ##### Test #21-23: MultiVector Retriever (Store multiple vectors per document) #####
                # The methods to create multiple vectors per document include:
                # (1) Smaller chunks: split a document into smaller chunks, and embed those (this is ParentDocumentRetriever).
                # (2) Summary: create a summary for each document, embed that along with (or instead of) the document.
                # (3) Hypothetical questions: create hypothetical questions that each document would be appropriate to answer, embed those along with (or instead of) the document.
                embeddings_model = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)             
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase, temperature=0, max_retries=0, model="gpt-4")

                loaders = [
                    TextLoader('./data/paul_graham_essay.txt', encoding='utf8'),
                    TextLoader('./data/state_of_the_union.txt', encoding='utf8'),
                ]
                docs = []
                for l in loaders:
                    #print(f"Length of docs={len(docs)}")
                    docs.extend(l.load())
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
                print(f"Length of docs={len(docs)}")
                docs = text_splitter.split_documents(docs)
                print(f"Length of docs={len(docs)}")
                
                if test_option == 21: 
                    # (1) Smaller chunks
                    # The vectorstore to use to index the child chunks
                    vectorstore = Chroma(
                        collection_name="full_documents",
                        embedding_function=embeddings_model
                    )
                    # The storage layer for the parent documents
                    store = InMemoryStore()
                    id_key = "doc_id"
                    # The retriever (empty to start)
                    retriever = MultiVectorRetriever(
                        vectorstore=vectorstore, 
                        docstore=store, 
                        id_key=id_key,
                    )                
                    doc_ids = [str(uuid.uuid4()) for _ in docs]
                    # The splitter to use to create smaller chunks
                    child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
                    sub_docs = []
                    for i, doc in enumerate(docs):
                        _id = doc_ids[i]
                        #print(f"Doc[{i}] ID: {_id}")
                        _sub_docs = child_text_splitter.split_documents([doc])
                        for _doc in _sub_docs:
                            _doc.metadata[id_key] = _id
                        sub_docs.extend(_sub_docs)
                    # Vectorstore abd docstore
                    retriever.vectorstore.add_documents(sub_docs)
                    retriever.docstore.mset(list(zip(doc_ids, docs)))

                    print("Smaller chunks OUTPUT: ")
                    # Vectorstore alone retrieves the small chunks
                    print(len(retriever.vectorstore.similarity_search("justice breyer")[0].page_content))
                    # Retriever returns larger chunks
                    print(len(retriever.get_relevant_documents("justice breyer")[0].page_content))
                elif test_option == 22:
                    # Summary: Oftentimes a summary may be able to distill more accurately what a chunk is about, leading to better retrieval.
                    chain = (
                        {"doc": lambda x: x.page_content}
                        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
                        | chatllm #ChatOpenAI(max_retries=0)
                        | StrOutputParser()
                    )
                    summaries = chain.batch(docs, {"max_concurrency": 5})
                    # The vectorstore to use to index the child chunks
                    vectorstore = Chroma(
                        collection_name="summaries",
                        embedding_function=embeddings_model
                    )
                    # The storage layer for the parent documents
                    store = InMemoryStore()
                    id_key = "doc_id"
                    # The retriever (empty to start)
                    retriever = MultiVectorRetriever(
                        vectorstore=vectorstore, 
                        docstore=store, 
                        id_key=id_key,
                    )
                    doc_ids = [str(uuid.uuid4()) for _ in docs]
                    summary_docs = [Document(page_content=s,metadata={id_key: doc_ids[i]}) for i, s in enumerate(summaries)]
                    retriever.vectorstore.add_documents(summary_docs)
                    retriever.docstore.mset(list(zip(doc_ids, docs)))
                    # Vectorstore abd docstore
                    retriever.vectorstore.add_documents(summary_docs)
                    retriever.docstore.mset(list(zip(doc_ids, docs)))

                    print("Summary OUTPUT: ")
                    # Vectorstore alone retrieves the small chunks
                    sub_docs = vectorstore.similarity_search("justice breyer")
                    print(sub_docs[0].page_content)
                    print(len(sub_docs[0].page_content))
                    # Retriever returns larger chunks
                    retrieved_docs = retriever.get_relevant_documents("justice breyer")
                    print(len(retrieved_docs[0].page_content))
                elif test_option == 23:
                    # Hypothetical Queries
                    # An LLM can also be used to generate a list of hypothetical questions that could be asked of a particular document. These questions can then be embedded
                    functions = [
                        {
                            "name": "hypothetical_questions",
                            "description": "Generate hypothetical questions",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "questions": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        },
                                    },
                                },
                                "required": ["questions"]
                            }
                        }
                    ]
                    chain = (
                        {"doc": lambda x: x.page_content}
                        # Only asking for 3 hypothetical questions, but this could be adjusted
                        | ChatPromptTemplate.from_template("Generate a list of 3 hypothetical questions that the below document could be used to answer:\n\n{doc}")
                        | chatllm.bind(functions=functions, function_call={"name": "hypothetical_questions"})
                        | JsonKeyOutputFunctionsParser(key_name="questions")
                    )
                    print("Hypothetical Queries OUTPUT: Not Working!")
                    #print(chain.invoke(docs[0]))
                else:
                    print("Mission Impossileb!")
            case 24|25:
                ##### Test #24 and #25: Parent Document Retriever #####
                # ParentDocumentRetriever first fetches the small chunks but then looks up the parent ids for those chunks and returns those larger documents.
                # The "parent document" refers to the document that a small chunk originated from.
                embeddings_model = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)             
                loaders = [
                    TextLoader('./data/paul_graham_essay.txt', encoding='utf8'),
                    TextLoader('./data/state_of_the_union.txt', encoding='utf8'),
                ]
                docs = []
                for l in loaders:
                    docs.extend(l.load())

                # This text splitter is used to create the parent documents
                parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
                # This text splitter is used to create the child documents
                # It should create documents smaller than the parent
                child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
                # The vectorstore to use to index the child chunks
                vectorstore = Chroma(collection_name="split_parents", embedding_function=embeddings_model)
                # The storage layer for the parent documents
                store = InMemoryStore()
                
                if test_option == 24:
                    # Test #24 Retrieving full documents: Only use the child splitter.
                    retriever = ParentDocumentRetriever(
                        vectorstore=vectorstore, 
                        docstore=store, 
                        child_splitter=child_splitter,
                    )
                    retriever.add_documents(docs, ids=None)
                    print("Length of documents: ", len(list(store.yield_keys()))) # This should yield two keys, because we added two documents.
                    
                    print("ParentDocumentRetriever (Full documents) OUTPUT:")
                    sub_docs = vectorstore.similarity_search("justice breyer")
                    print("(1) Small chunks (vectorstore.similarity_search):\n", sub_docs[0].page_content)
                    retrieved_docs = retriever.get_relevant_documents("justice breyer")
                    print("\n(2) Size of large documents (retriever.get_relevant_documents): ", len(retrieved_docs[0].page_content))
                elif test_option == 25:
                    # Test #25 Retrieving larger chunks
                    # The full documents can be too big to want to retrieve them as is.
                    # Split the raw documents into larger chunks, and then split it into smaller chunks.
                    retriever = ParentDocumentRetriever(
                        vectorstore=vectorstore, 
                        docstore=store, 
                        child_splitter=child_splitter,
                        parent_splitter=parent_splitter,
                    )
                    retriever.add_documents(docs)
                    print("Length of documents: ", len(list(store.yield_keys()))) # The larger chunks are much more than two documents.

                    print("ParentDocumentRetriever (Larger chunks) OUTPUT:")
                    sub_docs = vectorstore.similarity_search("justice breyer")
                    print("(1) Small chunks (vectorstore.similarity_search):\n", sub_docs[0].page_content)
                    retrieved_docs = retriever.get_relevant_documents("justice breyer")
                    print("\n(2) Size of large documents (retriever.get_relevant_documents): ", len(retrieved_docs[0].page_content))
                else:
                    print("Mission Impossileb!")
            case 26:
                ##### Test #26: Self-querying #####
                # The retriever uses a "query-constructing LLM chain" to write a structured query and then applies that structured query to its underlying VectorStore.
                embeddings_model = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)             
                docs = [
                    Document(
                        page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose",
                        metadata={"year": 1993, "rating": 7.7, "genre": "science fiction"},
                    ),
                    Document(
                        page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...",
                        metadata={"year": 2010, "director": "Christopher Nolan", "rating": 8.2},
                    ),
                    Document(
                        page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea",
                        metadata={"year": 2006, "director": "Satoshi Kon", "rating": 8.6},
                    ),
                    Document(
                        page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them",
                        metadata={"year": 2019, "director": "Greta Gerwig", "rating": 8.3},
                    ),
                    Document(
                        page_content="Toys come alive and have a blast doing so",
                        metadata={"year": 1995, "genre": "animated"},
                    ),
                    Document(
                        page_content="Three men walk into the Zone, three men walk out of the Zone",
                        metadata={
                            "year": 1979,
                            "rating": 9.9,
                            "director": "Andrei Tarkovsky",
                            "genre": "science fiction",
                            "rating": 9.9,
                        },
                    ),
                ]
                vectorstore = Chroma.from_documents(docs, embeddings_model)

                # Creating our self-querying retriever
                metadata_field_info = [
                    AttributeInfo(
                        name="genre",
                        description="The genre of the movie",
                        type="string or list[string]",
                    ),
                    AttributeInfo(
                        name="year",
                        description="The year the movie was released",
                        type="integer",
                    ),
                    AttributeInfo(
                        name="director",
                        description="The name of the movie director",
                        type="string",
                    ),
                    AttributeInfo(
                        name="rating", description="A 1-10 rating for the movie", type="float"
                    ),
                ]
                document_content_description = "Brief summary of a movie"
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase, temperature=0)
                retriever = SelfQueryRetriever.from_llm(
                    chatllm,
                    vectorstore,
                    document_content_description,
                    metadata_field_info,
                    #enable_limit=True, # Filter k: the number of documents to fetch.
                    verbose=True,
                )

                print("Self-querying OUTPUT:")
                # This example only specifies a relevant query
                print("\n(1) A relevant query only:\n")
                retrieved_docs = retriever.get_relevant_documents("What are some movies about dinosaurs")
                _pretty_print_docs(retrieved_docs)
                # This example only specifies a filter
                print("\n(2) A filter only:\n")
                retrieved_docs = retriever.get_relevant_documents("I want to watch a movie rated higher than 8.5")
                _pretty_print_docs(retrieved_docs)
                # This example specifies a query and a filter
                print("\n(3) A query and a filter:\n")
                retrieved_docs = retriever.get_relevant_documents("Has Greta Gerwig directed any movies about women")
                _pretty_print_docs(retrieved_docs)
                # This example specifies a composite filter
                print("\n(4) A composite filter only:\n")
                retrieved_docs = retriever.get_relevant_documents(
                    "What's a highly rated (above 8.5) science fiction film?"
                )
                _pretty_print_docs(retrieved_docs)
                # This example specifies a query and composite filter
                print("\n(5) A query and composite filter:\n")
                retrieved_docs = retriever.get_relevant_documents(
                    "What's a movie after 1990 but before 2005 that's all about toys, and preferably is animated"
                )
                _pretty_print_docs(retrieved_docs)
            case 27:
                ##### Test #27: Time-weighted vector store retriever #####
                # Define your embedding model
                embeddings_model = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)             
                #chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase, temperature=0, max_retries=0, model="gpt-4")

                # Initialize the vectorstore as empty
                embedding_size = 1536
                index = faiss.IndexFlatL2(embedding_size)
                vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
                
                print("Time-weighted vector store retriever OUTPUT:")
                # Type 1 Low decay rate
                retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.0000000000000000000000001, k=1)
                yesterday = datetime.now() - timedelta(days=1)
                retriever.add_documents([Document(page_content="hello world", metadata={"last_accessed_at": yesterday})])
                retriever.add_documents([Document(page_content="hello foo")])
               
                print("\n(1) Low decay rate:\n")
                # "Hello World" is returned first because it is most salient, and the decay rate is close to 0., meaning it's still recent enough
                retrieved_docs = retriever.get_relevant_documents("hello world") 
                _pretty_print_docs(retrieved_docs)

                # Type 2 High decay rate
                print("\n(2) High decay rate:\n")
                retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.999, k=1)
                yesterday = datetime.now() - timedelta(days=1)
                retriever.add_documents([Document(page_content="hello world", metadata={"last_accessed_at": yesterday})])
                retriever.add_documents([Document(page_content="hello foo")])
                # "Hello Foo" is returned first because "hello world" is mostly forgotten
                retrieved_docs = retriever.get_relevant_documents("hello world") 
                _pretty_print_docs(retrieved_docs)

                # Type 3 Virtual time: Mock out the time component.
                print("\n(3) Virtual time:\n")
                # Notice the last access time is that date time
                with mock_now(datetime(2023, 10, 30, 10, 11)):
                    #_pretty_print_docs(retriever.get_relevant_documents("hello world"))
                    retrieved_docs = retriever.get_relevant_documents("hello world") 
                    _pretty_print_docs(retrieved_docs)
            case 28:
                ##### Test #28: Router #####
                embeddings_model = OpenAIEmbeddings(deployment=azure_embd_deployment, openai_api_key=azure_apikey, openai_api_version=azure_apiversion, openai_api_type=azure_apitype, openai_api_base=azure_apibase)             
                chatllm = AzureChatOpenAI(deployment_name=azure_gptx_deployment, openai_api_version=azure_apiversion, openai_api_key=azure_apikey, openai_api_base=azure_apibase, temperature=0, max_retries=0)

                physics_template = """You are a very smart physics professor. \
                You are great at answering questions about physics in a concise and easy to understand manner. \
                When you don't know the answer to a question you admit that you don't know.

                Here is a question:
                {input}"""

                math_template = """You are a very good mathematician. You are great at answering math questions. \
                You are so good because you are able to break down hard problems into their component parts, \
                answer the component parts, and then put them together to answer the broader question.

                Here is a question:
                {input}"""

                prompt_infos = [
                    {
                        "name": "physics",
                        "description": "Good for answering questions about physics",
                        "prompt_template": physics_template,
                    },
                    {
                        "name": "math",
                        "description": "Good for answering math questions",
                        "prompt_template": math_template,
                    },
                ]

                destination_chains = {}
                for p_info in prompt_infos:
                    name = p_info["name"]
                    prompt_template = p_info["prompt_template"]
                    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
                    chain = LLMChain(llm=chatllm, prompt=prompt)
                    destination_chains[name] = chain
                default_chain = ConversationChain(llm=chatllm, output_key="text")

                # Type 1 LLMRouterChain: This chain uses an LLM to determine how to route things.
                destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
                destinations_str = "\n".join(destinations)
                router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
                router_prompt = PromptTemplate(
                    template=router_template,
                    input_variables=["input"],
                    output_parser=RouterOutputParser(),
                )
                router_chain = LLMRouterChain.from_llm(chatllm, router_prompt)

                chain = MultiPromptChain(
                    router_chain=router_chain,
                    destination_chains=destination_chains,
                    default_chain=default_chain,
                    verbose=True,
                )
                print("(1) LLMRouterChain OUTPUT:")
                print("(1-1) Query: What is black body radiation?")
                print(chain.run("What is black body radiation?"))
                print("\n\n")
                print("(1-2) Query: What is the first prime number greater than 40 such that one plus the prime number is divisible by 3?")
                print(
                    chain.run(
                        "What is the first prime number greater than 40 such that one plus the prime number is divisible by 3?"
                    )
                )
                print("\n\n")
                print("(1-3) Query: ", test_phrase)
                print(chain.run(test_phrase))

                print("\n\n\n")
                # Type 2 EmbeddingRouterChain: This chain uses embeddings and similarity to route between destination chains.
                names_and_descriptions = [
                    ("physics", ["for questions about physics"]),
                    ("math", ["for questions about math"]),
                ]
                router_chain = EmbeddingRouterChain.from_names_and_descriptions(
                    names_and_descriptions, Chroma, embeddings_model, routing_keys=["input"]
                )
                chain = MultiPromptChain(
                    router_chain=router_chain,
                    destination_chains=destination_chains,
                    default_chain=default_chain,
                    verbose=True,
                )
                print("(2) EmbeddingRouterChain OUTPUT:")
                print("(2-1) Query: What is black body radiation?")
                print(chain.run("What is black body radiation?"))
                print("\n\n")
                print("(2-2) Query: What is the first prime number greater than 40 such that one plus the prime number is divisible by 3?")
                print(
                    chain.run(
                        "What is the first prime number greater than 40 such that one plus the prime number is divisible by 3?"
                    )
                )
                print("\n\n")
                print("(2-3) Query: ", test_phrase)
                print(chain.run(test_phrase))
            case _:
                print(f"Error: Wrong Test Option ({test_option})")

    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())