import os
import sys
import time
from langserve import RemoteRunnable

os.environ["NO_PROXY"] = "localhost, 127.0.0.1"
test_option = 5

def api_test(option):
    print("######## LangServe API Test ########")
    
    try:
        match option:
            case 0:
                print('Just Test!')
            case 1:
                print("----- Query (Chat without history) -----")
                remote_chain = RemoteRunnable("http://localhost:8000/query/")
                start = time.perf_counter()
                result = remote_chain.invoke({"question": "How to cycling around Taiwan?"})
                end = time.perf_counter()
                print(result.content)
                print("----- Query exec-time: %f secs -----" % (end - start))
            case 2:
                print("----- Chat with history -----")
                remote_chain = RemoteRunnable("http://localhost:8000/chat/")
                start = time.perf_counter()
                result = remote_chain.invoke(
                    {"question": "What is the current US president?"},
                    config={"configurable": {"session_id": "foobar"}},
                )
                end = time.perf_counter()
                print(result.content)
                print("----- Chat 1 exec-time: %f secs -----" % (end - start))

                start = time.perf_counter()
                result = remote_chain.invoke(
                    {"question": "How old is he?"},
                    config={"configurable": {"session_id": "foobar"}},
                )
                end = time.perf_counter()
                print(result.content)
                print("----- Chat 2 exec-time: %f secs -----" % (end - start))
            case 3:
                print("----- Cypher Generator -----")
                neo_schema = """
                Node properties are the following:
                Movie {name: STRING},Actor {name: STRING}
                Relationship properties are the following:

                The relationships are the following:
                (:Actor)-[:ACTED_IN]->(:Movie)
                """
                remote_chain = RemoteRunnable("http://localhost:8000/cypher/")
                start = time.perf_counter()
                result = remote_chain.invoke({"neo4j_schema": neo_schema, "question": "Who played in Top Gun?"})
                end = time.perf_counter()
                print(result.content)
                print("----- Cypher Gen exec-time: %f secs -----" % (end - start))
            case 4:
                print("--- Chat with obliged knowledge ---")
                print("Step 1: Get History and Knowledge")
                chat_session_id = "id_obliged"
                remote_chain = RemoteRunnable("http://localhost:8000/get_history/")
                history_messages = remote_chain.invoke({"session_id": chat_session_id})
                print(history_messages)
                
                print("Step 2: Ask Question with History and Knowledge (Note: Streamable)")
                human_question = "Who is Taipei city mayor?"
                required_language = "Traditional Chinese"
                given_knowledge = ""
                remote_chain = RemoteRunnable("http://localhost:8000/custom_obliged_query/")
                ai_answer = remote_chain.invoke({"history": history_messages, "language": required_language, "knowledge": given_knowledge, "question": human_question})
                print(ai_answer)

                print("Step 3: Save the Latest Question and Answer")
                remote_chain = RemoteRunnable("http://localhost:8000/save_history/")
                result = remote_chain.invoke({"session_id": chat_session_id, "question": human_question, "answer": ai_answer})
                print(result)
            case 5:
                print("--- Chat with referential knowledge ---")
                print("Step 1: Get History and Knowledge")
                chat_session_id = "id_referential"
                remote_chain = RemoteRunnable("http://localhost:8000/get_history/")
                history_messages = remote_chain.invoke({"session_id": chat_session_id})
                print(history_messages)
                
                print("Step 2: Ask Question with History and Knowledge (Note: Streamable)")
                human_question = "Who is Tokyo city mayor?"
                required_language = "Traditional Chinese"
                given_knowledge = ""
                remote_chain = RemoteRunnable("http://localhost:8000/custom_referential_query/")
                ai_answer = remote_chain.invoke({"history": history_messages, "language": required_language, "knowledge": given_knowledge, "question": human_question})
                print(ai_answer)

                print("Step 3: Save the Latest Question and Answer")
                remote_chain = RemoteRunnable("http://localhost:8000/save_history/")
                result = remote_chain.invoke({"session_id": chat_session_id, "question": human_question, "answer": ai_answer})
                print(result)
            case _:
                print('Error: Wrong option!')

    except ValueError as ve:
        print(str(ve))
        return str(ve)

def main():
    try:
        api_test(test_option)
    
    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())