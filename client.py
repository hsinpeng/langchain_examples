import sys
import time
import requests
from langserve import RemoteRunnable

def main():
    try:
        querystr = "love"

        # Python requests
        print("----- Python requests -----")
        start = time.perf_counter()
        response = requests.post(
            "http://localhost:8000/test_chain/invoke/",
            json={'input': {'text': querystr}}
        )
        end = time.perf_counter()
        print(response.json()["output"]["content"])
        print("----- Python requests exec-time: %f secs -----" % (end - start))
        
        # RemoteRunnable initialization
        remote_chain = RemoteRunnable("http://localhost:8000/test_chain/")
        
        # Invoke
        print("----- Invoke -----")
        start = time.perf_counter()
        result = remote_chain.invoke({"text": querystr})
        end = time.perf_counter()
        print(result.content)
        print("----- Invoke exec-time: %f secs -----" % (end - start))

        # Stream
        print("----- Stream -----")
        chk = True
        start = time.perf_counter()
        for s in remote_chain.stream({"text": querystr}):
            print(s.content, end="", flush=True)
            if chk:
                end = time.perf_counter()
                chk = False
        print("\n----- Stream response-time: %f secs -----" % (end - start))

        # Batch
        print("----- Batch -----")
        start = time.perf_counter()
        results = remote_chain.batch([{"text": querystr}, {"text": querystr}])
        end = time.perf_counter()
        for e in results:
            print("output:")
            print(e.content)
        print("----- Batch exec-time: %f secs -----" % (end - start))
    
    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())