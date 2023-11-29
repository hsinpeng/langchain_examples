import os
import sys
import time
import asyncio
import requests
from langserve import RemoteRunnable

os.environ["NO_PROXY"] = "localhost, 127.0.0.1"

async def async_test():
    print("######## Async Calls ########")
     # RemoteRunnable initialization
    remote_chain = RemoteRunnable("http://localhost:8000/test_chain/")
    
    try:
        # aInvoke
        print("----- AsyncInvoke -----")
        start = time.perf_counter()
        result = await remote_chain.ainvoke({"text": "Taiwan"})
        end = time.perf_counter()
        print(result.content)
        print("----- AsyncInvoke exec-time: %f secs -----" % (end - start))

        # aInvoke with task
        print("----- AsyncInvoke with task -----")
        task = asyncio.create_task(remote_chain.ainvoke({"text": "Love"}))
        await asyncio.sleep(1) # MUST have for executing the task above

        for i in range(5):
            time.sleep(1)
            print(i)

        start = time.perf_counter()
        result = await task
        end = time.perf_counter()

        print(result.content)
        print("----- AsyncInvoke with task exec-time: %f 秒 -----" % (end - start))

        # Batch aInvoke
        print("----- Batch AsyncInvoke -----")
        job1 = remote_chain.ainvoke({"text": "Taipei"})
        job2 = remote_chain.ainvoke({"text": "Taichung"})
        job3 = remote_chain.ainvoke({"text": "Kaohsiung"})

        start = time.perf_counter()
        results = await asyncio.gather(job1, job2, job3)
        end = time.perf_counter()

        for ret in results:
            print("output:")
            print(ret.content)
        print("----- Batch AsyncInvoke exec-time: %f secs -----" % (end - start))

        # aBatch
        print("----- AsyncBatch -----")
        start = time.perf_counter()
        results = await remote_chain.abatch([{"text": "Taipei"}, {"text": "Taichung"}, {"text": "Kaohsiung"}])
        end = time.perf_counter()
        for ret in results:
            print("output:")
            print(ret.content)
        print("----- AsyncBatch exec-time: %f secs -----" % (end - start))

    except ValueError as ve:
        print(str(ve))
        return str(ve)
    

def sync_test():
    print("######## Sync Calls ########")
    querystr = "love"
    
    try:
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
        print(str(ve))
        return str(ve)

def main():
    try:
        # Sync
        #sync_test()

        # Async
        #asyncio.run(async_test())
        asyncio.get_event_loop().run_until_complete(async_test())
    
    except ValueError as ve:
        return str(ve)

if __name__ == "__main__":
    sys.exit(main())