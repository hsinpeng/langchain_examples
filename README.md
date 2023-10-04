# Introduction 
My personal study of langchain with Azure OpenAI. The most content are modified from the official langchain document.

The official langchain document website: https://python.langchain.com/docs/

# Getting Started
1. Clone this project from GitHub.
```text=
git clone [Project URL]
```

2. Enter project folder (langchain_examples).
```text=
cd langchain_examples
```

3. New the parameter file (param.json) in the project folder, the formate is as the following.

Note: Before using this project, you have to set and get these parameter from Azure AI service.
```text=
{
    "azure_apikey" : "Please, do it by yourself", 
    "azure_apibase"  : "Please, do it by yourself",
    "azure_apitype" : "azure",
    "azure_apiversion" : "2023-05-15",
    "azure_gptx_deployment" : "Please, do it by yourself",
    "azure_embd_deployment" : "Please, do it by yourself"
}
```

4. Install related python packages.
```text=
python3 -m pip install --no-cache-dir -U pip
python3 -m pip install --no-cache-dir -U openai
python3 -m pip install --no-cache-dir -U langchain
python3 -m pip install --no-cache-dir -U charomadb
python3 -m pip install --no-cache-dir -U tiktoken
python3 -m pip install --no-cache-dir -U faiss-cpu
python3 -m pip install --no-cache-dir -U nltk
python3 -m pip install --no-cache-dir -U wikipedia
python3 -m pip install --no-cache-dir -U rank_bm25
python3 -m pip install --no-cache-dir -U lark 
```

5. The main program is 'test.py', check it and learn how to use it by inspecting source code.
```text=
Hints:
(1) Change test example by modifying the test_option variable.
(2) Change test phrase by modifying the test_phrase variable.
```

6. Program execution.
```text=
python test.py
```

# Contribute
TODO: Explain how other users and developers can contribute to make your code better. 