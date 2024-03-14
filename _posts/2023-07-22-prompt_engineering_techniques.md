---
layout: post
title: Prompt Engineering Techniques for LLM's
subtitle: 
cover-img: 
thumbnail-img: 
share-img: 
tags: [LLM, Prompt Engineering, chain of thoughts, ReAct, LangChain]
author: Daksh Jain
---
In this notebook objective is to demonstrate various prompt engineering techniques like zero-shot , one-shot , few-shot , chain of thoughts & ReAct using OpenAI python SDK & Langchain



```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
!pip install openai 

```

    Requirement already satisfied: openai in /opt/conda/lib/python3.10/site-packages (1.14.0)
    Requirement already satisfied: anyio<5,>=3.5.0 in /opt/conda/lib/python3.10/site-packages (from openai) (4.2.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /opt/conda/lib/python3.10/site-packages (from openai) (1.9.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /opt/conda/lib/python3.10/site-packages (from openai) (0.27.0)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /opt/conda/lib/python3.10/site-packages (from openai) (2.5.3)
    Requirement already satisfied: sniffio in /opt/conda/lib/python3.10/site-packages (from openai) (1.3.0)
    Requirement already satisfied: tqdm>4 in /opt/conda/lib/python3.10/site-packages (from openai) (4.66.1)
    Requirement already satisfied: typing-extensions<5,>=4.7 in /opt/conda/lib/python3.10/site-packages (from openai) (4.9.0)
    Requirement already satisfied: idna>=2.8 in /opt/conda/lib/python3.10/site-packages (from anyio<5,>=3.5.0->openai) (3.6)
    Requirement already satisfied: exceptiongroup>=1.0.2 in /opt/conda/lib/python3.10/site-packages (from anyio<5,>=3.5.0->openai) (1.2.0)
    Requirement already satisfied: certifi in /opt/conda/lib/python3.10/site-packages (from httpx<1,>=0.23.0->openai) (2024.2.2)
    Requirement already satisfied: httpcore==1.* in /opt/conda/lib/python3.10/site-packages (from httpx<1,>=0.23.0->openai) (1.0.4)
    Requirement already satisfied: h11<0.15,>=0.13 in /opt/conda/lib/python3.10/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
    Requirement already satisfied: annotated-types>=0.4.0 in /opt/conda/lib/python3.10/site-packages (from pydantic<3,>=1.9.0->openai) (0.6.0)
    Requirement already satisfied: pydantic-core==2.14.6 in /opt/conda/lib/python3.10/site-packages (from pydantic<3,>=1.9.0->openai) (2.14.6)



```python
# loading secret key for API calls

from openai import OpenAI
import openai
import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
openai.api_key = "add you API key here"#user_secrets.get_secret("openai_api")
client = OpenAI(
    # This is the default and can be omitted
    api_key=openai.api_key,
)
```


```python
#function to prompt gpt-3.5-turbo openai model
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content
```

# Basic Prompt Engineering Techniques

Below is an example of zero shot prompting (where no example is provided to model)


```python
prompt = f"""
Classify this review:
I love red apples
Sentiment:
"""
response = get_completion(prompt)
print(response)
```

    Positive


Below is example of one shot prompting (where only single example is provided to model)


```python
prompt = f"""
Classify this review:
I love red apples
Sentiment: Positive

Classify this review
I adore vegetables
Sentiment :
"""
response = get_completion(prompt)
print(response)
```

    Positive


Below is example of few shot prompting (where more than one example is provided to model)


```python
prompt = f"""
Classify this review:
I love red apples
Sentiment: positive

Classify this review
This is not delicious
Sentiment : negative

Classify this review:
I don't like meat
Sentiment:
"""
response = get_completion(prompt)
print(response)
```

    Sentiment: negative


# Prompt Engineering - Using chain of thoughts

Below is an example comparison where we see how chain of thoughts improve model results


```python
prompt = f"""
Q: Roger has 5 tennis balls. He buys 2 more cans of
tennis balls. Each can has 3 tennis balls. How many
tennis balls does he have now?

A: The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to
make lunch and bought 6 more, how many apples
do they have?
"""
response = get_completion(prompt)
print(response)
```

    A: The cafeteria now has 9 apples.


now we will add more explaination to our result which will help model to learn better and provide reason for the result


```python
prompt = f"""
Q: Roger has 5 tennis balls. He buys 2 more cans of
tennis balls. Each can has 3 tennis balls. How many
tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls
each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: The cafeteria had 23 apples. If they used 20 to
make lunch and bought 6 more, how many apples
do they have?
"""
response = get_completion(prompt)
print(response)
```

    A: The cafeteria started with 23 apples. They used 20 for lunch, so they have 23 - 20 = 3 apples left. Then they bought 6 more, so they now have 3 + 6 = 9 apples. The answer is 9.


# ReACT (Reasoning & acting)- using Langchain

This method uses LLM to decide for reasoning & actions for a task by interacting with external enviornment (resources like web)


```python
!pip install --upgrade langchain
!pip install --upgrade python-dotenv
!pip install google-search-results
 
# import libraries
import os
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from dotenv import load_dotenv
load_dotenv()
 
# load API keys; you will need to obtain these if you haven't yet
os.environ["OPENAI_API_KEY"] = #add you api key here
os.environ["SERPER_API_KEY"] = #add your api key here
```

    Collecting langchain
      Downloading langchain-0.1.12-py3-none-any.whl.metadata (13 kB)
    Requirement already satisfied: PyYAML>=5.3 in /opt/conda/lib/python3.10/site-packages (from langchain) (6.0.1)
    Requirement already satisfied: SQLAlchemy<3,>=1.4 in /opt/conda/lib/python3.10/site-packages (from langchain) (2.0.25)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /opt/conda/lib/python3.10/site-packages (from langchain) (3.9.1)
    Requirement already satisfied: async-timeout<5.0.0,>=4.0.0 in /opt/conda/lib/python3.10/site-packages (from langchain) (4.0.3)
    Requirement already satisfied: dataclasses-json<0.7,>=0.5.7 in /opt/conda/lib/python3.10/site-packages (from langchain) (0.6.4)
    Requirement already satisfied: jsonpatch<2.0,>=1.33 in /opt/conda/lib/python3.10/site-packages (from langchain) (1.33)
    Collecting langchain-community<0.1,>=0.0.28 (from langchain)
      Downloading langchain_community-0.0.28-py3-none-any.whl.metadata (8.3 kB)
    Collecting langchain-core<0.2.0,>=0.1.31 (from langchain)
      Downloading langchain_core-0.1.31-py3-none-any.whl.metadata (6.0 kB)
    Collecting langchain-text-splitters<0.1,>=0.0.1 (from langchain)
      Downloading langchain_text_splitters-0.0.1-py3-none-any.whl.metadata (2.0 kB)
    Collecting langsmith<0.2.0,>=0.1.17 (from langchain)
      Downloading langsmith-0.1.25-py3-none-any.whl.metadata (13 kB)
    Requirement already satisfied: numpy<2,>=1 in /opt/conda/lib/python3.10/site-packages (from langchain) (1.26.4)
    Requirement already satisfied: pydantic<3,>=1 in /opt/conda/lib/python3.10/site-packages (from langchain) (2.5.3)
    Requirement already satisfied: requests<3,>=2 in /opt/conda/lib/python3.10/site-packages (from langchain) (2.31.0)
    Requirement already satisfied: tenacity<9.0.0,>=8.1.0 in /opt/conda/lib/python3.10/site-packages (from langchain) (8.2.3)
    Requirement already satisfied: attrs>=17.3.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (23.2.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/conda/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (6.0.4)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/conda/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.9.3)
    Requirement already satisfied: frozenlist>=1.1.1 in /opt/conda/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.4.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /opt/conda/lib/python3.10/site-packages (from aiohttp<4.0.0,>=3.8.3->langchain) (1.3.1)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /opt/conda/lib/python3.10/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain) (3.20.2)
    Requirement already satisfied: typing-inspect<1,>=0.4.0 in /opt/conda/lib/python3.10/site-packages (from dataclasses-json<0.7,>=0.5.7->langchain) (0.9.0)
    Requirement already satisfied: jsonpointer>=1.9 in /opt/conda/lib/python3.10/site-packages (from jsonpatch<2.0,>=1.33->langchain) (2.4)
    Requirement already satisfied: anyio<5,>=3 in /opt/conda/lib/python3.10/site-packages (from langchain-core<0.2.0,>=0.1.31->langchain) (4.2.0)
    Collecting packaging<24.0,>=23.2 (from langchain-core<0.2.0,>=0.1.31->langchain)
      Downloading packaging-23.2-py3-none-any.whl.metadata (3.2 kB)
    Collecting orjson<4.0.0,>=3.9.14 (from langsmith<0.2.0,>=0.1.17->langchain)
      Downloading orjson-3.9.15-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (49 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m49.5/49.5 kB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: annotated-types>=0.4.0 in /opt/conda/lib/python3.10/site-packages (from pydantic<3,>=1->langchain) (0.6.0)
    Requirement already satisfied: pydantic-core==2.14.6 in /opt/conda/lib/python3.10/site-packages (from pydantic<3,>=1->langchain) (2.14.6)
    Requirement already satisfied: typing-extensions>=4.6.1 in /opt/conda/lib/python3.10/site-packages (from pydantic<3,>=1->langchain) (4.9.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests<3,>=2->langchain) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests<3,>=2->langchain) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests<3,>=2->langchain) (1.26.18)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests<3,>=2->langchain) (2024.2.2)
    Requirement already satisfied: greenlet!=0.4.17 in /opt/conda/lib/python3.10/site-packages (from SQLAlchemy<3,>=1.4->langchain) (3.0.3)
    Requirement already satisfied: sniffio>=1.1 in /opt/conda/lib/python3.10/site-packages (from anyio<5,>=3->langchain-core<0.2.0,>=0.1.31->langchain) (1.3.0)
    Requirement already satisfied: exceptiongroup>=1.0.2 in /opt/conda/lib/python3.10/site-packages (from anyio<5,>=3->langchain-core<0.2.0,>=0.1.31->langchain) (1.2.0)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /opt/conda/lib/python3.10/site-packages (from typing-inspect<1,>=0.4.0->dataclasses-json<0.7,>=0.5.7->langchain) (1.0.0)
    Downloading langchain-0.1.12-py3-none-any.whl (809 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m809.1/809.1 kB[0m [31m21.4 MB/s[0m eta [36m0:00:00[0m00:01[0m
    [?25hDownloading langchain_community-0.0.28-py3-none-any.whl (1.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.8/1.8 MB[0m [31m40.1 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hDownloading langchain_core-0.1.31-py3-none-any.whl (258 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m258.8/258.8 kB[0m [31m13.6 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading langchain_text_splitters-0.0.1-py3-none-any.whl (21 kB)
    Downloading langsmith-0.1.25-py3-none-any.whl (67 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m67.6/67.6 kB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading orjson-3.9.15-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (138 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m138.5/138.5 kB[0m [31m6.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading packaging-23.2-py3-none-any.whl (53 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m53.0/53.0 kB[0m [31m2.6 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: packaging, orjson, langsmith, langchain-core, langchain-text-splitters, langchain-community, langchain
      Attempting uninstall: packaging
        Found existing installation: packaging 21.3
        Uninstalling packaging-21.3:
          Successfully uninstalled packaging-21.3
      Attempting uninstall: orjson
        Found existing installation: orjson 3.9.10
        Uninstalling orjson-3.9.10:
          Successfully uninstalled orjson-3.9.10
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    keras-cv 0.8.2 requires keras-core, which is not installed.
    keras-nlp 0.8.1 requires keras-core, which is not installed.
    tensorflow-decision-forests 1.8.1 requires wurlitzer, which is not installed.
    apache-beam 2.46.0 requires dill<0.3.2,>=0.3.1.1, but you have dill 0.3.8 which is incompatible.
    apache-beam 2.46.0 requires numpy<1.25.0,>=1.14.3, but you have numpy 1.26.4 which is incompatible.
    apache-beam 2.46.0 requires pyarrow<10.0.0,>=3.0.0, but you have pyarrow 15.0.0 which is incompatible.
    google-cloud-bigquery 2.34.4 requires packaging<22.0dev,>=14.3, but you have packaging 23.2 which is incompatible.
    jupyterlab 4.1.2 requires jupyter-lsp>=2.0.0, but you have jupyter-lsp 1.5.1 which is incompatible.
    jupyterlab-lsp 5.0.3 requires jupyter-lsp>=2.0.0, but you have jupyter-lsp 1.5.1 which is incompatible.
    libpysal 4.9.2 requires shapely>=2.0.1, but you have shapely 1.8.5.post1 which is incompatible.
    momepy 0.7.0 requires shapely>=2, but you have shapely 1.8.5.post1 which is incompatible.
    osmnx 1.9.1 requires shapely>=2.0, but you have shapely 1.8.5.post1 which is incompatible.
    spopt 0.6.0 requires shapely>=2.0.1, but you have shapely 1.8.5.post1 which is incompatible.
    tensorflow 2.15.0 requires keras<2.16,>=2.15.0, but you have keras 3.0.5 which is incompatible.
    ydata-profiling 4.6.4 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.[0m[31m
    [0mSuccessfully installed langchain-0.1.12 langchain-community-0.0.28 langchain-core-0.1.31 langchain-text-splitters-0.0.1 langsmith-0.1.25 orjson-3.9.15 packaging-23.2
    Requirement already satisfied: python-dotenv in /opt/conda/lib/python3.10/site-packages (1.0.0)
    Collecting python-dotenv
      Downloading python_dotenv-1.0.1-py3-none-any.whl.metadata (23 kB)
    Downloading python_dotenv-1.0.1-py3-none-any.whl (19 kB)
    Installing collected packages: python-dotenv
      Attempting uninstall: python-dotenv
        Found existing installation: python-dotenv 1.0.0
        Uninstalling python-dotenv-1.0.0:
          Successfully uninstalled python-dotenv-1.0.0
    Successfully installed python-dotenv-1.0.1
    Collecting google-search-results
      Downloading google_search_results-2.4.2.tar.gz (18 kB)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: requests in /opt/conda/lib/python3.10/site-packages (from google-search-results) (2.31.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests->google-search-results) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests->google-search-results) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests->google-search-results) (1.26.18)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests->google-search-results) (2024.2.2)
    Building wheels for collected packages: google-search-results
      Building wheel for google-search-results (setup.py) ... [?25ldone
    [?25h  Created wheel for google-search-results: filename=google_search_results-2.4.2-py3-none-any.whl size=32010 sha256=4ceb08d878e5516aaa099d9d2f488eb30a459d6d4b89d57ae0c880bbd1a3ff3b
      Stored in directory: /root/.cache/pip/wheels/d3/b2/c3/03302d12bb44a2cdff3c9371f31b72c0c4e84b8d2285eeac53
    Successfully built google-search-results
    Installing collected packages: google-search-results
    Successfully installed google-search-results-2.4.2



```python
llm = OpenAI(model="gpt-3.5-turbo-instruct" ,temperature=0)
tools = load_tools(["google-serper", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
```


```python
agent.run("Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?")
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m I should use google_serper to find information about Olivia Wilde's boyfriend.
    Action: google_serper
    Action Input: "Olivia Wilde boyfriend"[0m
    Observation: [36;1m[1;3mTao Ruspoli[0m
    Thought:[32;1m[1;3m Now that I know Olivia Wilde's boyfriend's name, I can use google_serper to find his current age.
    Action: google_serper
    Action Input: "Tao Ruspoli age"[0m
    Observation: [36;1m[1;3mOlivia Wilde[0m
    Thought:[32;1m[1;3m I should use Calculator to raise Tao Ruspoli's current age to the 0.23 power.
    Action: Calculator
    Action Input: 32 ^ 0.23[0m
    Observation: [33;1m[1;3mAnswer: 2.21913894413569[0m
    Thought:[32;1m[1;3m I now know the final answer.
    Final Answer: 2.21913894413569[0m
    
    [1m> Finished chain.[0m





    '2.21913894413569'




```python

```
