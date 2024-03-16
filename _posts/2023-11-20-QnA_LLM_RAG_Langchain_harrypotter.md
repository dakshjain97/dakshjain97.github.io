---
layout: post
title: Question & Answer on Harry Potter and the Sorcerer's Stone using LLM
subtitle: 
cover-img: 
thumbnail-img: 
share-img: 
tags: [LLM, LangChain, RAG, FAISS, Harry potter]
author: Daksh Jain
---
In this notebook idea is to demonstrate a POC on question and answer model on Harry Potter and the Sorcerer's Stone book using LLM . Comparison is done by querying a prompt to LLM vs using Langchain to add additional context to prompt from book, as a result we observe by using RAG technique LLM is able to provide more precise response as compared to general response by LLM . For similarity search & vector db storage FAISS is used


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

    /kaggle/input/harry-potter-sorcerers-stone/harry-potter-sorcerers-stone.pdf



```python
! pip install sentence_transformers==2.2.2

! pip install -qq -U langchain
! pip install -qq -U tiktoken
! pip install -qq -U pypdf
! pip install -qq -U faiss-gpu
! pip install -qq -U InstructorEmbedding 

! pip install -qq -U transformers 
! pip install -qq -U accelerate
! pip install -qq -U bitsandbytes
```

    Collecting sentence_transformers==2.2.2
      Downloading sentence-transformers-2.2.2.tar.gz (85 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m86.0/86.0 kB[0m [31m929.1 kB/s[0m eta [36m0:00:00[0m [36m0:00:01[0mm
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: transformers<5.0.0,>=4.6.0 in /opt/conda/lib/python3.10/site-packages (from sentence_transformers==2.2.2) (4.38.1)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.10/site-packages (from sentence_transformers==2.2.2) (4.66.1)
    Requirement already satisfied: torch>=1.6.0 in /opt/conda/lib/python3.10/site-packages (from sentence_transformers==2.2.2) (2.1.2)
    Requirement already satisfied: torchvision in /opt/conda/lib/python3.10/site-packages (from sentence_transformers==2.2.2) (0.16.2)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.10/site-packages (from sentence_transformers==2.2.2) (1.26.4)
    Requirement already satisfied: scikit-learn in /opt/conda/lib/python3.10/site-packages (from sentence_transformers==2.2.2) (1.2.2)
    Requirement already satisfied: scipy in /opt/conda/lib/python3.10/site-packages (from sentence_transformers==2.2.2) (1.11.4)
    Requirement already satisfied: nltk in /opt/conda/lib/python3.10/site-packages (from sentence_transformers==2.2.2) (3.2.4)
    Requirement already satisfied: sentencepiece in /opt/conda/lib/python3.10/site-packages (from sentence_transformers==2.2.2) (0.2.0)
    Requirement already satisfied: huggingface-hub>=0.4.0 in /opt/conda/lib/python3.10/site-packages (from sentence_transformers==2.2.2) (0.20.3)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.10/site-packages (from huggingface-hub>=0.4.0->sentence_transformers==2.2.2) (3.13.1)
    Requirement already satisfied: fsspec>=2023.5.0 in /opt/conda/lib/python3.10/site-packages (from huggingface-hub>=0.4.0->sentence_transformers==2.2.2) (2024.2.0)
    Requirement already satisfied: requests in /opt/conda/lib/python3.10/site-packages (from huggingface-hub>=0.4.0->sentence_transformers==2.2.2) (2.31.0)
    Requirement already satisfied: pyyaml>=5.1 in /opt/conda/lib/python3.10/site-packages (from huggingface-hub>=0.4.0->sentence_transformers==2.2.2) (6.0.1)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/conda/lib/python3.10/site-packages (from huggingface-hub>=0.4.0->sentence_transformers==2.2.2) (4.9.0)
    Requirement already satisfied: packaging>=20.9 in /opt/conda/lib/python3.10/site-packages (from huggingface-hub>=0.4.0->sentence_transformers==2.2.2) (21.3)
    Requirement already satisfied: sympy in /opt/conda/lib/python3.10/site-packages (from torch>=1.6.0->sentence_transformers==2.2.2) (1.12)
    Requirement already satisfied: networkx in /opt/conda/lib/python3.10/site-packages (from torch>=1.6.0->sentence_transformers==2.2.2) (3.2.1)
    Requirement already satisfied: jinja2 in /opt/conda/lib/python3.10/site-packages (from torch>=1.6.0->sentence_transformers==2.2.2) (3.1.2)
    Requirement already satisfied: regex!=2019.12.17 in /opt/conda/lib/python3.10/site-packages (from transformers<5.0.0,>=4.6.0->sentence_transformers==2.2.2) (2023.12.25)
    Requirement already satisfied: tokenizers<0.19,>=0.14 in /opt/conda/lib/python3.10/site-packages (from transformers<5.0.0,>=4.6.0->sentence_transformers==2.2.2) (0.15.2)
    Requirement already satisfied: safetensors>=0.4.1 in /opt/conda/lib/python3.10/site-packages (from transformers<5.0.0,>=4.6.0->sentence_transformers==2.2.2) (0.4.2)
    Requirement already satisfied: six in /opt/conda/lib/python3.10/site-packages (from nltk->sentence_transformers==2.2.2) (1.16.0)
    Requirement already satisfied: joblib>=1.1.1 in /opt/conda/lib/python3.10/site-packages (from scikit-learn->sentence_transformers==2.2.2) (1.3.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /opt/conda/lib/python3.10/site-packages (from scikit-learn->sentence_transformers==2.2.2) (3.2.0)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /opt/conda/lib/python3.10/site-packages (from torchvision->sentence_transformers==2.2.2) (9.5.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.10/site-packages (from packaging>=20.9->huggingface-hub>=0.4.0->sentence_transformers==2.2.2) (3.1.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.10/site-packages (from jinja2->torch>=1.6.0->sentence_transformers==2.2.2) (2.1.3)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests->huggingface-hub>=0.4.0->sentence_transformers==2.2.2) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests->huggingface-hub>=0.4.0->sentence_transformers==2.2.2) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests->huggingface-hub>=0.4.0->sentence_transformers==2.2.2) (1.26.18)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests->huggingface-hub>=0.4.0->sentence_transformers==2.2.2) (2024.2.2)
    Requirement already satisfied: mpmath>=0.19 in /opt/conda/lib/python3.10/site-packages (from sympy->torch>=1.6.0->sentence_transformers==2.2.2) (1.3.0)
    Building wheels for collected packages: sentence_transformers
      Building wheel for sentence_transformers (setup.py) ... [?25ldone
    [?25h  Created wheel for sentence_transformers: filename=sentence_transformers-2.2.2-py3-none-any.whl size=125923 sha256=677841498185b6ae82a21d51c000a4d432f31d4eb091fdca26ac3b5d24719519
      Stored in directory: /root/.cache/pip/wheels/62/f2/10/1e606fd5f02395388f74e7462910fe851042f97238cbbd902f
    Successfully built sentence_transformers
    Installing collected packages: sentence_transformers
    Successfully installed sentence_transformers-2.2.2
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    cudf 23.8.0 requires cubinlinker, which is not installed.
    cudf 23.8.0 requires cupy-cuda11x>=12.0.0, which is not installed.
    cudf 23.8.0 requires ptxcompiler, which is not installed.
    cuml 23.8.0 requires cupy-cuda11x>=12.0.0, which is not installed.
    dask-cudf 23.8.0 requires cupy-cuda11x>=12.0.0, which is not installed.
    keras-cv 0.8.2 requires keras-core, which is not installed.
    keras-nlp 0.8.1 requires keras-core, which is not installed.
    tensorflow-decision-forests 1.8.1 requires wurlitzer, which is not installed.
    apache-beam 2.46.0 requires dill<0.3.2,>=0.3.1.1, but you have dill 0.3.8 which is incompatible.
    apache-beam 2.46.0 requires numpy<1.25.0,>=1.14.3, but you have numpy 1.26.4 which is incompatible.
    apache-beam 2.46.0 requires pyarrow<10.0.0,>=3.0.0, but you have pyarrow 11.0.0 which is incompatible.
    cudf 23.8.0 requires cuda-python<12.0a0,>=11.7.1, but you have cuda-python 12.3.0 which is incompatible.
    cudf 23.8.0 requires pandas<1.6.0dev0,>=1.3, but you have pandas 2.1.4 which is incompatible.
    cudf 23.8.0 requires protobuf<5,>=4.21, but you have protobuf 3.20.3 which is incompatible.
    cuml 23.8.0 requires dask==2023.7.1, but you have dask 2024.2.0 which is incompatible.
    dask-cuda 23.8.0 requires dask==2023.7.1, but you have dask 2024.2.0 which is incompatible.
    dask-cuda 23.8.0 requires pandas<1.6.0dev0,>=1.3, but you have pandas 2.1.4 which is incompatible.
    dask-cudf 23.8.0 requires dask==2023.7.1, but you have dask 2024.2.0 which is incompatible.
    dask-cudf 23.8.0 requires pandas<1.6.0dev0,>=1.3, but you have pandas 2.1.4 which is incompatible.
    distributed 2023.7.1 requires dask==2023.7.1, but you have dask 2024.2.0 which is incompatible.
    google-cloud-bigquery 2.34.4 requires packaging<22.0dev,>=14.3, but you have packaging 23.2 which is incompatible.
    jupyterlab 4.1.2 requires jupyter-lsp>=2.0.0, but you have jupyter-lsp 1.5.1 which is incompatible.
    jupyterlab-lsp 5.0.3 requires jupyter-lsp>=2.0.0, but you have jupyter-lsp 1.5.1 which is incompatible.
    libpysal 4.9.2 requires shapely>=2.0.1, but you have shapely 1.8.5.post1 which is incompatible.
    momepy 0.7.0 requires shapely>=2, but you have shapely 1.8.5.post1 which is incompatible.
    osmnx 1.9.1 requires shapely>=2.0, but you have shapely 1.8.5.post1 which is incompatible.
    raft-dask 23.8.0 requires dask==2023.7.1, but you have dask 2024.2.0 which is incompatible.
    spopt 0.6.0 requires shapely>=2.0.1, but you have shapely 1.8.5.post1 which is incompatible.
    tensorflow 2.15.0 requires keras<2.16,>=2.15.0, but you have keras 3.0.5 which is incompatible.
    ydata-profiling 4.6.4 requires numpy<1.26,>=1.16.0, but you have numpy 1.26.4 which is incompatible.[0m[31m
    [0m


```python
import glob
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
import torch
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
import textwrap
```


```python
sorted(glob.glob('/kaggle/input/harry-potter-sorcerers-stone/*'))
```




    ['/kaggle/input/harry-potter-sorcerers-stone/harry-potter-sorcerers-stone.pdf']




```python
# LLMs
model_name = 'llama2-13b-chat' # wizardlm, llama2-7b-chat, llama2-13b-chat, mistral-7B
temperature = 0
top_p = 0.95
repetition_penalty = 1.15    

# splitting
split_chunk_size = 800
split_overlap = 0

# embeddings
embeddings_model_repo = 'sentence-transformers/all-MiniLM-L6-v2'    

# similar passages
k = 6

# paths
PDFs_path = '/kaggle/input/harry-potter-sorcerers-stone/'
Embeddings_path =  './harry-potter-vectordb/faiss_index_hp/'
Output_folder = './harry-potter-vectordb'
```


```python
model_repo = 'daryl149/llama-2-13b-chat-hf'
        
tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True)

#used for Quantization of weights of model (to decrease bits)
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.float16,
    bnb_4bit_use_double_quant = True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_repo,
    quantization_config = bnb_config,       
    device_map = 'auto',
    low_cpu_mem_usage = True,
    trust_remote_code = True
)

max_len = 3000
```


    config.json:   0%|          | 0.00/507 [00:00<?, ?B/s]



    pytorch_model.bin.index.json:   0%|          | 0.00/33.4k [00:00<?, ?B/s]



    Downloading shards:   0%|          | 0/3 [00:00<?, ?it/s]



    pytorch_model-00001-of-00003.bin:   0%|          | 0.00/9.95G [00:00<?, ?B/s]



    pytorch_model-00002-of-00003.bin:   0%|          | 0.00/9.90G [00:00<?, ?B/s]



    pytorch_model-00003-of-00003.bin:   0%|          | 0.00/6.18G [00:00<?, ?B/s]



    Loading checkpoint shards:   0%|          | 0/3 [00:00<?, ?it/s]


    /opt/conda/lib/python3.10/site-packages/torch/_utils.py:831: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
      return self.fget.__get__(instance, owner)()



    generation_config.json:   0%|          | 0.00/137 [00:00<?, ?B/s]



```python
### hugging face pipeline
pipe = pipeline(
    task = "text-generation",
    model = model,
    tokenizer = tokenizer,
    pad_token_id = tokenizer.eos_token_id,
#     do_sample = True,
    max_length = max_len,
    temperature = temperature,
    top_p = top_p,
    repetition_penalty = repetition_penalty
)

### langchain pipeline
llm = HuggingFacePipeline(pipeline = pipe)
```


```python
### testing model, not using the harry potter books yet
query = "What are requirements that students need in HOGWARTS"
llm.invoke(query)
```

    /opt/conda/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:410: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
      warnings.warn(
    /opt/conda/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:415: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.95` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
      warnings.warn(





    "?\n\nIn the Harry Potter series, Hogwarts School of Witchcraft and Wizardry is a magical boarding school where young wizards and witches attend classes to learn various spells, potions, and other magical skills. Here are some of the requirements that students need to have in order to attend Hogwarts:\n\n1. Magic Bloodline: To be eligible for admission to Hogwarts, students must have at least one parent or grandparent who is a human with magical abilities. This ensures that they have inherited magic genetically and can learn and control their powers safely.\n2. Age: Students must be between 11 and 18 years old to attend Hogwarts. First-year students typically start attending the school when they turn 11 years old.\n3. Acceptance Letter: In order to gain entry into Hogwarts, students must receive an acceptance letter from the school. The letter is usually delivered by owls and contains information about the student's schedule, supplies, and other important details.\n4. Magical Abilities: While not all students at Hogwarts possess exceptional magical abilities, having some natural talent or aptitude for magic can be helpful in mastering certain subjects like Charms, Transfiguration, and Potions.\n5. Good Behavior: Students are expected to behave well and follow the rules of the school. Misbehaving or breaking the law can result in disciplinary action, such as detention or expulsion.\n6. Financial Means: Attending Hogwarts can be expensive, so students must have sufficient financial means to cover the costs of tuition, books, and other supplies. Scholarships and financial aid may be available for students who cannot afford these expenses.\n7. Health and Wellness: Students must be physically and mentally healthy enough to handle the rigors of academic life at Hogwarts. Regular check-ups with the school nurse and a balanced diet are essential to maintain good health.\n8. Transportation: Students must arrange transportation to and from Hogwarts. Many students travel on broomsticks or by Floo Network, while others use more conventional methods like cars or trains."




```python

```

We will follow below steps for our chat bot 

* load PDF files
* split into chunks
* create embeddings
* save the embeddings in a vector store
* After that load the saved embeddings to do similarity search with the user query, and then use the LLM to answer the question


```python
loader = DirectoryLoader(
    PDFs_path,
    glob="./*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True
)

documents = loader.load()
```

    100%|██████████| 1/1 [00:28<00:00, 28.49s/it]



```python
print(f'We have {len(documents)} pages in total')
```

    We have 221 pages in total



```python
print(documents[8].page_content)
```

    agree.”
    						He	didn’t	say	another	word	on	the	subject	as	they	went	upstairs	to	bed.
    While	Mrs.	Dursley	was	in	the	bathroom,	Mr.	Dursley	crept	to	the	bedroom
    window	and	peered	down	into	the	front	garden.	The	cat	was	still	there.	It	was
    staring	down	Privet	Drive	as	though	it	were	waiting	for	something.
    						Was	he	imagining	things?	Could	all	this	have	anything	to	do	with	the
    Potters?	If	it	did...if	it	got	out	that	they	were	related	to	a	pair	of	—	well,	he	didn’t
    think	he	could	bear	it.
    						The	Dursleys	got	into	bed.	Mrs.	Dursley	fell	asleep	quickly	but	Mr.
    Dursley	lay	awake,	turning	it	all	over	in	his	mind.	His	last,	comforting	thought
    before	he	fell	asleep	was	that	even	if	the	Potters	were	involved,	there	was	no
    reason	for	them	to	come	near	him	and	Mrs.	Dursley.	The	Potters	knew	very	well
    what	he	and	Petunia	thought	about	them	and	their	kind....He	couldn’t	see	how	he
    and	Petunia	could	get	mixed	up	in	anything	that	might	be	going	on	—	he
    yawned	and	turned	over	—	it	couldn’t	affect	them.…
    						How	very	wrong	he	was.
    						Mr.	Dursley	might	have	been	drifting	into	an	uneasy	sleep,	but	the	cat	on
    the	wall	outside	was	showing	no	sign	of	sleepiness.	It	was	sitting	as	still	as	a
    statue,	its	eyes	fixed	unblinkingly	on	the	far	corner	of	Privet	Drive.	It	didn’t	so
    much	as	quiver	when	a	car	door	slammed	on	the	next	street,	nor	when	two	owls
    swooped	overhead.	In	fact,	it	was	nearly	midnight	before	the	cat	moved	at	all.
    						A	man	appeared	on	the	corner	the	cat	had	been	watching,	appeared	so
    suddenly	and	silently	you’d	have	thought	he’d	just	popped	out	of	the	ground.
    The	cat’s	tail	twitched	and	its	eyes	narrowed.
    						Nothing	like	this	man	had	ever	been	seen	on	Privet	Drive.	He	was	tall,
    thin,	and	very	old,	judging	by	the	silver	of	his	hair	and	beard,	which	were	both
    long	enough	to	tuck	into	his	belt.	He	was	wearing	long	robes,	a	purple	cloak	that
    swept	the	ground,	and	high-heeled,	buckled	boots.	His	blue	eyes	were	light,
    bright,	and	sparkling	behind	half-moon	spectacles	and	his	nose	was	very	long
    and	crooked,	as	though	it	had	been	broken	at	least	twice.	This	man’s	name	was
    Albus	Dumbledore.
    						Albus	Dumbledore	didn’t	seem	to	realize	that	he	had	just	arrived	in	a
    street	where	everything	from	his	name	to	his	boots	was	unwelcome.	He	was
    busy	rummaging	in	his	cloak,	looking	for	something.	But	he	did	seem	to	realize
    he	was	being	watched,	because	he	looked	up	suddenly	at	the	cat,	which	was	still
    staring	at	him	from	the	other	end	of	the	street.	For	some	reason,	the	sight	of	the
    cat	seemed	to	amuse	him.	He	chuckled	and	muttered,	“I	should	have	known.”
    						He	found	what	he	was	looking	for	in	his	inside	pocket.	It	seemed	to	be	a
    silver	cigarette	lighter.	He	flicked	it	open,	held	it	up	in	the	air,	and	clicked	it.	The



```python

```

Splitting the text into chunks so its passages are easily searchable for similarity



```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = split_chunk_size,
    chunk_overlap = split_overlap
)

texts = text_splitter.split_documents(documents)

print(f'We have created {len(texts)} chunks from {len(documents)} pages')
```

    We have created 711 chunks from 221 pages



```python

```

Create embeddings and store the texts in a Vector database (FAISS) for semantic search (which will be provided as context in LLM prompt for Question/Answer


```python
### download embeddings model
embeddings = HuggingFaceInstructEmbeddings(
    model_name = embeddings_model_repo,
    model_kwargs = {"device": "cuda"}
)

### create embeddings and DB
vectordb = FAISS.from_documents(
    documents = texts, 
    embedding = embeddings
)

### persist vector database
vectordb.save_local(f"{Output_folder}/faiss_index_hp")
```

    load INSTRUCTOR_Transformer
    max_seq_length  512



```python

```

After saving the vector database, we just load it 


```python
### load vector DB embeddings
vectordb = FAISS.load_local(
    Embeddings_path, # from input folder
#     CFG.Output_folder + '/faiss_index_hp', # from output folder
    embeddings,
    allow_dangerous_deserialization = True
)
```


```python
### test if vector DB was loaded correctly
for doc in vectordb.similarity_search('Hogwarts School requirements'):
    print(doc.page_content)
```

    Harry	stretched	out	his	hand	at	last	to	take	the	yellowish	envelope,
    addressed	in	emerald	green	to	Mr.	H.	Potter,	The	Floor,	Hut-on-the-Rock,	The
    Sea.	He	pulled	out	the	letter	and	read:
    	
    HOGWARTS	SCHOOL	of	WITCHCRAFT	and	WIZARDRY
    	
    Headmaster:	
    ALBUS	DUMBLEDORE
    (Order	of	Merlin,	First	Class,	Grand	Sorc.,	Chf.	Warlock,	Supreme
    Mugwump,	International	Confed.	of	Wizards)
    	
    Dear	Mr.	Potter,
    We	are	pleased	to	inform	you	that	you	have	been	accepted	at	Hogwarts
    School	of	Witchcraft	and	Wizardry.	Please	find	enclosed	a	list	of	all	necessary
    books	and	equipment.
    Term	begins	on	September	1.	We	await	your	owl	by	no	later	than	July	31.
    Yours	sincerely,
    	
    Minerva	McGonagall,
    Deputy	Headmistress
    	
    Questions	exploded	inside	Harry’s	head	like	fireworks	and	he	couldn’t
    “And	what	are	Slytherin	and	Hufflepuff?”
    						“School	houses.	There’s	four.	Everyone	says	Hufflepuff	are	a	lot	o’
    duffers,	but	—”
    						“I	bet	I’m	in	Hufflepuff,”	said	Harry	gloomily.
    						“Better	Hufflepuff	than	Slytherin,”	said	Hagrid	darkly.	“There’s	not	a
    single	witch	or	wizard	who	went	bad	who	wasn’t	in	Slytherin.	You-Know-Who
    was	one.”
    						“Vol-,	sorry	—You-Know-Who	was	at	Hogwarts?”
    						“Years	an’	years	ago,”	said	Hagrid.
    						They	bought	Harry’s	school	books	in	a	shop	called	Flourish	and	Blotts
    where	the	shelves	were	stacked	to	the	ceiling	with	books	as	large	as	paving
    stones	bound	in	leather;	books	the	size	of	postage	stamps	in	covers	of	silk;	books
    full	of	peculiar	symbols	and	a	few	books	with	nothing	in	them	at	all.	Even
    First-year	students	will	require:
    1.	Three	sets	of	plain	work	robes	(black)
    2.	One	plain	pointed	hat	(black)	for	day	wear
    3.	One	pair	of	protective	gloves	(dragon	hide	or	similar)
    4.	One	winter	cloak	(black,	silver	fastenings)
    Please	note	that	all	pupils’	clothes	should	carry	name	tags
    	
    COURSE	BOOKS
    All	students	should	have	a	copy	of	each	of	the	following:
    The	Standard	Book	of	Spells	(Grade	1)	
    by	Miranda	Goshawk
    A	History	of	Magic	
    by	Bathilda	Bagshot
    Magical	Theory	
    by	Adalbert	Waffling
    A	Beginners’	Guide	to	Transfiguration	
    by	Emeric	Switch
    One	Thousand	Magical	Herbs	and	Fungi	
    by	Phyllida	Spore
    Magical	Drafts	and	Potions	
    by	Arsenius	Jigger
    Fantastic	Beasts	and	Where	to	Find	Them	
    by	Newt	Scamander
    The	Dark	Forces:	A	Guide	to	Self-Protection	
    by	Quentin	Trimble
    	
    OTHER	EQUIPMENT
    HP	1	-	Harry	Potter	and	the
    Sorcerer's	Stone



```python

```

# Prompt


```python
prompt_template = """
Don't try to make up an answer, if you don't know just say that you don't know.
Answer in the same language the question was asked.
Use only the following pieces of context to answer the question at the end.

{context}

Question: {question}
Answer:"""


PROMPT = PromptTemplate(
    template = prompt_template, 
    input_variables = ["context", "question"]
)
```


```python

```

Retriever to retrieve relevant passages
Chain to answer questions


```python
retriever = vectordb.as_retriever(search_kwargs = {"k": k, "search_type" : "similarity"})

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever, 
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = False
)
```


```python

```

* Format llm response
* Cite sources (PDFs)


```python
def wrap_text_preserve_newlines(text, width=700):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    ans = wrap_text_preserve_newlines(llm_response['result'])
    
    sources_used = ' \n'.join(
        [
            source.metadata['source'].split('/')[-1][:-4]
            + ' - page: '
            + str(source.metadata['page'])
            for source in llm_response['source_documents']
        ]
    )
    
    ans = ans + '\n\nSources: \n' + sources_used
    return ans
```


```python
query = "What things student needs for visiting to Hogwarts explain in depth?"
llm_response = qa_chain.invoke(query)
ans = process_llm_response(llm_response)
print(ans)
```

    /opt/conda/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:410: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
      warnings.warn(
    /opt/conda/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:415: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.95` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
      warnings.warn(


     According to the passage, students need three sets of plain work robes (black), one pointed hat (black), and one winter cloak (black with silver fastenings). Additionally, all pupils' clothes should carry name tags. The passage also mentions that students will need course books such as "The Standard Book of Spells," "A History of Magic," "Magical Theory," "A Beginner's Guide to Transfiguration," "One Thousand Magical Herbs and Fungi," and "Magical Drafts and Potions." Other equipment includes a wand, parchment, and quills.
    
    Sources: 
    harry-potter-sorcerers-stone - page: 50 
    harry-potter-sorcerers-stone - page: 76 
    harry-potter-sorcerers-stone - page: 59 
    harry-potter-sorcerers-stone - page: 192 
    harry-potter-sorcerers-stone - page: 11 
    harry-potter-sorcerers-stone - page: 177



```python
### using pure llm response without harry potter book reference
print(llm.invoke(query))
```

    /opt/conda/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:410: UserWarning: `do_sample` is set to `False`. However, `temperature` is set to `0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.
      warnings.warn(
    /opt/conda/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:415: UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0.95` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
      warnings.warn(


    
    
    I'm planning a Harry Potter themed party and I want to make sure that my guests have everything they need to feel like they are truly attending Hogwarts. Here are some items that I think would be helpful for students to bring with them:
    
    1. Acceptance letter and invitation: This is the most important item, as it serves as proof of their acceptance to Hogwarts and grants them entry into the school. The letter should include the student's name, address, and date of birth, as well as information about the school and its expectations.
    2. Wand: Every student at Hogwarts is required to have a wand, which is used for casting spells and performing magic. The wand should be made of wood or another magical material, and should have a unique design that reflects the student's personality or interests.
    3. School robes: Students at Hogwarts are expected to wear traditional school robes, which consist of a long black gown with a white collar and cuffs. The robes should be comfortable and durable, and should be worn at all times during classes and other school activities.
    4. Pointy hat: First-year students at Hogwarts are required to wear pointy hats, which are designed to help them navigate the castle and find their way to their classes. The hats should be made of sturdy materials and should be worn at all times during the first year.
    5. Spellbook: Students at Hogwarts are expected to learn a wide range of spells and incantations, which are recorded in a special spellbook. The spellbook should contain detailed instructions on how to cast each spell, as well as any relevant notes or illustrations.
    6. Magic potions: Students at Hogwarts may be required to brew and drink magic potions as part of their coursework. To do this, they will need a variety of ingredients, such as eye of newt and wing of bat, as well as a cauldron and other equipment.
    7. Broomstick: Many students at Hogwarts enjoy flying on broomsticks, either for recreation or as part of their Quidditch team. A high-quality broomstick can be expensive, but it is an essential item for any student who wants to fly.
    8. Pensieve: Some students at Hogwarts may be assigned to use a pensieve, which is a device that allows them to view and analyze memories. The pensieve should be kept clean and free of dust, and should only be used by authorized personnel.
    9. Wizarding currency: While many transactions at Hogwarts are conducted using Galleons, Sickles, and Knuts (the standard wizarding currencies), some students may need to purchase items from the school's shop or pay fines for breaking rules. It is therefore advisable for students to carry some amount of wizarding currency with them at all times.
    10. Personal belongings: Finally, students at Hogwarts should bring any personal belongings that they may need during their time at the school, such as photographs of family members, favorite books or toys, or other mementos that provide comfort and support.


Above we can observe using Harry potter books response is much more specific to exact details along with page sources

LLM response is more general


```python

```
