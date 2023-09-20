![KDB.AI Logo](https://kdb.ai/images/logo-kdb.ai-black-bg.png)

The example [KDB.AI](https://kdb.ai) samples provided aim to demonstrate examples of the use of the KDB.AI vector database in a number of scenarios ranging from getting started guides to industry specific use-cases.

In order to execute these notebooks, you will need to sign up for access to the [KDB.AI Cloud Portal](https://trykdb.kx.com/kdbai/signup/). Here, you will be given a Entrypoint URL and an API Key which can be used to connect to a KDB.AI Cloud session and execute the code against.


## What is KDB.AI?

KDB.AI is a time-based vector database that allows developers to build scalable, reliable, and real-time applications by providing advanced search, recommendation, and personalization for Generative AI applications. KDB.AI is a key component of full-stack Generative AI applications that use Retrieval Augmented Generation (RAG).

Built by KX, the creators of kdb+, KDB.AI provides users with the ability to combine unstructured vector embedding data with structured time-series datasets to allow for hybrid use-cases which benefit from the rigor of conventional time-series data analytics and the usage patterns provided by vector databases within the Generative AI space.


## What does KDB.AI support?

KDB.AI supports the following feature set:

- Multiple index types: Flat, IVF, IVFPQ and HNSW.
- Multiple distance metrics: Euclidean, Inner-Product, Cosine.
- Top-N and metadata filtered retrieval
- Python and REST Interfaces


## Sample Breakdown

At this time, the repository contains the following samples:

### Getting Started

- [Python Quickstart](quickstarts/python_quickstart.ipynb): A quick introduction to the KDB.AI APIs in Python.

### Use-Cases

- [Document Search](document_search): Semantic Search on PDF Documents.
- [Image Search](image_search): Image Search on Brain MRI Scans.
- [Recommendation System](music_recommendation): Music Recommendation on Spotify Data.
- [Pattern Matching](pattern_matching): Pattern Matching on Sensor Data.
- [Retreival Augmented Generation with LangChain](retrieval_augmented_generation): Retrieval Augmented Generation (RAG) with LangChain.
- [Sentiment Analysis](sentiment_analysis): Sentiment Analysis on Disneyland Resort Reviews.


## What Platforms Does KDB.AI Integrate With?

- [ChatGPT Retrieval Plugin](https://github.com/KxSystems/chatgpt-retrieval-plugin/blob/KDB.AI/examples/providers/kdbai/ChatGPT_QA_Demo.ipynb): Example showing a question and answer session using a ChatGPT retrieval plugin using KDB.AI Vector Database.
- [Langchain](https://github.com/KxSystems/langchain/blob/KDB.AI/docs/extras/integrations/vectorstores/kdbai.ipynb): Example showing a question and answer session using a Langchain integration with the KDB.AI Vector Database.


## Setup

To install the relevent packages needed to run all of the samples contained in this repository, run the following:
```pip install -r requirements.txt```
