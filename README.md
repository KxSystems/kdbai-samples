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
- [Langchain](https://github.com/KxSystems/langchain/blob/KDB.AI/docs/docs/integrations/vectorstores/kdbai.ipynb): Example showing a question and answer session using a Langchain integration with the KDB.AI Vector Database.


## Setup

This section details the setup steps required to run these samples locally on your machine.

### Prerequisites

This setup guide assumes the following:
  1. You are using a Unix terminal or similar
  1. You have `python` >= 3.8 installed
  1. You have `pip` installed

### Install Python Packages

1. Use the `requirements.txt` file in the repository to install the relevent Python packages needed to run all of the samples:

    ```bash
    pip install -r requirements.txt
    ```

### Install English Sentence Tokeniser

1. Open a Python interpreter:

    ```bash
    python3
    ```

1. Install the `punkt` data resouce for the `nltk` Python package:

    ```python
    import nltk
    nltk.download("punkt")
    ```

1. Exit the Python interpreter:

    ```python
    exit()
    ```

### View & Execute The Samples

1. Run a jupter notebook session:

    ```bash
    jupyter notebook --no-browser
    ```

    This will load up the jupyter session in the background and display a URL on screen for you.

1. Paste this URL into your browser

    This will bring up the samples for you to interact with.


## Dataset Disclaimer

In this repository, we may make available to you certain datasets for use with the Software.
You are not obliged to use such datasets (with the Software or otherwise), but any such use is at your own risk.
Any datasets that we may make available to you are provided “as is” and without any warranty, including as to their accuracy or completeness.
We accept no liability for any use you may make of such datasets.
