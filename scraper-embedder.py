import time
import advertools as adv
import pandas as pd
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from tqdm.auto import tqdm
import tiktoken
import openai
import pinecone
import toml

# Load the secrets.toml file
secrets = toml.load(".streamlit/secrets.toml")

# Retrieve the values
PINECONE_API_KEY = secrets["API"]["PINECONE_API_KEY"]
OPEN_AI_API_KEY = secrets["API"]["OPEN_AI_API_KEY"]


sitemaps = [
    "https://www.athleticinsight.com/post-sitemap1.xml",
    "https://www.athleticinsight.com/post-sitemap2.xml",
    "https://www.athleticinsight.com/post-sitemap3.xml",
    "https://www.athleticinsight.com/page-sitemap.xml",
    "https://www.athleticinsight.com/local-sitemap.xml",
]


def scrape_sitemaps(sitemaps):
    """Scrape a list of sitemaps and return a dataframe with the results"""
    df = pd.concat([adv.sitemap_to_df(sitemap) for sitemap in sitemaps])
    df.to_csv("sitemap.csv", index=False)

    return df


def convert_df_to_set(df):
    # Get list of all locations
    location_list = df["loc"].tolist()

    # Remove duplicates and return result
    return list(set(location_list))


def scrape_pages(url_list):
    adv.crawl(url_list, output_file="page_data.jl", follow_links=False)
    # convert jl to csv
    df = pd.read_json("page_data.jl", lines=True)
    df = df[["url", "title", "body_text"]]
    df.to_json("page_data.json", orient="records", lines=True)
    df.to_csv("page_data.csv", index=False)
    return df


def convert_df_to_dict(df):
    df = pd.read_json(
        "page_data.json", lines=True, orient="records", encoding="utf-8", dtype=str
    )
    df = df.to_dict(orient="records")
    return df


# ## Step 1
# df = scrape_sitemaps(sitemaps)
# urls = convert_df_to_set(df)
# scrape_pages(urls)


# create the length function
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("p50k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def create_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for idx, record in enumerate(tqdm(data)):
        texts = text_splitter.split_text(record["body_text"])
        chunks.extend(
            [
                {
                    "id": str(uuid4()),
                    "title": record["title"],
                    "text": texts[i],
                    "chunk": i,
                    "url": record["url"],
                }
                for i in range(len(texts))
            ]
        )
    return chunks


def init_pinecone(api_key, environment):
    """
    Initializes a connection to the Pinecone service.

    Args:
        api_key (str): The API key for the Pinecone service.
        environment (str): The environment name for the Pinecone service.

    Returns:
        None
    """
    pinecone.init(api_key=api_key, environment=environment)


def create_index_if_not_exists(index_name, dimension, metric):
    # Check if the index already exists
    if index_name not in pinecone.list_indexes():
        # If the index does not exist, create a new index
        pinecone.create_index(index_name, dimension=dimension, metric=metric)
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")


def create_embeddings(chunks, embed_model, index, batch_size=100):
    """
    Create embeddings for text chunks and insert them into Pinecone index.

    Args:
        chunks (list): List of text chunks to be embedded.
        embed_model (str): Name of the OpenAI GPT model to use for embedding.
        index (pinecone.Index): Pinecone index object to insert embeddings into.
        batch_size (int): Number of embeddings to create and insert at once.

    Returns:
        None
    """
    # Set up OpenAI API key
    openai.api_key = OPEN_AI_API_KEY

    for i in tqdm(range(0, len(chunks), batch_size)):
        # find end of batch
        i_end = min(len(chunks), i + batch_size)
        meta_batch = chunks[i:i_end]
        # get ids
        ids_batch = [x["id"] for x in meta_batch]
        # get texts to encode
        texts = [x["text"] for x in meta_batch]
        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = openai.Embedding.create(input=texts, engine=embed_model)
        except:
            done = False
            while not done:
                time.sleep(5)
                try:
                    res = openai.Embedding.create(input=texts, engine=embed_model)
                    done = True
                except:
                    pass
        embeds = [record["embedding"] for record in res["data"]]
        # cleanup metadata
        meta_batch = [
            {
                "title": x["title"],
                "text": x["text"],
                "chunk": x["chunk"],
                "url": x["url"],
            }
            for x in meta_batch
        ]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        index.upsert(vectors=to_upsert)


## STEP 2 Embedding and indexing


data = pd.read_json(
    "page_data.json", lines=True, orient="records", encoding="utf-8", dtype=str
)
# convert to list of dicts
data = data.to_dict(orient="records")

chunks = create_chunks(data)


init_pinecone(api_key=PINECONE_API_KEY, environment="us-west1-gcp-free")

create_index_if_not_exists(
    index_name="athleticinsight-docs", dimension=1536, metric="dotproduct"
)

index = pinecone.GRPCIndex("athleticinsight-docs")


create_embeddings(
    chunks=chunks,
    embed_model="text-embedding-ada-002",
    index=index,
    batch_size=100,
)
