import arxiv
import os
#%%
# class names of arxiv
# https://gist.github.com/jozefg/c2542f51a0b9b3f6efe528fcec90e334
CS_CLASSES = [
    'cs.' + cat for cat in [
        'AI', 'AR', 'CC', 'CE', 'CG', 'CL', 'CR', 'CV', 'CY', 'DB',
        'DC', 'DL', 'DM', 'DS', 'ET', 'FL', 'GL', 'GR', 'GT', 'HC',
        'IR', 'IT', 'LG', 'LO', 'MA', 'MM', 'MS', 'NA', 'NE', 'NI',
        'OH', 'OS', 'PF', 'PL', 'RO', 'SC', 'SD', 'SE', 'SI', 'SY',
    ]
]

MATH_CLASSES = [
    'math.' + cat for cat in [
        'AC', 'AG', 'AP', 'AT', 'CA', 'CO', 'CT', 'CV', 'DG', 'DS',
        'FA', 'GM', 'GN', 'GR', 'GT', 'HO', 'IT', 'KT', 'LO',
        'MG', 'MP', 'NA', 'NT', 'OA', 'OC', 'PR', 'QA', 'RA',
        'RT', 'SG', 'SP', 'ST', 'math-ph'
    ]
]

QBIO_CLASSES = [
    'q-bio.' + cat for cat in [
        'BM', 'CB', 'GN', 'MN', 'NC', 'OT', 'PE', 'QM', 'SC', 'TO'
    ]
]

# Which categories do we search
CLASSES = CS_CLASSES + MATH_CLASSES + QBIO_CLASSES


# Define the search query
search_query = "cat:cs.* AND all:diffusion OR all:score-based"  # You can change this to a specific field or topic

# Fetch papers
search = arxiv.Search(
    query=search_query,
    max_results=30000,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)
#%%
# Print titles and abstracts of the latest papers
paper_collection = []
paper_id = 0
for paper in arxiv.Client(delay_seconds=5.0, num_retries=50).results(search):
    paper_collection.append(paper)
    id_pure = paper.entry_id.strip("http://arxiv.org/abs/")
    print(f"{paper_id} [{id_pure}] ({paper.published.date()})",
          paper.title)
    paper_id += 1
    # print("Abstract:", paper.summary)
    # print("Categories:", paper.categories, end=" ")
    # print("ID:", paper.entry_id, end=" ")
    # print("-" * 80)
#%%
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd
pkl.dump(paper_collection, open("arxiv_collection.pkl", "wb"))
df = pd.DataFrame(paper_collection)
df.to_csv("arxiv_collection.csv")
#%%
paper_collection = pkl.load(open("arxiv_collection.pkl", "rb"))

#%%
# plot the distribution of time
import seaborn as sns
time_col = [paper.published for paper in paper_collection]
plt.hist(time_col, bins=50)
plt.title("Distribution of publication time")
plt.ylabel("count")
plt.show()
#%%
def entry2string(paper):
    id_pure = paper.entry_id.strip("http://arxiv.org/abs/")
    return f"[{id_pure}] Title: {paper.title}\nAbstract: {paper.summary}\nDate: {paper.published}"

embedstr_col = [entry2string(paper) for paper in paper_collection]
#%%
# embed all the abstracts
import openai
client = openai.OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
#%%
from tqdm import tqdm
batch_size = 20
embedding_col = []
for i in tqdm(range(0, len(embedstr_col), batch_size)):
    embedstr_batch = embedstr_col[i:i+batch_size]
    response = client.embeddings.create(
        input=embedstr_batch,
        model="text-embedding-ada-002"
    )
    embedding_col.extend(response.data)
    # print(response.data[0].embedding)
    # raise ValueError("This is a test")

#%%
# save the embeddings
import numpy as np
import pickle as pkl
pkl.dump(embedding_col, open("arxiv_embedding.pkl", "wb"))
# format as array
embedding_arr = np.stack([embed.embedding for embed in embedding_col])
pkl.dump([embedding_arr, paper_collection], open("arxiv_embedding_arr.pkl", "wb"))



#%%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0, metric="cosine")
embedding_tsne = tsne.fit_transform(embedding_arr)
#%%
# cluster the embeddings
# from sklearn.cluster import _agglomerative
from sklearn.cluster import AgglomerativeClustering, KMeans
# cluster = AgglomerativeClustering(n_clusters=20, affinity="cosine", linkage="average")
# cluster.fit(embedding_arr)
cluster = KMeans(n_clusters=20, random_state=0)
cluster.fit(embedding_arr)
#%%
# find nearest neighbor of papers
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=15, metric="cosine")
nn.fit(embedding_arr)
#%%#%%
# plot the tsne
# sns.set_context("talk")
plt.figure(figsize=(8, 8))
plt.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], s=20,
            c=cluster.labels_, alpha=0.5, cmap="tab20")
plt.title("TSNE of arxiv abstracts")
plt.show()
#%%
# show the representative papers from each cluster
for i in range(20):
    idx = np.where(cluster.labels_ == i)[0][0]
    paper = paper_collection[idx]
    print(f"[Cluster {i}]")
    author_names = [author.name for author in paper.authors]
    print(f"({paper.published.date()})", paper.title, author_names)
    dists, idxs = nn.kneighbors(embedding_arr[idx:idx+1, :])
    for dist, idx in zip(dists[0][1:10], idxs[0][1:10]):
        paper = paper_collection[idx]
        author_names = [author.name for author in paper.authors]
        print(f"\t- Cos:{dist:.3f} ({paper.published.date()})", paper.title, author_names)
    print("\n")
#%%
query_idx = 100
dists, idxs = nn.kneighbors(embedding_arr[query_idx:query_idx+1, :])
for dist, idx in zip(dists[0], idxs[0]):
    paper = paper_collection[idx]
    author_names = [author.name for author in paper.authors]
    print(f"Cos:{dist:.3f} ({paper.published.date()})", paper.title, author_names)

#%%



#%%
# search for latest papers
search = arxiv.Search(
    query="",
    max_results=500,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)
#%%
def fetch_K_results(search_obj, K=10, offset=0):
    """Fetches K results from the search object, starting from offset, and returns a list of results."""
    results = []
    try:
        for entry in search_obj.results(offset=offset):
            results.append(entry)
            if len(results) >= K:
                break
    except StopIteration:
        pass
    return results

#%%
# search for latest papers
search = arxiv.Search(
    query="au:Yann LeCun",
    max_results=20,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)
results = fetch_K_results(search, K=20, offset=0)
#%%
print(len(results))
#%%
import arxiv
search = arxiv.Search(
  query="au:del_maestro AND ti:checkerboard",
  max_results = 10,
  sort_by=arxiv.SortCriterion.SubmittedDate
)

for result in arxiv.Client().results(search):
  print(result.title)
#%%
import arxiv

# Construct the default API client.
client = arxiv.Client()

# Search for the 10 most recent articles matching the keyword "quantum."
search = arxiv.Search(
  query = "quantum",
  max_results = 10,
  sort_by = arxiv.SortCriterion.SubmittedDate
)

results = client.results(search)

# `results` is a generator; you can iterate over its elements one by one...
for r in client.results(search):
  print(r.title)
# ...or exhaust it into a list. Careful: this is slow for large results sets.
all_results = list(results)
print([r.title for r in all_results])

# For advanced query syntax documentation, see the arXiv API User Manual:
# https://arxiv.org/help/api/user-manual#query_details
search = arxiv.Search(query = "au:del_maestro AND ti:checkerboard")
first_result = next(client.results(search))
print(first_result)

# Search for the paper with ID "1605.08386v1"
search_by_id = arxiv.Search(id_list=["1605.08386v1"])
# Reuse client to fetch the paper, then print its title.
first_result = next(client.results(search))
print(first_result.title)
  #%%
import arxiv

client = arxiv.Client()
search = arxiv.Search(id_list=["1605.08386v1"])

paper = next(arxiv.Client().results(search))
print(paper.title)


#%%
from pyarxiv import query, download_entries
from pyarxiv.arxiv_categories import ArxivCategory, arxiv_category_map
#query(max_results=100, ids=[], categories=[],
#                title='', authors='', abstract='', journal_ref='',
#                querystring='')
entries = query(title='WaveNet')
titles = map(lambda x: x['title'], entries)
#%%
print(list(titles), sep='\n')


#download_entries(entries_or_ids_or_uris=[], target_folder='.',
#                     use_title_for_filename=False, append_id=False,
#                     progress_callback=(lambda x, y: id))
download_entries(entries)
#%%

entries_with_category = query(categories=[ArxivCategory.cs_AI], max_results=10)
# print(arxiv_category_map(ArxivCategory.cs_AI))
print(entries_with_category)
for entry in entries_with_category:
    print(entry['title'])


#%%
