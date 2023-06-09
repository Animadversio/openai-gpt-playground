import os
from notion_client import Client
from notion_tools import print_entries
import arxiv
import questionary
import textwrap
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory, FileHistory
database_id = "d3e3be7fc96a45de8e7d3a78298f9ccd"
notion = Client(auth=os.environ["NOTION_TOKEN"])

history = FileHistory("notion_arxiv_history.txt")
session = PromptSession(history=history)

def arxiv_entry2page_blocks(paper: arxiv.arxiv.Result):
    title = paper.title
    authors = [author.name for author in paper.authors]
    pubyear = paper.published
    abstract = paper.summary
    arxiv_id = paper.entry_id.split("/")[-1]
    abs_url = paper.entry_id
    page_prop = {
        'Name': {
            "title": [
                {
                    "text": {
                        "content": f"[{arxiv_id}] {title}"
                    }
                }],
        },
        "Author": {
            "multi_select": [
                {'name': name} for name in authors
            ]
        },
        'Publishing/Release Date': {
            'date': {'start': pubyear.date().isoformat(), }
        },
        'Link': {
            'url': abs_url
        }
    }
    content_block = [{'quote': {"rich_text": [{"text": {"content": abstract}}]}},
                     {'heading_2': {"rich_text": [{"text": {"content": "Related Work"}}]}},
                     {'paragraph': {"rich_text": [{"text": {"content": ""}}]}},
                     {'heading_2': {"rich_text": [{"text": {"content": "Techniques"}}]}},
                     {'paragraph': {"rich_text": [{"text": {"content": ""}}]}},
                     ]
    return page_prop, content_block


def arxiv_entry2page(database_id, paper: arxiv.arxiv.Result):
    page_prop, content_block = arxiv_entry2page_blocks(paper)
    new_page = notion.pages.create(parent={"database_id": database_id}, properties=page_prop)
    notion.blocks.children.append(new_page["id"], children=content_block)
    return new_page["id"], new_page


def print_arxiv_entry(paper: arxiv.arxiv.Result):
    title = paper.title
    authors = [author.name for author in paper.authors]
    pubyear = paper.published
    abstract = paper.summary
    arxiv_id = paper.entry_id.split("/")[-1]
    abs_url = paper.entry_id
    print(f"[{arxiv_id}] {title}")
    print("Authors:", ", ".join(authors))
    print("Published:", pubyear.date().isoformat())
    print("Abstract:")
    print(textwrap.fill(abstract, width=100))
    print("URL:", abs_url)


def blocks2text(blocks):
    if "results" in blocks:
        blocks = blocks["results"]
    for block in blocks:
        if block["type"] == "paragraph":
            for parts in block["paragraph"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))

        elif block["type"] == "heading_2":
            for parts in block["heading_2"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))

        elif block["type"] == "quote":
            for parts in block["quote"]["rich_text"]:
                print(textwrap.fill(parts["plain_text"], width=100))
        else:
            print(block["type"])


# query = "2106.05963"
while True:
    # query = questionary.text("Enter arXiv ID:").ask()
    query = session.prompt("Enter arXiv ID: ", multiline=False)
    # check if entry already exists in Notion database
    results_notion = notion.databases.query(database_id=database_id,
                          filter={"property": "Link", "url": {"contains": query}})
    if len(results_notion["results"]) == 0:
        # if entry exists, ask if user wants to update it
        results_arxiv = list(arxiv.Search(query).results())
        if len(results_arxiv) > 1:
            print("Multiple results found. Please select one:")
            for i, paper in enumerate(results_arxiv):
                print(f"{i+1}: {paper.title}")
            selection = questionary.select("Select paper:", choices=[str(i+1) for i in range(len(results_arxiv))]).ask()
            paper = results_arxiv[int(selection)-1]
        elif len(results_arxiv) == 1:
            paper = results_arxiv[0]
        else:
            print("No results found.")
            continue

        title = paper.title
        authors = [author.name for author in paper.authors]
        pubyear = paper.published
        abstract = paper.summary  # arxiv.download(paper['id'], prefer_source_tex=False).summary
        arxiv_id = paper.entry_id.split("/")[-1]
        abs_url = paper.entry_id
        print_arxiv_entry(paper)
        print(f"Adding entry paper {arxiv_id}: {title}")
        page_id, page = arxiv_entry2page(database_id, paper)
        print(f"Added entry {page_id} for arxiv paper {arxiv_id}: {title}")
        # paper.download_source()
        print_entries([page], print_prop=("url", ))
    else:
        print_entries(results_notion, print_prop=("url", ))
        print("Entry already exists as above. Exiting.")
        for page in results_notion["results"]:
            print_entries([page], print_prop=("url", ))
            try:
                blocks = notion.blocks.children.list(page["id"])
                blocks2text(blocks)
            except Exception as e:
                print(e)
        continue
#%%


# blocks2text(blocks)
#%% Scratch zone
# page_prop = {
#     'Name': {
#         "title": [
#             {
#                 "text": {
#                     "content": f"[{arxiv_id}] {title}"
#                 }
#             }],
#     },
#     "Author": {
#                 "multi_select": [
#                     {'name': name} for name in authors
#                 ]
#             },
#     'Publishing/Release Date': {
#                'date': {'start': pubyear.date().isoformat(), }
#     },
#     'Link': {
#                'url': abs_url
#     }
# }
# content_block = [{'quote': {"rich_text": [{"text": {"content": abstract}}]}},
#                  {'heading_2': {"rich_text": [{"text": {"content": "Related Work"}}]}},
#                  {'paragraph': {"rich_text": [{"text": {"content": ""}}]}},
#                  {'heading_2': {"rich_text": [{"text": {"content": "Techniques"}}]}},
#                  {'paragraph': {"rich_text": [{"text": {"content": ""}}]}},
#                  ]
#
# new_page = notion.pages.create(parent={"database_id": database_id}, properties=page_prop)
# notion.blocks.children.append(new_page["id"], children=content_block)
# #%%
# results_notion = notion.databases.query(database_id, filter={"property": "Link", "url": {"is_not_empty": True}})
#%%
# %%
# # unzip .tar.gz file
# import tarfile
# import os
# import shutil
# tarfile.open('./2106.05963v3.Learning_to_See_by_Looking_at_Noise.tar.gz').extractall()

#%%
notion.blocks.children.list('767b3f0b-f4f6-4f55-81c6-3a2c7b8a8b23')
