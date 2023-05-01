
import os
from os.path import join
import textwrap
import pickle as pkl
import urllib.request
import requests
from langchain.document_loaders import PDFMinerLoader, PyPDFLoader, BSHTMLLoader, NotionDBLoader, UnstructuredURLLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings  # for creating embeddings
from langchain.vectorstores import Chroma  # for the vectorization part
from langchain.chains import ChatVectorDBChain  # for chatting with the pdf
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import OpenAI  # the LLM model we'll use (CHatGPT)
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
import questionary
from notion_client import Client
import arxiv
from notion_tools import QA_notion_blocks, clean_metadata, print_entries, save_qa_history

if os.environ["COMPUTERNAME"] == 'PONCELAB-OFF6':
    embed_rootdir = r"D:\DL_Projects\NLP\Embed_data"
    pdf_download_root = r"D:\DL_Projects\NLP\arxiv_pdf"
elif os.environ["COMPUTERNAME"] == 'DESKTOP-MENSD6S':
    embed_rootdir = r"E:\DL_Projects\NLP\Embed_data"
    pdf_download_root = r"E:\DL_Projects\NLP\arxiv_pdf"
else:
    # get temp dir
    embed_rootdir = os.path.join(os.environ["TMP"], "Embed_data")
    pdf_download_root = os.path.join(os.environ["TMP"], "arxiv_pdf")
    os.makedirs(embed_rootdir, exist_ok=True)
    os.makedirs(pdf_download_root, exist_ok=True)
    print(f"Embedding data will be saved to {embed_rootdir}")
    print(f"PDFs will be saved to {pdf_download_root}")
#%%
database_id = "d3e3be7fc96a45de8e7d3a78298f9ccd"
notion = Client(auth=os.environ["NOTION_TOKEN"])


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
    arxiv_id = questionary.text("Enter arXiv ID:").ask()
    # Search it on arxiv
    results_arxiv = list(arxiv.Search(arxiv_id).results())
    if len(results_arxiv) > 1:
        print("Multiple results found. Please select one:")
        for i, paper in enumerate(results_arxiv):
            print(f"{i + 1}: {paper.title}")
        selection = questionary.select("Select paper:", choices=[str(i + 1) for i in range(len(results_arxiv))]).ask()
        paper = results_arxiv[int(selection) - 1]
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

    # check if entry already exists in Notion database
    results_notion = notion.databases.query(database_id=database_id,
                          filter={"property": "Link", "url": {"contains": arxiv_id}})
    if len(results_notion["results"]) == 0:
        # if entry doesn't exist, add it to the database
        print(f"Adding entry paper {arxiv_id}: {title}")
        page_id, page = arxiv_entry2page(database_id, paper)
        print(f"Added entry {page_id} for arxiv paper {arxiv_id}: {title}")
        print_entries([page], print_prop=("url", ))
        # paper.download_source()
    elif len(results_notion["results"]) == 1:
        page_id, page = results_notion["results"][0]["id"], results_notion["results"][0]
    else:
        # if entry exists, use it instead
        print_entries(results_notion, ) # print_prop=("url", )
        print("Entry already exists as above. Exiting.")
        for page in results_notion["results"]:
            print_entries([page], print_prop=("url",))
            try:
                blocks = notion.blocks.children.list(page["id"])
                blocks2text(blocks)
            except Exception as e:
                print(e)
        page_id = questionary.select("Select paper:",
                                     choices=[page["id"] for page in results_notion["results"]]).ask()
        page = notion.pages.retrieve(page_id)
    #%%
    save_page_id = page_id
    embed_persist_dir = join(embed_rootdir, arxiv_id)
    # doctype = questionary.select("Select download document type:", default="ar5iv",
    #                                 choices=["ar5iv", "arxiv"]).ask() # , "pdf", "pdf+arxiv", "pdf+ar5iv"
    ar5iv_url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    r = requests.get(ar5iv_url, allow_redirects=True, )
    if r.url.startswith("https://ar5iv.labs.arxiv.org/html"):
        open(join(pdf_download_root, f"{arxiv_id}.html"), 'wb').write(r.content)
        loader = BSHTMLLoader(join(pdf_download_root, f"{arxiv_id}.html"),
                              open_encoding="utf8", bs_kwargs={"features": "html.parser"})
        pages = loader.load_and_split()
    else:
        print(f"redirected to {r.url}")
        print("ar5iv not found, downloading pdf instead ")
        r = requests.get(pdf_url, allow_redirects=True, )
        open(join(pdf_download_root, f"{arxiv_id}.pdf"), 'wb').write(r.content)
        loader = PyPDFLoader(join(pdf_download_root, f"{arxiv_id}.pdf"))
        # loader = PDFMinerLoader(pdf_path)
        pages = loader.load_and_split()

    embeddings = OpenAIEmbeddings()
    if os.path.exists(embed_persist_dir):
        print("Loading embeddings from", embed_persist_dir)
        vectordb = Chroma(persist_directory=embed_persist_dir, embedding_function=embeddings)
    else:
        print("Creating embeddings and saving to", embed_persist_dir)

        vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                         persist_directory=embed_persist_dir, )
        vectordb.persist()

    chat_temperature = questionary.text("Sampling temperature for ChatGPT?", default="0.3").ask()
    chat_temperature = float(chat_temperature)
    # ref_maxlen = questionary.text("Max length of reference document?", default="300").ask()
    ref_maxlen = 200 #int(ref_maxlen)
    pdf_qa_new = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=chat_temperature, model_name="gpt-3.5-turbo"),
                                        vectordb.as_retriever(), return_source_documents=True, max_tokens_limit=4000)

    qa_path = embed_persist_dir + "_qa_history"
    os.makedirs(qa_path, exist_ok=True)
    with get_openai_callback() as cb:
        while True:
            # query = "For robotics purpose, which algorithm did they used, PPO, Q-learning, etc.?"
            query = questionary.text("Question: ", multiline=True).ask()
            if query == "" or query is None:
                if questionary.confirm("Exit?").ask():
                    break
                else:
                    continue

            result = pdf_qa_new({"question": query, "chat_history": ""})

            answer = result["answer"]
            refdocs = result['source_documents']
            refstrs = [str(refdoc.metadata) + refdoc.page_content[:ref_maxlen] for refdoc in refdocs]
            print("\nAnswer:")
            print(textwrap.fill(result["answer"], 80))
            print("\nReference:")
            for refdoc in refdocs:
                print("Ref doc:\n", refdoc.metadata)
                print(textwrap.fill(refdoc.page_content[:ref_maxlen], 80))
            print("\n")
            save_qa_history(query, result, qa_path)
            if save_page_id is not None:
                try:
                    notion.blocks.children.append(save_page_id, children=QA_notion_blocks(query, answer, refstrs))
                except Exception as e:
                    print("Failed to save to notion")
                    print(e)
                    refstrs_meta = [str(refdoc.metadata) for refdoc in refdocs]
                    notion.blocks.children.append(save_page_id, children=QA_notion_blocks(query, answer, refstrs_meta))

        print(f"Finish conversation")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
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
