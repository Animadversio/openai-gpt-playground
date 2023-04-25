
import os
from os.path import join
import textwrap
import pickle as pkl
import urllib.request
from notion_client import Client
from langchain.document_loaders import PDFMinerLoader, PyPDFLoader, BSHTMLLoader, NotionDBLoader, UnstructuredURLLoader # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings  # for creating embeddings
from langchain.vectorstores import Chroma  # for the vectorization part
from langchain.chains import ChatVectorDBChain  # for chatting with the pdf
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import OpenAI  # the LLM model we'll use (CHatGPT)
from langchain.chat_models import ChatOpenAI
import questionary
from notion_tools import QA_notion_blocks, clean_metadata, print_entries, save_qa_history

notion = Client(auth=os.environ["NOTION_TOKEN"])
database_id = "d3e3be7fc96a45de8e7d3a78298f9ccd"
if os.environ["COMPUTERNAME"] == 'PONCELAB-OFF6':
    embed_rootdir = r"D:\DL_Projects\NLP\Embed_data"
    pdf_download_root = r"D:\DL_Projects\NLP\arxiv_pdf"
else:
    embed_rootdir = r"E:\DL_Projects\NLP\Embed_data"
    pdf_download_root = r"E:\DL_Projects\NLP\arxiv_pdf"

#%%
savestr = questionary.text("Directory to save the embedding").ask()
use_exist = questionary.confirm("Use existing embedding?", default=True).ask()

while True:
    save_page_id = questionary.text("Notion page id for the page to save into?").ask()
    if save_page_id == "" or save_page_id is None:
        if questionary.confirm("Search for a relavent Notion page?", default=True).ask():
            text_query = questionary.text("Title search query?").ask()
            entries_return = notion.databases.query(database_id=database_id, filter={
                "property": "Name", "title": {"contains": text_query}})
            print_entries(entries_return)
            entry = entries_return["results"][0]
        else:
            print("Will not be saved to notion page")
            save_page_id = None
            break
    else:
        if questionary.confirm(f"Save to this page? {save_page_id}").ask():
            break

# save_page_id = None
# chat_temperature = 0.5
# use_exist = True
# savestr = "cnn_2023_04_05"
# doctype = "html"
# ref_maxlen = 300
embed_persist_dir = join(embed_rootdir, savestr)
embeddings = OpenAIEmbeddings()
if os.path.exists(embed_persist_dir) and use_exist:
    print("Loading embeddings from", embed_persist_dir)
    vectordb = Chroma(persist_directory=embed_persist_dir, embedding_function=embeddings)
else:
    doctype = questionary.select(
        "Type of document you want to load?",
        choices=[
            'html', 'pdf', 'notion', "arxiv", "ar5iv"
        ]).ask()
    #%%
    if doctype == "html":
        # ["https://www.cnn.com/2023/03/30/politics/donald-trump-indictment/index.html",
        #  "https://www.cnn.com/politics/live-news/trump-indictment-stormy-daniels-news-04-03-23/index.html",
        #  "https://www.cnn.com/politics/live-news/donald-trump-court-charges-04-05-23/index.html"]
        raw_str = questionary.text("What's the html url to parse?", multiline=True).ask()
        html_urls = raw_str.split("\n")
        html_urls = [url.strip() for url in html_urls if url.strip() != ""]
        # strip the url
        loader = UnstructuredURLLoader(urls=html_urls)
        pages = loader.load_and_split()
    elif doctype == "pdf":
        pdf_path = questionary.path("What's the path to the pdf file?").ask()
        loader = PyPDFLoader(pdf_path)
        # loader = PDFMinerLoader(pdf_path)
        pages = loader.load_and_split()
    elif doctype == "arxiv":
        import requests
        arxiv_id = questionary.text("What's the arxiv id?").ask()
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        # loader = UnstructuredURLLoader(urls=[url])
        r = requests.get(url, allow_redirects=True, )
        open(join(pdf_download_root, f"{arxiv_id}.pdf"), 'wb').write(r.content)
        loader = PyPDFLoader(join(pdf_download_root, f"{arxiv_id}.pdf"))
        # loader = PDFMinerLoader(pdf_path)
        pages = loader.load_and_split()
    elif doctype == "ar5iv":
        import requests
        # load url with bs4
        # parse the html
        from bs4 import BeautifulSoup
        import urllib.request
        import re
        arxiv_id = questionary.text("What's the arxiv id?").ask()
        url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        # loader = UnstructuredURLLoader(urls=[url])
        r = requests.get(url, allow_redirects=True, )
        open(join(pdf_download_root, f"{arxiv_id}.html"), 'wb').write(r.content)
        # raise NotImplementedError
        loader = BSHTMLLoader(join(pdf_download_root, f"{arxiv_id}.html"),
                  open_encoding="utf8", bs_kwargs={"features": "html.parser"})
        # loader = PyPDFLoader(join(pdf_download_root, f"{arxiv_id}.pdf"))
        # loader = PDFMinerLoader(pdf_path)
        # bs = BeautifulSoup(urllib.request.urlopen(url).read(), features="html5lib")
        # bs = BeautifulSoup(urllib.request.urlopen(url).read(), "html.parser")
        pages = loader.load_and_split()
        #%%
    elif doctype == "notion":
        loader = NotionDBLoader(os.environ["NOTION_TOKEN"], database_id)
        page_id2read = questionary.text("What's the page id of the page in notion?").ask()
        doc = loader.load_page(page_id2read)
        doc.metadata = clean_metadata(doc.metadata)
        # page_save_name = page_title.replace("|", " ").replace(":", " ").replace(" ", "_")
        pages = RecursiveCharacterTextSplitter().split_documents([doc])
    else:
        raise NotImplemented("doctype must be one of 'html', 'pdf', 'arxiv', or 'notion'")
    #%% Examine documents

    if questionary.confirm("Examine documents?").ask():
        for page in pages:
            print(page.metadata, "\n", page.page_content[:500], "\n")
            if questionary.confirm("Next page?").ask():
                pass
            else:
                break

    #%% get embedding from pdf or htmls
    if questionary.confirm("Compute embedding now?").ask():
        print("Creating embeddings and saving to", embed_persist_dir)
        vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                         persist_directory=embed_persist_dir, )
        vectordb.persist()
    else:
        exit()
#%%
chat_temperature = questionary.text("Sampling temperature for ChatGPT?", default="0.5").ask()
chat_temperature = float(chat_temperature)
ref_maxlen = questionary.text("Max length of reference document?", default="300").ask()
ref_maxlen = int(ref_maxlen)
pdf_qa_new = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=chat_temperature, model_name="gpt-3.5-turbo"),
                                    vectordb.as_retriever(), return_source_documents=True, max_tokens_limit=4000)

#%%
qa_path = embed_persist_dir + "_qa_history"
os.makedirs(qa_path, exist_ok=True)
while True:
    # query = "For robotics purpose, which algorithm did they used, PPO, Q-learning, etc.?"
    query = questionary.text("Question: ", multiline=True).ask()
    if query == "" or query is None:
        is_exit = questionary.confirm("Exit?").ask()
        if is_exit:
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
