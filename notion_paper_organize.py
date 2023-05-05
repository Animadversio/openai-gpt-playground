import os
from notion_client import Client
from notion_tools import print_entries
import arxiv
#%%
results = list(arxiv.Search("2106.05963").results())
#%%
print(results[0].title)
print(results[0].authors)
#%%

database_id = "d3e3be7fc96a45de8e7d3a78298f9ccd"
notion = Client(auth=os.environ["NOTION_TOKEN"])
#%%
# query database, search for entries with "arxiv" in the Link field
entries = notion.databases.query(database_id=database_id, page_size=10,
                                 filter={"property": "Link", "url": {"contains": "arxiv"}})
print_entries(entries)
#%%
arxiv_filter = {"property": "Link", "url": {"contains": "arxiv"}}
all_results = []
# Keep querying pages of results until there are no more
next_page = None
while True:
    # Query the database with the filter criterion and the next page URL
    results = notion.databases.query(
        **{
            "database_id": database_id,
            "filter": arxiv_filter,
            "page_size": 100,
            "start_cursor": next_page
        }
    )
    # Add the current page of results to the list
    all_results.extend(results["results"])
    # Check if there are more pages of results
    if not results["has_more"]:
        break
    # Get the URL of the next page of results
    next_page = results["next_cursor"]
    print(len(all_results))
#%%
print_entries(all_results)
#%%
paper_page = notion.pages.retrieve("375d5e9f-a9d9-4d29-88f1-53c59da368f4")
#%%
notion.blocks.children.list("91f81888-c432-4c5c-967c-62d0fc7f7878")
notion.blocks.delete('601ec4be-b864-43c8-92cc-943d4f9c35ff')
#%%
#%%
# modify title
# modify title
def update_author(notion: Client, page_id, names):
    update_struct = {
        "properties": {
            "Author": {
                "multi_select": [
                    {'name': name} for name in names
                ]
            }
        }
    }
    notion.pages.update(page_id, **update_struct)
# modify title
def update_title(notion: Client, page_id, title):
    update_struct = {
        "properties": {
            "title": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            }
        }
    }
    notion.pages.update(page_id, **update_struct)


def update_discipline(notion: Client, page_id, discipline="ML"):
    update_struct = {
        "properties": {
            'Discipline': {
               'multi_select': [{ 'name': discipline, }]},}
    }
    notion.pages.update(page_id, **update_struct)

def update_date(notion: Client, page_id, date):
    update_struct = {
        "properties": {
            'Publishing/Release Date': {
               'date': {'start': date, }}}
    }
    notion.pages.update(page_id, **update_struct)

def update_url(notion: Client, page_id, url):
    update_struct = {
        "properties": {
            'Link': {
               'url': url}}
    }
    notion.pages.update(page_id, **update_struct)


def add_quote_block(notion: Client, page_id, quote):

    quote_block = {'quote': {"rich_text": [{"text": {"content": quote}}]}}
    notion.blocks.children.append(page_id, **{
        "children": [quote_block]
    })
#%%
from tqdm import tqdm
# update discipline to include ML
for entry in tqdm(all_results):
    entry_id = entry["id"]
    title = entry["properties"]["Name"]["title"][0]["plain_text"]
    arxiv_id = entry["properties"]["Link"]["url"].split("/")[-1]
    update_discipline(notion, entry_id, "ML")
#%%
for entry in tqdm(all_results):
    entry_id = entry["id"]
    title = entry["properties"]["Name"]["title"][0]["plain_text"]
    arxiv_id = entry["properties"]["Link"]["url"].split("/")[-1]
    results = list(arxiv.Search(arxiv_id).results())
    if len(results) == 0:
        print("No results found for", arxiv_id)
        continue
    results = results[0]
    try:
        # update author
        authors = results.authors
        authors = [author.name for author in authors]
        update_author(notion, entry_id, authors)
        # update date
        date = results.published.date()
        update_date(notion, entry_id, date.isoformat())
    except Exception as e:
        print(e)
        print("Error updating", arxiv_id)
        continue

#%%
# for entries with pdf link, use the abs link instead, add abstract to the notion
for entry in tqdm(all_results):
    entry_id = entry["id"]
    title = entry["properties"]["Name"]["title"][0]["plain_text"]
    arxiv_id = entry["properties"]["Link"]["url"].split("/")[-1]
    if arxiv_id.endswith(".pdf"):
        arxiv_id = arxiv_id[:-4]
        results = list(arxiv.Search(arxiv_id).results())
        if len(results) == 0:
            print("No results found for", arxiv_id)
            continue
        results = results[0]
        try:
            # update author
            authors = results.authors
            authors = [author.name for author in authors]
            update_author(notion, entry_id, authors)
            # update date
            date = results.published.date()
            update_date(notion, entry_id, date.isoformat())
            # update url
            update_url(notion, entry_id, results.pdf_url.replace("pdf", "abs"))
            # Add abstract as block
            add_quote_block(notion, entry_id, results.summary)
        except Exception as e:
            print(e)
            print("Error updating", arxiv_id)
            continue
    else:
        continue

#%%
entries = notion.databases.query(database_id=database_id, )
import datetime
for entry in entries["results"]:
    page_id = entry["id"]
    # page_prop = notion.pages.retrieve(page_id)
    if entry["properties"]['Date']['date'] is None:
        date = ""
    else:
        date = entry["properties"]['Date']['date']['start']  #["plain_text"]
    if len(entry["properties"]["Name"]["title"]) == 0:
        title_old = ""
    else:
        title_old = entry["properties"]["Name"]["title"][0]["plain_text"]
    if title_old == "Daily Entry" or title_old == "":
        if date == "":
            date = entry["created_time"][:10]
        print(date)
        print(title_old, page_id)
        date_ = datetime.date.fromisoformat(date)
        # format date like Apr. 1, 2021
        datestr_new = date_.strftime("%b.%d, %Y")
        update_title(page_id, "Diary "+datestr_new)
        print(datestr_new)
    # update_title(entry["id"], date)
    # print(entry["properties"]["Name"]["title"][0]["plain_text"], entry["id"])
    # update_title(entry["id"], "test")
# page_prop = notion.pages.retrieve(entries["results"][0]["id"])
# date = page_prop["properties"]['Date']['date']['start']  #["plain_text"]
# update_title(entries["results"][0]["id"], "test")
