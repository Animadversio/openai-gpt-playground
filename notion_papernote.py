#!pip install notion
#https://www.redgregory.com/notion/2020/6/15/9zuzav95gwzwewdu1dspweqbv481s5
#%%
# from notion.client import NotionClient
# from notion.block import PageBlock, TextBlock, ImageBlock
# from notion.block import VideoBlock
# client = NotionClient(token_v2=TOKEN_V2)
# #%%
# parent_page = client.get_block('https://www.notion.so/binxus-mind-palace/d3e3be7fc96a45de8e7d3a78298f9ccd?v=14b50bcfaf61435ab17725d2c0d6282b')
# #%%
# child_page = parent_page.children.add_new(PageBlock)
# child_page.title = 'NEW PAGE'
# for row in parent_page.collection.get_rows(search=""):
#     print("We estimate the value of '{}'".format(row.name, ))
# paper_page_url = r"https://www.notion.so/binxus-mind-palace/1906-04358-Weight-Agnostic-Neural-Networks-c32a0e2dcb6e4c26816871197aaeb082"
# paper_page = client.get_block(paper_page_url)
# #%%
# client.get_record_data("s-f40fd12e855a43bb8fba85b960ddbbc5")

#%%
"""
Using notion API to save chat history
"""
#%%
import os
from notion_client import Client
# os.environ["NOTION_TOKEN"] =
notion = Client(auth=os.environ["NOTION_TOKEN"])
#%%
# https://www.python-engineer.com/posts/notion-api-python/
# https://www.notion.so/{workspace_name}/{database_id}?v={view_id}
# https://www.notion.so/binxus-mind-palace/d3e3be7fc96a45de8e7d3a78298f9ccd?v=14b50bcfaf61435ab17725d2c0d6282b&pvs=4
# add connection
database_id = "d3e3be7fc96a45de8e7d3a78298f9ccd"
view_id = "14b50bcfaf61435ab17725d2c0d6282b"
#%%
entries = notion.databases.query(database_id=database_id, )
#%%
for entry in entries["results"]:
    print(entry["properties"]["Name"]["title"][0]["plain_text"], entry["id"])

#%% Query time
entries_return = notion.databases.query(database_id=database_id, filter={
      "property": "Created Time", "date": { "past_week": {} }})

#%% Query title
entries_return = notion.databases.query(database_id=database_id, filter={
      "property": "Name", "title": { "contains": "1906.04358" }})

entry = entries_return["results"][0]
#%%
page_id = "c32a0e2d-cb6e-4c26-8168-71197aaeb082"  # entry["id"]
#%%
notepage_dict = notion.pages.retrieve(page_id)
notion.blocks.retrieve(page_id,)
#%%
blocks = notion.blocks.children.list(page_id)
#%%
def QA_notion_blocks(Q, A, refs=()):
    """
    notion.blocks.children.append(page_id, children=QA_notion_blocks("Q1", "A1"))
    notion.blocks.children.append(page_id, children=QA_notion_blocks("Q1", "A1", ("ref1", "ref2")))

    :param Q: str question
    :param A: str answer
    :param refs: list or tuple of str references
    :return:
    """
    ref_blocks = []
    for ref in refs:
        ref_blocks.append({'quote': {"rich_text": [{"text": {"content": ref}}]}})
    return [
        {'paragraph': {"rich_text": [{"text": {"content": f"Question:"}, 'annotations': {'bold': True}}, ]}},
        {'paragraph': {"rich_text": [{"text": {"content": Q}}]}},
        {'paragraph': {"rich_text": [{"text": {"content": f"Answer:"}, 'annotations': {'bold': True}}, ]}},
        {'paragraph': {"rich_text": [{"text": {"content": A}}]}},
        {'toggle': {"rich_text": [{"text": {"content": f"Reference:"}, 'annotations': {'bold': True}}, ],
                    "children": ref_blocks, }},
        {'divider': {}},
    ]
#%%
import pickle as pkl
chat_history = pkl.load(open("1906.04358_qa_history\\chat_history.pkl", "rb"))
#%% Add Chat history to notion
for query, ans_struct in chat_history:
    answer = ans_struct["answer"]
    refdocs = ans_struct['source_documents']
    refstrs = [refdoc.page_content[:250] for refdoc in refdocs]
    notion.blocks.children.append(page_id, children=QA_notion_blocks(query, answer, refstrs))






#%% Scratch
# notion.blocks.children.list()

notion.blocks.children.append(page_id, children=[
        {
        "heading_2": {
          "rich_text": [
            {
              "text": {
                "content": "Lacinato kale"
              }
            }
          ]
        }
      },
      {
        "paragraph": {
          "rich_text": [
            {
              "text": {
                "content": "Lacinato kale is a variety of kale with a long tradition in Italian cuisine, especially that of Tuscany. It is also known as Tuscan kale, Italian kale, dinosaur kale, kale, flat back kale, palm tree kale, or black Tuscan palm.",
                "link": {
                  "url": "https://en.wikipedia.org/wiki/Lacinato_kale"
                }
              }
            }
          ]
        }
      }
])


# notion.blocks.delete()



#%%
from notion_client.helpers import collect_paginated_api

all_results = collect_paginated_api(
    notion.databases.query, database_id="897e5a76-ae52-4b48-9fdf-e71f5945d1af"
)