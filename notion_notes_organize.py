import os
from notion_client import Client

diary_database_id = "43209ee3f26d4e2e8cdc757de1a41173"

notion = Client(auth=os.environ["NOTION_TOKEN"])
#%%
from notion_tools import print_entries
entries = notion.databases.query(database_id=diary_database_id, )
print_entries(entries)
#%%
#%%
#%%
# modify title
def update_title(page_id, title):
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
#%%
entries = notion.databases.query(database_id=diary_database_id, )
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
