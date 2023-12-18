# https://openreview.net/group?id=NeurIPS.cc/2023/Conference#tab-accept-poster

#%%
import requests
import json
import bs4
import re
#%%
# parse the html using beautiful soup and store in variable `soup`
import openreview

client = openreview.Client(baseurl='https://api.openreview.net',
                           username='binxu.wang@wustl.edu', password='Zeratul001')
#%%
notes = openreview.tools.iterget_notes(client,
               invitation='ICLR.cc/2019/Conference/-/Blind_Submission')
for note in notes:
    print(note.content['title'])
#%%

notes = openreview.tools.iterget_notes(client,
               invitation='NeurIPS.cc/2022/Conference/-/Blind_Submission')
for note in notes:
    print(note.content['title'])
#%%
client.get_group('NeurIPS.cc/2022/Conference')
#%%
client.get_all_invitations(signature='NeurIPS.cc/2022/Conference')
#%%
notes = openreview.tools.iterget_notes(client,
       invitation="NeurIPS.cc/2022/Conference/-/Blind_Submission")
for note in notes:
    print(note.content['title'])
    # print(note.content['abstract'])

#%%