import requests
# load url with bs4
# parse the html
from bs4 import BeautifulSoup
import urllib.request
arxiv_id = "2303.01469"
# arxiv_id = questionary.text("What's the arxiv id?").ask()
url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
# loader = UnstructuredURLLoader(urls=[url])
# r = requests.get(url, allow_redirects=True, )
# open(join(pdf_download_root, f"{arxiv_id}.html"), 'wb').write(r.content)
# # raise NotImplementedError
# loader = BSHTMLLoader(join(pdf_download_root, f"{arxiv_id}.html"), bs_kwargs)
# loader = PyPDFLoader(join(pdf_download_root, f"{arxiv_id}.pdf"))
# loader = PDFMinerLoader(pdf_path)
# pages = loader.load_and_split()
#%%
bs = BeautifulSoup(urllib.request.urlopen(url).read(), features="html5lib", )
#%%
# find children of the class "ltx_page_content"
elem = bs.find_all("div", class_="ltx_page_content")
#%%
elem[0].text
