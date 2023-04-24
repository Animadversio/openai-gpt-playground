import requests
from os.path import join

def download_pdf(arxiv_id, save_root=""):
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    r = requests.get(url, allow_redirects=True,)
    open(join(save_root, f"{arxiv_id}.pdf"), 'wb').write(r.content)


# download_pdf("1906.04358")