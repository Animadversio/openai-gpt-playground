import pdfminer
import pdfminer.high_level
import pdfminer.layout
import re
import unicodedata
#%%
def remove_non_ascii(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text)
                   if unicodedata.category(c) != 'Mn')


def remove_unrecognizable_chars(text):
    # Use a regular expression to match non-printable characters
    pattern = r'[\x00-\x1F\x7F-\xFF]'
    return re.sub(pattern, '', text)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from pdf using pdfminer
    """
    text = pdfminer.high_level.extract_text(pdf_path)
    return text


def convert_to_ascii_only(text):
    str_en = text.encode("ascii", "ignore")
    str_de = str_en.decode()
    return str_de
#%%
pdf_path = r"D:\DL_Projects\NLP\arxiv_pdf\2303.01469.pdf"
raw_text = extract_text_from_pdf(pdf_path)
#%%
import textwrap
print(textwrap.fill(raw_text, 100))
#%%
pp_text = remove_non_ascii(remove_unrecognizable_chars(raw_text))
print(textwrap.fill(pp_text, 100))
#%%
# Function to convert PDF to text
def pdf_to_txt(file_path):
    import PyPDF2
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f) # PdfFileReader(f)
        text = ''
        pages = []
        # Extract text from each page of the PDF
        for page_num in range(len(pdf_reader.pages)):
            pages.append(pdf_reader.pages[page_num].extract_text())
            text += pages[-1]

    return text, pages


pdf_path = r"D:\DL_Projects\NLP\arxiv_pdf\2303.01469.pdf"
text, pages = pdf_to_txt(pdf_path)
#%%
#%%
print(textwrap.fill(remove_unrecognizable_chars(text), 100))
