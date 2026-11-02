from pymupdf import open
import re
from typing import List
from pathlib import Path


def extract_paragraphs(pdf_path: str | Path) -> List[str]:   

    """
    this function takes a pdf file path as input and returns a list of paragraphs extracted from the pdf file.
    """

    doc = open(pdf_path)    
    paragraphs = []
    for page in doc:
        # get blocks of text from the page
        blocks = page.get_text("blocks")
        # sort the blocks from top to buttom
        blocks = sorted( blocks, key=lambda b: (b[1], b[0]) )

        for block in blocks:
            # get the text from the block
            text = block[4]
            # replace new lines with space and remove extra spaces
            text = re.sub(r'\r+', ' ', text).strip()

            if text:
                paragraphs.append(text)

    return paragraphs


# print(extract_paragraphs("../pdfs/Penguins_ACL.pdf"))

