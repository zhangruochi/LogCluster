# -*- coding: utf-8 -*-"
import os
from cStringIO import StringIO
import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


def convert_pdf_2_text(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    device = TextConverter(rsrcmgr, retstr, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    with open(path, 'rb') as fp:
        for page in PDFPage.get_pages(fp, set()):
            interpreter.process_page(page)
        text = retstr.getvalue()
    device.close()
    retstr.close()
    return text


def wrire_text(text, output_file):
    with open(output_file, mode="a") as f:
        f.write(text)


def main(dir):
    if not os.path.exists("output.txt"):
        file = open("output.txt", 'w')
        file.close()

    for file in os.listdir(dir):
        if file.endswith(".pdf"):
            try:
                text = convert_pdf_2_text(os.path.join(dir, file))
            except:
                continue    
            text = text.replace("\n", "")
            wrire_text(text, "output.txt")
            print("finished process {}".format(os.path.join(dir, file)))
            


if __name__ == '__main__':
    """
    path = "docs.pdf"
    text = convert_pdf_2_text(path)
    print text
    save_txt(text,path)
    """
    path = "all_docs"
    main(path)
