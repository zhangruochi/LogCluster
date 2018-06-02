import os
import docx2txt

def convert_docs_2_text(path):
    text = docx2txt.process(path)
    return text


def wrire_text(text, output_file):
    with open(output_file, mode="a", encoding = "utf8") as f:
        f.write(text)


def main(dir):
    if not os.path.exists("output.txt"):
        file = open("output.txt", 'w')
        file.close()

    for file in os.listdir(dir):
        if file.endswith(".doc") or file.endswith(".docx"):
            try:
                text = convert_docs_2_text(os.path.join(dir, file))
            except:
                continue    
            text = text.replace("\n", "")
            wrire_text(text, "output.txt")
            print("finished process {}".format(os.path.join(dir, file)))

if __name__ == '__main__':
    #main("all_docs")
    print(convert_docs_2_text("test.docx").replace("\n",""))