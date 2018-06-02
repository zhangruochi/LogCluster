import os


def recursive_main(path):
    if os.path.isfile(path):
        if path.endswith(".pdf") or path.endswith(".docx") or path.endswith(".docx") or path.endswith(".txt"):
            os.system("mv '{}' /Volumes/Database/docs/all_docs".format(path))
    else:
        for file in os.listdir(path):
            if file == "all_docs":
                continue
            recursive_main(os.path.join(path,file))


def rename_all_docs(dir):
    index = 0
    for file in os.listdir(dir):
        new_name = str(index) + "." + file.split(".")[-1]
        os.system("mv '{}' {}".format(os.path.join(dir,file), os.path.join(dir,new_name)))
        index += 1


if __name__ == '__main__':
    #recursive_main(".")
    rename_all_docs("all_docs")
