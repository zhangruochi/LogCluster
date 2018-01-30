from sklearn.feature_extraction.text import CountVectorizer


texts=["10:11, dog-cat fish","dog:cat cat","fish bird", 'bird']

vectorizer = CountVectorizer()
tf = vectorizer.fit_transform(texts)

print(vectorizer.get_feature_names())
print(tf.toarray())

"""
with open("note.txt","r") as f:
    line = f.readline()
    print(line)
    line = f.readline()
    print(line)
"""

from collections import Counter

array = [1,2,3,4,1,2,3,5]

print(Counter(array))


