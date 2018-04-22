This package includes some scripts related to log clustering

## file specification

**cluster.py** 
- implement log clustering 
- conmmand: python3 cluster.py
- to change the parameters, you need modify the [CLUSTER] section in config.cof

**counting.py**
- an assistant algorithm for cluster.py, you can use it to find the abnormal logs which have some lower-frequency words
- command: python3 counting.py
- to change the parameters, you need modify the [COUNTING] section in config.cof

**database.py**
- the result of clustering is saved in a database(pickle), you can access it to get more details.
- command: python3 database.py database_name

**config.cof**
- the config file


## parameters specification of config.cof
- **filename**: 
    1. string 
    2. the name of log file you want to deal with

- **clusters**: 
    1. int
    2. the number of cluster you want to get

- **max**: 
    1. float in range [0.0, 1.0] or int
    2. When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts.

- **min**: 
    1. float in range [0.0, 1.0] or int
    2. When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. 

- **chars**: 
    1. list of string
    2. the chars not used for split log

- **threshold**: 
  1. int 
  2. if the frequent of a word is lower than hreshold, we define this word as low-frequency words


- **num**: 
    1. int 
    2. if the number of low-frequency words in a log is larger than this parameter, we define this log as abnormal  