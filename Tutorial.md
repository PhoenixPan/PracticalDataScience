# Find Mutual Friends Using Hadoop MapReduce

## Introduction
#### MapReduce

The MapReduce programming model is designed to process big dataset using a large number of machines in-parallel. It could be illustrated as a "server farm". The main advantage of MapReduce is to process data on a flexible scale over many machines. Imagine you are processing a dataset of 100TB, it may take days to complete this task on a single machine, even it's a very powerful one, but perhaps only a few hours on a cluster of twenty regular machines, as they are working simultaneously and collaboratively on one dataset. Though Google pioneered this model, the company keeps it a secret, so common developers have to turn to Hadoop MapReduce. 

#### Hadoop MapReduce

Hadoop MapReduce is an open-source implementation of Google's MapReduce. Obviously, it is a software framework used to process vast amounts of data in-parallel on a large cluster of machines in a reliable, fault-tolerant manner. Though Hadoop MapReduce is written in Java, its implementation can be realzied through other languages, such as Python and C. Hadoop presents MapReduce as the analytics engine and uses the Hadoop Distributed File System (HDFS) as the underlying file system. 

#### How it works
Hadoop MapReduce has mainly two phases: Map and Reduce. The "map phrase" includes split, map, and partition tasks and the "reduce phase" includes shuffle, merge & sort, and reduce tasks. There should be at least one map task and zero or more reduce tasks in a MapReduce job. The job first creates HDFS partitions by spliting the dataset into chunks of a certain size and then distribute the chunks to all the nodes in the cluster. MapReduce jobs can process HDFS blocks in parallel at distributed machines and save the output to one directory.

A standard MapReduce would follow these six steps:  
1. **Map-Spliting**: The dataset is firstly splited into HDFS blocks chunks. If a file is smaller than the size limit, it remains intact, otherwise it will be split. The default size of each chunk is 100MB, but this can be determined by user configuraion. For example, if we have two files for the MapReduce job, one at 87MB and the other at 123MB, we will have three HDFS chunks: 87MB, 100MB, 23MB. This process will be completed automatically by Hadoop MapReduce and is not controled by our code; 

2. **Map-Map**: This task is where our code for Map phase gets executed. For each line of the data, our map() function is invoked to process the line according to the desired patterns. The output of our map() function would be key-value pairs. For example, if we analyze an article, we may break lines into words, and generate a key-value pair for each word as ['word':1]. This is a process where we "extend" our data with redundancy;

3. **Map-Partition**: The output of Map function is exported directly to a in-memory buffer using "print" in Python (System.out.println() in Java). Each time when the buffer is almost full, 80% by default, it created a sorted partition. The output of this task are partitions with various features;

4. **Reduce-Shuffle**: The data partitions from Map phase will be shuffled. Partitions share certain charactistics will be recognized as one group.This is a process controlled by MapReduce's Resource Manager;

5. **Reduce-Merge&Sort**: Those partitions that have been categorized as one group will together form a single, large partition, which will be used to feed the Reduce task. This is a process controlled by MapReduce's Resource Manager;

6. **Reduce-Reduce**: A partiion, a group of data, will be processed by our reduce() function in this task. Ususally, it should summarize the output, leading to a significant reduce in the data size, as this task removes redundancy;

(Step 3, 4 and 5 are often recogized as one "Shuffle" stage in many cases, as they are closed related and are not controlled by user functions)  


## Understanding MapReduce: Wordcount

Wordcount is the simplest implementation of MapReduce task. Let's get our hands dirty now. Given the sentence below:

__"I have a pan, I have an applie, ah, apple-pan."__

We firstly clean up data using the function below. Though we could choose to clean data using similar code inside our map() function, that will be processed for each word and significantly slows the tasks in some cases. 
```
text = "I have a pan, I have an applie, ah, apple-pan."

import nltk
import string

def clean(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
    result = []
    processed = text.lower()
    processed = processed.replace("'s", "")
    processed = processed.replace("'", "")
    remove_punct = string.maketrans(string.punctuation," "*len(string.punctuation))
    processed = processed.translate(remove_punct)
    for element in nltk.word_tokenize(processed):
        try:
            result.append(str(lemmatizer.lemmatize(element)))
        except:
            continue
    return ' '.join(result)
```
We shall have：
```
print clean(text)
```
#### Execution Steps:

##### Split
The Map phase will firstly split the data into HDFS chunk and distribute them to several different machines, let's say three machines:  

Assume each chunk can have maximum four words (corresponding to 100MB):  
HDFS 1: i have a pan  
HDFS 2: i have an apple  
HDFS 3: ah apple pan  

Machine 1: HDFS1(i have a pan)  
Machine 2: HDFS2(i have an apple)  
Machine 3: HDFS3(ah apple pan)  

##### Map  
Now we execute a map() function for each chunk. After the Map function, we shall have many uncategorized partitions in each machine:  

Machine 1: {"i":1}  
Machine 1: {"have":1}  
Machine 1: {"a":1}  
Machine 1: {"pan":1}  
  
Machine 2: {"i":1}  
Machine 2: {"have":1}  
Machine 2: {"an":1}  
Machine 2: {"apple":1}   
  
Machine 3: {"ah":1}   
Machine 3: {"apple":1}    
Machine 3: {"pan":1}

##### Shuffle    
We categorize and sort similar partitions and make them new partitions, which we used to feed our Reducer.     
  
New Partition 1: {"i":1}  
New Partition 1: {"i":1}  
New Partition 1: {"have":1}  
New Partition 1: {"have":1}  
  
New Partition 2: {"a":1}  
New Partition 2: {"an":1}  
New Partition 2: {"ah":1}  
  
New Partition 3: {"pan":1}    
New Partition 3: {"pan":1}  
New Partition 3: {"apple":1}     
New Partition 3: {"apple":1}     

##### Reduce   
Finally we invode reduce() function on each newly-formed partition and get the word counts for each word.  
Reducer 1: {"i":2}   
Reducer 1: {"have":2}  
  
Reducer 2: {"a":1}  
Reducer 2: {"an":1}  
Reducer 2: {"ah":1}  
  
Reducer 3: {"pan":2}   
Reducer 3: {"apple":2}  

The final results of word counts will be put to target HDFS directory at the user's disposal.

## References
Setting up Hadoop for single node ubuntu machine  
http://thepowerofdata.io/setting-up-a-apache-hadoop-2-7-single-node-on-ubuntu-14-04/  
