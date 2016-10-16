## Bag of words
1. Represent a document in the number of words in it and their counts;  
2. Word cloud;  
3. Term frequency: occurance of each word in a document;  
4. Though effective, throw away a lot of meaning of the original phrases: "great not bad" vs "bad not great";  
![figure1](https://cloud.githubusercontent.com/assets/14355257/19225451/1e1a87e0-8e6b-11e6-9fd3-b4b2d5b59e99.png)


# Natural Language Processing
## Introduction

In this problem you will develop two techniques for analyzing free text documents: a bag of words approach based upon creating a TFIDF matrix, and an n-gram language model.

You'll be applying your models to the text from the Federalist Papers.  The Federalist papers were a series of essay written in 1787 and 1788 by Alexander Hamilton, James Madison, and John Jay (they were published anonymously at the time), that promoted the ratification of the U.S. Constitution.  If you're curious, you can read more about them here: https://en.wikipedia.org/wiki/The_Federalist_Papers . They are a particularly interesting data set, because although the authorship of most of the essays has been long known since around the deaths of Hamilton and Madison, there was still some question about the authorship of certain articles into the 20th century.  You'll use document vectors and language models to do this analysis for yourself.

## The dataset

You'll use a copy of the Federalist Papers downloaded from Project Guttenberg, available here: http://www.gutenberg.org/ebooks/18 .  Specifically, the "pg18.txt" file included with the homework is the raw text downloaded from Project Guttenberg.  To ensure that everyone starts with the exact same corpus, we are providing you the code to load and tokenize this document, as given below.

```
import re

def load_federalist_corpus(filename):
    """ Load the federalist papers as a tokenized list of strings, one for each eassay"""
    with open(filename, "rt") as f:
        data = f.read()
    papers = data.split("FEDERALIST")
    
    # all start with "To the people of the State of New York:" (sometimes . instead of :)
    # all end with PUBLIUS (or no end at all)
    locations = [(i,[-1] + [m.end()+1 for m in re.finditer(r"of the State of New York", p)],
                 [-1] + [m.start() for m in re.finditer(r"PUBLIUS", p)]) for i,p in enumerate(papers)]
    papers_content = [papers[i][max(loc[1]):max(loc[2])] for i,loc in enumerate(locations)]

    # discard entries that are not actually a paper
    papers_content = [p for p in papers_content if len(p) > 0]

    # replace all whitespace with a single space
    papers_content = [re.sub(r"\s+", " ", p).lower() for p in papers_content]

    # add spaces before all punctuation, so they are separate tokens
    punctuation = set(re.findall(r"[^\w\s]+", " ".join(papers_content))) - {"-","'"}
    for c in punctuation:
        papers_content = [p.replace(c, " "+c+" ") for p in papers_content]
    papers_content = [re.sub(r"\s+", " ", p).lower().strip() for p in papers_content]
    
    authors = [tuple(re.findall("MADISON|JAY|HAMILTON", a)) for a in papers]
    authors = [a for a in authors if len(a) > 0]
    
    numbers = [re.search(r"No\. \d+", p).group(0) for p in papers if re.search(r"No\. \d+", p)]
    
    return papers_content, authors, numbers
        
papers, authors, numbers = load_federalist_corpus("pg18.txt")
```
You're welcome to dig through the code here if you're curious, but it's more important that you look at the objects that the function returns.  The `papers` object is a list of strings, each one containing the full content of one of the Federalist Papers.  All tokens (words) in the text are separated by a single space (this includes some puncutation tokens, which have been modified to include spaces both before and after the punctuation. The `authors` object is a list of lists, which each list contains the author (or potential authors) of a given paper.  Finally the `numbers` list just contains the number of each Federalist paper.  You won't need to use this last one, but you may be curious to compare the results of your textual analysis to the opinion of historians.

## Q1: Bag of words, and TFIDF [6+6pts]

In this portion of the question, you'll use a bag of words model to describe the corpus, and write routines to build a TFIDF matrix and a cosine similarity function.  Specifically, you should first implement the TFIDF function below.  This should return a _sparse_ TFIDF matrix (as for the Graph Library assignment, make sure to directly create a sparse matrix using `scipy.sparse.coo_matrix()` rather than create a dense matrix and then convert it to be sparse).

Important: make sure you do _not_ include the empty token `""` as one of your terms. 

```
import collections # optional, but we found the collections.Counter object useful
import scipy.sparse as sp
import numpy as np

def tfidf(docs):
    """
    Create TFIDF matrix.  This function creates a TFIDF matrix from the
    docs input.

    Args:
        docs: list of strings, where each string represents a space-separated
              document
    
    Returns: tuple: (tfidf, all_words)
        tfidf: sparse matrix (in any scipy sparse format) of size (# docs) x
               (# total unique words), where i,j entry is TFIDF score for 
               document i and term j
        all_words: list of strings, where the ith element indicates the word
                   that corresponds to the ith column in the TFIDF matrix
    """
    from collections import Counter
    cnt = Counter()
    dict_occurances = {}
    for doc in docs:
        cnt = cnt + Counter(doc.split())
        for each in Counter(doc.split()):
            if dict_occurances.has_key(each):
                dict_occurances[each] += 1
            else:
                dict_occurances[each] = 1
                
    all_words = []
    for word in cnt.items():
        all_words.append(word[0])
          
    # 86 words in all docs
    oooo = 0
    for each in dict_occurances:
        if dict_occurances[each] == len(docs):
            oooo += 1
    print oooo
    
    all_words = []
    for key in dict_occurances:
        all_words.append(key)
        
    import math
    denominator = float(len(docs))
    idf = {}
    for word in all_words:
        this_idf = math.log(denominator/dict_occurances[word])
        idf[word] = this_idf
    
    row = []
    col = []
    data = []
    row_count = 0
    col_count = 0
    for doc in docs:
        doc_words = doc.split()
        tf = Counter(doc_words)
        counter_words = []
        for word in all_words:
            tfdif = idf[word] * tf[word]
            if word in doc_words and tfdif != 0:
                row.append(row_count)
                col.append(col_count)
                data.append(tfdif)
            col_count += 1
        row_count += 1
        col_count = 0
    
    coo = sp.coo_matrix((data, (row, col)))
    
    return coo, all_words

data = [
    "the goal of this lecture is to explain the basics of free text processing",
    "the bag of words model is one such approach",
    "text processing via bag of words"
]
# Test 1
X_tfidf, words = tfidf(data)
print X_tfidf.todense()
print words

# Test 2
X_tfidf, words = tfidf(papers)
print X_tfidf.nnz

```
Our version results the following result (just showing the type, size, and # of non-zero elements):

    <86x8686 sparse matrix of type '<type 'numpy.float64'>'
        with 57607 stored elements in Compressed Sparse Row format>
     
For testing, you can also run the algorithm on the following "data set" from class:

For our implementation, this returns the following output:

    [[ 0.          0.          1.09861229  1.09861229  0.          1.09861229
       0.          0.40546511  0.40546511  1.09861229  0.          1.09861229
       0.          0.          0.40546511  1.09861229  0.81093022  0.
       1.09861229]
     [ 1.09861229  1.09861229  0.          0.          0.40546511  0.          0.
       0.40546511  0.          0.          1.09861229  0.          0.
       0.40546511  0.          0.          0.40546511  1.09861229  0.        ]
     [ 0.          0.          0.          0.          0.40546511  0.          0.
       0.          0.40546511  0.          0.          0.          1.09861229
       0.40546511  0.40546511  0.          0.          0.          0.        ]]
    ['model', 'such', 'basics', 'goal', 'bag', 'this', 'of', 'is', 'processing', 'free', 'one', 'to', 'via', 'words', 'text', 'lecture', 'the', 'approach', 'explain']
    

## Cosine similarity
```
def cosine_similarity(X):
    """
    Return a matrix of cosine similarities.
    
    Args:
        X: sparse matrix of TFIDF scores or term frequencies
    
    Returns:
        M: dense numpy array of all pairwise cosine similarities.  That is, the 
           entry M[i,j], should correspond to the cosine similarity between the 
           ith and jth rows of X.
    """
    import numpy
    row = []
    col = []
    data = []
    row_count = 0
    while row_count < X.shape[0]:
        col_count = 0
        while col_count < X.shape[0]:
            x = X.getrow(row_count)
            y = X.getrow(col_count)
            son = (x * y.T).toarray()[0][0]
            base = numpy.linalg.norm(x.toarray()) * numpy.linalg.norm(y.toarray())
            similarity = son / base
            data.append(similarity)
            row.append(row_count)
            col.append(col_count)
            col_count += 1
        row_count += 1
    return sp.coo_matrix((data, (row, col))).todense()

print cosine_similarity(X_tfidf)
```

If you apply this function to the example from class:

    M = cosine_similarity(X_tfidf)
    
we get the result presented in the slides:

    [[ 1.          0.06796739  0.07771876]
     [ 0.06796739  1.          0.10281225]
     [ 0.07771876  0.10281225  1.        ]]
     
Finally, use this model to analyze potential authorship of the unknown Federalist Papers.  Specifically, compute the average cosine similarity between all the _known_ Hamilton papers and all the _unknown_ papers (and similarly between known Madison and unknown, and Jay and unknown).  Populate the following variables with these averages.  As a quick check, our value for the `jay_mean_cosine_similarity` (and we know definitively that the unknown papers were _not_ written by Jay), is 0.064939.

# N-gram Language model
In this question, you will implement an n-gram model to be able to model the language used in the Federalist Papers in a more structured manner than the simple bag of words approach.  You will fill in the following class:

```
class LanguageModel:
    def __init__(self, docs, n):
        """
        Initialize an n-gram language model.
        
        Args:
            docs: list of strings, where each string represents a space-separated
                  document
            n: integer, degree of n-gram model
        """
        from collections import defaultdict
        from collections import deque
        from collections import Counter
        self.counts = defaultdict(list)
        self.count_sums = {}
        self.n = n
        
        cnt = Counter() 
        ngram = []
        for doc in docs:
            cnt = cnt + Counter(doc.split())
            dq = deque(maxlen=n)
            for element in doc.split():
                dq.append(element)
                if len(dq) == n:
                    ngram.append(list(dq))
                    
        self.unique_words = []
        for word in cnt.items():
            self.unique_words.append(word[0])
        
        dd = defaultdict(list)
        for each in ngram:
            prefix = ' '.join(each[:-1])
            nword = each[n-1]
            if len(dd[prefix]) == 0:
                dd[prefix] = {}
                dd[prefix][nword] = 1
            elif not dd[prefix].has_key(nword):
                dd[prefix][nword] = 1
            elif dd[prefix].has_key(nword):
                dd[prefix][nword] += 1
                
        self.counts = dd
        
        sums = {}
        for key in dd:
            key_sum = 0
            for nword in dd[key]:
                key_sum += dd[key][nword]
            sums[key] = key_sum
        self.count_sums = sums
        
        
    def perplexity(self, text, alpha=1e-3):
        """
        Evaluate perplexity of model on some text.
        
        Args:
            text: string containing space-separated words, on which to compute
            alpha: constant to use in Laplace smoothing
            
        Note: for the purposes of smoothing, the dictionary size (i.e, the D term)
        should be equal to the total number of unique words used to build the model
        _and_ in the input text to this function.
            
        Returns: perplexity
            perplexity: floating point value, perplexity of the text as evaluted
                        under the model.
        """
        from collections import Counter
        from collections import deque
        import math
        cnt = Counter(text.split())
        
        all_words = []
        text_unique_words = []
        for word in cnt.items():
            all_words.append(word[0])
        for word in all_words:
            if word not in self.unique_words:
                text_unique_words.append(word)
        D = len(text_unique_words) + len(self.unique_words) #7017

        dq = deque(maxlen=self.n - 1)
        possibility = 0
        count = 0
        for word in text.split():
            if count < self.n - 1:
                count += 1
                dq.append(word)
                continue
            prefix = ' '.join(list(dq))
            son = alpha
            base = D * alpha
            if self.counts.has_key(prefix):
                base += self.count_sums[prefix]
                if word in self.counts[prefix]:
                    son += self.counts[prefix][word]
            possibility += math.log((son / base),2)
            dq.append(word)
        preplexity = 2**(-1 * possibility / (len(text.split()) - self.n + 1))
        return preplexity
        
    def sample(self, k):
        """
        Generate a random sample of k words.
        
        Args:
            k: integer, indicating the number of words to sample
            
        Returns: text
            text: string of words generated from the model.
        """
        import random
        from collections import deque
        word_count = 0
        text = ""
        preceeding = random.choice(self.counts.keys())
        dq = deque(maxlen=self.n - 1)
        dq.append(preceeding.split()[0])
        dq.append(preceeding.split()[1])
        while word_count < k:
            preceeding = ' '.join(list(dq))
            if not self.counts.has_key(preceeding):
                preceeding = random.choice(self.counts.keys())            
            new_word_pool= []
            for each in self.counts[preceeding]:
                index = 0
                while index < self.counts[preceeding][each]:
                    new_word_pool.append(each)
                    index += 1
#             print new_word_pool
            new_word = random.choice(new_word_pool)        
            text += " " + new_word
            dq.append(new_word)
            word_count += 1
        return text
        
    
hamdocs = [paper for paper,author in zip(papers,authors) if author == ('HAMILTON',)]
l_hamilton = LanguageModel(hamdocs, 3)
print l_hamilton.perplexity(papers[0])

sample = l_hamilton.sample(200)
# print sample
print l_hamilton.perplexity(sample)
```

### Part A: Initializing the language model

First, implement the `__init__()` function in the `LanguageModel` class.  You can store the information any way you want, but we did this by building a two-level dictionary (in fact, we used the `collections.defaultdict` class, but this only make a few things a little bit shorter ... you are free to use it or not) `self.counts`, where the first key refers to the previous $n-1$ tokens, and the second key refers to the $n$th token, and the value simply stores the count of the number of times this combination was seen.  For ease of use in later function, we also created a `self.count_sums`, which contained the number of total times each $n-1$ combination was seen.

For example, letting `l_hamilton` be a LanguageModel object built just from all the known Hamilton papers and with `n = 3`, the following varibles are populated in the object:

    l_hamilton.counts["privilege of"] = {'being': 1, 'originating': 1, 'paying': 1, 'the': 1}
    l_hamilton.count_sums["privilege of"] = 4
    
We also build a `self.dictionary` variable, which is just a `set` object containing all the unique words in across the entire set of input document.

### Part B: Computing perplexity

Next, implement the `perplexity()` function, which takes a text sample and computes the perplexity of this sample under the model.  Use the formula for perplexity from the class nodes (which is actually not exact, since it only so, being careful to not multiply togther probabilites that get too small (hint: instead of taking the log of a large product, take the sum of the logs).

You'll want to be careful about dictionary sizes when it comes to the Laplace smoothing term: make sure your dictionary size $D$ is equal to the total number of unique words that occur `in either` the original data used to build the language model _or_ in the text we are evaluating the perplexity of (so the size of the union of the two).

As a simple example, if we build our `l_hamilton` model above (again, with `n=3`) and using default settings so that `alpha = 1e-3`, and run in on `papers[0]` (which was written by Hamilton), we get:

    l_hamilton.perplexity(papers[0]) = 12.5877
    
Using this model, evaluate the mean of the perplexity of the unknown Federalist papers for the language models from each of the three authors (again, using `n=3` and the default of `alpha=1e-3`).  Populate the following variables with the mean perplexities.

### Part C: Generating text

Finally, implement the `sample()` function to generate random samples of text.  Essentially, you want to pick some random starting $n-1$ tuples (you can do this any way you want), then sample according to the model.  Here you should _not_ use any Laplace smoothing, but just sample from the raw underlying counts.

One potential failure case, since we're just using raw counts, is if we generate an n-gram that _only_ occurs at the very end of a document (and so has no following n-gram observed in the data).  In this happens, just generate a new random set of $n-1$ tuples, and continue generating.

We'll be pretty lax in grading here: we're just going to be evaluating the perplexity of the generated text and make sure it's within some very loose bounds.  This is more for the fun of seeing some generated text than for actual concrete evaluation, since it's generating a random sample.  Here's what a sample of 200 words from our Hamilton model looks like (of course all random samples will be different). 

    'erroneous foundation . the authorities essential to the mercy of the body politic against these two legislatures coexisted for ages in two ways : either by actual possession of which , if it cease to regard the servile pliancy of the states which have a salutary and powerful means , by those who appoint them . they are rather a source of delinquency , it would enable the national situation , and consuls , judges of their own immediate aggrandizement ? would she not have been felt by those very maxims and councils which would be mutually questions of property ? and will he not only against the united netherlands with that which has been presumed . the censure attendant on bad measures among a multitude that might have been of a regular and effective system of civil polity had previously established in this state , but upon our neutrality . by thus circumscribing the plan of opposition , and the industrious habits of the trust committed to hands which could not be likely to do anything else . when will the time at which we might soar to a deed for conveying property of the people , were dreaded and detested'
