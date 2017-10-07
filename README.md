# Introduction to word embeddings

This tutorial demonstrates how to get started with word embeddings. Starting with a corpus of Supreme Court opinions (provided by [CourtListener](https://www.courtlistener.com)) we tokenize each opinion into sentences. We then build a word-word co-occurence matrix which counts the number of times words co-occur in the same sentence. We then use [SVD](https://en.wikipedia.org/wiki/Singular-value_decomposition) to compute a lower rank, dense vector representation for each word.

The first two notebooks deal with text processing; the second two deal with word statistics/embeddings.

The background material for this tutorial is

- [SLP3 chapter 15](https://web.stanford.edu/~jurafsky/slp3/15.pdf) vector semantics
- [SLP3 chapter 16](https://web.stanford.edu/~jurafsky/slp3/16.pdf) semantics with dense vectors
- The [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) by Jake VanderPlas is a wonderful (and free) python reference

Some additional resources are listed below. If you have any questions/comments shoot me an email (iain@unc.edu).


### text processing

[1_process_text_files.ipynb](https://github.com/idc9/word_embed_tutorial/blob/master/1_process_text_files.ipynb)
- process raw data
- turn 30,000 documents into one bag of sentences file (one sentence on each line)


[2_make_co_occurrence_matrix.ipynb](https://github.com/idc9/word_embed_tutorial/blob/master/2_make_co_occurrence_matrix.ipynb)
- create the word-word co-ouccrence matrix from the bag of sentences file

### word embeddings and statistics

[3_explore_raw_counts.ipynb](https://github.com/idc9/word_embed_tutorial/blob/master/3_explore_raw_counts.ipynb)
- explore the word-word counts matrix

[4_explore_word_embedding.ipynb](https://github.com/idc9/word_embed_tutorial/blob/master/4_explore_word_embedding.ipynb)
- possibly transform the word-word counts matrix (e.g. PPMI)
- use SVD to compute a word embedding
- explore word embedding (e.g. similar words)


# code installation

To get this tutorial on your computer clone the repository

```
git clone https://github.com/idc9/word_embed_tutorial.git
```

Make sure you have anaconda (e.g. jupyter, numpy, scipy, matplotlib, seaborn, pandas, and sklearn). Additionally, a few of the functions require: request, ast, json , and webcolors. Optionally, you can install [plotly](https://github.com/plotly/plotly.py) for interactive visualizations.



# Download data

You can either quickly get started with notebooks 3/4 by downloaded a pre-computed word co-occurence matrix, or you can download the raw data and run notebooks 1/2 yourself.

### Quick start with pre-computed data

You can download a pre-computed word-word co-occurence matrix from [**here**](https://drive.google.com/open?id=0B40b05f-8LWtVGsybWw4OTVyV00), place it in the `data/` folder then start with notebooks 3 and 4. 


### Raw data and pre-proecessing

If you want to process the raw data your self (notebooks 1 and 2) you can download the Supreme court by clicking the following the instructions in [1_process_text_files.ipynb](https://github.com/idc9/word_embed_tutorial/blob/master/1_process_text_files.ipynb). This raw data is only necessary for notebooks 1 and 2.


You can get much more data from CourtListener (e.g. they have 3 million opinions) see [https://www.courtlistener.com/api/](https://www.courtlistener.com/api/).


# Additional resources

These 
- [Natural Language Toolkit](http://www.nltk.org/book/) 
- [	Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)
- Sebastian Ruder has [an excellent series of blog posts](http://ruder.io/word-embeddings-1/) on word embeddings (more focused on neural network based embeddings)

For working with text data and doing natural language processing in python
- [spaCy](https://spacy.io/)
- [gensim](https://radimrehurek.com/gensim/)
- [Stanford Core NLP](https://github.com/dasmith/stanford-corenlp-python)
- [pytorch NLP tutorials](http://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)

If you use R then check out these resources
- [Text Mining with R](http://tidytextmining.com/)
- [R for Data Science](http://r4ds.had.co.nz/)
- [wordVectors](https://github.com/bmschmidt/wordVectors) (R package for word embeddings)