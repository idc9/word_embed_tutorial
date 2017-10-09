# Introduction to word embeddings

This tutorial demonstrates how to get started with word embeddings. Using a corpus of Supreme Court opinions (provided by [CourtListener](https://www.courtlistener.com)) build a word-word co-occurence matrix which counts the number of times words co-occur in the same sentence. We then use [SVD](https://en.wikipedia.org/wiki/Singular-value_decomposition) to compute a lower rank, dense vector representation for each word.

The background material for this tutorial is

- [SLP3 chapter 15](https://web.stanford.edu/~jurafsky/slp3/15.pdf) vector semantics
- [SLP3 chapter 16](https://web.stanford.edu/~jurafsky/slp3/16.pdf) semantics with dense vectors
- The [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) by Jake VanderPlas is a wonderful (and free) python reference

Some additional resources are listed below. If you have any questions/comments shoot me an email (iain@unc.edu).


# notebooks
The content of this tutorial is in three Jupyter notebooks. There is some additional code in the *code/* folder.

[1_make_word_co_ouccurence_matrix.ipynb](https://github.com/idc9/word_embed_tutorial/blob/master/1_make_word_co_ouccurence_matrix.ipynb)

- processs raw opinion documents and creates the word-word co-ouccrence matrix 

A small word co-occurence matrix comes with the repository so the code runs, but it is likely too small to be interesting. You can download a larger word co-occurence matrix from this [google drive](https://drive.google.com/open?id=0B40b05f-8LWtVGsybWw4OTVyV00).

[2_explore_raw_counts.ipynb](https://github.com/idc9/word_embed_tutorial/blob/master/2_explore_raw_counts.ipynb)

- explores the word-word counts matrix

[3_explore_word_embedding.ipynb](https://github.com/idc9/word_embed_tutorial/blob/master/3_explore_word_embedding.ipynb)

- possibly transform the word-word counts matrix (e.g. PPMI)
- uses SVD to compute a word embedding
- explores word embedding (e.g. similar words)


# code installation

To get this tutorial on your computer clone the repository

```
git clone https://github.com/idc9/word_embed_tutorial.git
```

Make sure you have anaconda (e.g. jupyter, numpy, scipy, matplotlib, seaborn, pandas, and sklearn). Additionally, a few of the functions require: request, ast, json , and webcolors. Optionally, you can install [plotly](https://github.com/plotly/plotly.py) for interactive visualizations.

TODO: iain make better instructions/easier installation

# Download data

You can either quickly get started with notebooks 2/3 by downloading a pre-computed word co-occurence matrix, or you can download the raw data and run notebook 1 yourself.

### Quick start with pre-computed data

You can download a pre-computed word-word co-occurence matrix on 5000 opinions from [**here**](https://drive.google.com/open?id=0B40b05f-8LWtVGsybWw4OTVyV00).


### Raw data and pre-proecessing

If you want to process the raw data your self (notebook 1) you can download the 30,000 Supreme Court opinions by following the instructions in [1_process_text_files.ipynb](https://github.com/idc9/word_embed_tutorial/blob/master/1_process_text_files.ipynb). This raw data is only necessary for notebook 1.


You can get much more data from CourtListener (e.g. they have 3 million opinions) see [https://www.courtlistener.com/api/](https://www.courtlistener.com/api/).


# Additional resources

- [	Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)
- Sebastian Ruder has [an excellent series of blog posts](http://ruder.io/word-embeddings-1/) on word embeddings (more focused on neural network based embeddings)
- [Natural Language Toolkit](http://www.nltk.org/book/) 

For working with text data and doing natural language processing in python

- [spaCy](https://spacy.io/)
- [gensim](https://radimrehurek.com/gensim/)
- [Stanford Core NLP](https://github.com/dasmith/stanford-corenlp-python)
- [pytorch NLP tutorials](http://pytorch.org/tutorials/beginner/deep_learning_nlp_tutorial.html)

If you use R then check out these resources

- [Text Mining with R](http://tidytextmining.com/)
- [R for Data Science](http://r4ds.had.co.nz/)
- [wordVectors](https://github.com/bmschmidt/wordVectors) (R package for word embeddings)