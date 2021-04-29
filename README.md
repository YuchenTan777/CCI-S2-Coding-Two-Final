# News text visualization - based on LDA model algorithm implementation ✨

## 1 Project Description

### 1.1 Background

With the development of information technology, people are living in a world full of information. While enjoying the convenience brought by various information services, we also have to face the situation of too much information that is difficult to handle. 

And as the main carrier of information, the phenomenon of information overload is most prominent. Therefore, the study of how to summarize the theme of text from text corpus has become a hot research topic in the field of text mining.

So for this project, I started from the most common news texts in our daily life, used `gensim` for topic modeling, and constructed a topic model based on `LDA` algorithm. At the same time, I used `matplotlib`, which I learned in class, to visualize the summarized text and make the text content more **readable and structured**.

### 1.2 Preparation

#### 1.2.1 LDA Introduction

LDA topic model was first proposed by David M. Blei, Andrew Y. Ng and Michael I. Jordan in 2002. In recent years, with the rise of social media, textual data has become an increasingly important source of analysis; the huge amount of textual data has put forward new demands on the analytical ability of social science researchers, so LDA As a probabilistic model that can extract topics from a large amount of text, topic models are increasingly used in social science research such as topic discovery and document tagging.

LDA Topic Model is a document generation model, which is an unsupervised machine learning technique. It considers a document as having multiple topics, and each topic corresponds to a different word. A document is constructed by first selecting a topic with a certain probability, and then selecting a word under this topic with a certain probability, so that the first word of this document is generated. This process is repeated continuously to generate the whole article (of course, it is assumed here that there is no order between words, i.e., all words are stacked in a large bag in an unordered manner, called a bag of words, which makes the algorithm relatively simple).

The use of LDA is the inverse of the above document generation process, i.e., based on a document obtained, to find out the topics of this document, and the words corresponding to these topics.

