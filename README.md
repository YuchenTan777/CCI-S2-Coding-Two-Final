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

![image](https://github.com/YuchenTan777/CCI-S2-Coding-Two-Final/blob/main/pic/LDA.jpg)
> Figure 1.对比传统K-Means等聚类算法，LDA主题模型在文本聚类上有何优缺点？ - 拓端数据科技的回答 - 知乎
https://www.zhihu.com/question/29778472/answer/1295340045


## 2 Data Collection

### 2.1 Import NewsGroups Dataset

I first imported the newsgroup dataset and kept only `4 target_names` categories. These four categories are: motorcycle news, sports news, political news, and religious news.

```
# Import Dataset
df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
df = df.loc[df.target_names.isin(['soc.religion.christian', 'rec.sport.hockey', 'talk.politics.mideast', 'rec.motorcycles']) , :]
print(df.shape)  #> (2361, 3)
df.head()
```
![image](https://github.com/YuchenTan777/CCI-S2-Coding-Two-Final/blob/main/pic/import%20dataset.png)

### 2.2 Tokenize Sentences and Clean

Removing the emails, new line characters, single quotes and finally split the sentence into a list of words using gensim’s simple_preprocess(). Setting the deacc=True option removes punctuations.

**Because the data inside the original text is very messy, it is especially important to clean the data**

```
def sent_to_words(sentences):
    for sent in sentences:
        sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
        sent = re.sub('\s+', ' ', sent)  # remove newline chars
        sent = re.sub("\'", "", sent)  # remove single quotes
        sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
        yield(sent)  

# Convert to list
data = df.content.values.tolist()
data_words = list(sent_to_words(data))
print(data_words[:1])
```
Results:
`[['from', 'irwin', 'arnstein', 'subject', 're', 'recommendation', 'on', 'duc', 'summary', 'whats', 'it', 'worth', 'distribution', 'usa', 'expires', 'sat', 'may', 'gmt', 'organization', 'computrac', 'inc', 'richardson', 'tx', 'keywords', 'ducati', 'gts', 'how', 'much', 'lines', 'have', 'line', 'on', 'ducati', 'gts', 'model', 'with', 'on', 'the', 'clock', 'runs', 'very', 'well', 'paint', 'is', 'the', 'bronze', 'brown', 'orange', 'faded', 'out', 'leaks', 'bit', 'of', 'oil', 'and', 'pops', 'out', 'of', 'st', 'with', 'hard', 'accel', 'the', 'shop', 'will', 'fix', 'trans', 'and', 'oil', 'leak', 'they', 'sold', 'the', 'bike', 'to', 'the', 'and', 'only', 'owner', 'they', 'want', 'and', 'am', 'thinking', 'more', 'like', 'any', 'opinions', 'out', 'there', 'please', 'email', 'me', 'thanks', 'it', 'would', 'be', 'nice', 'stable', 'mate', 'to', 'the', 'beemer', 'then', 'ill', 'get', 'jap', 'bike', 'and', 'call', 'myself', 'axis', 'motors', 'tuba', 'irwin', 'honk', 'therefore', 'am', 'computrac', 'richardson', 'tx', 'dod']]`

### 2.3 Build the Bigram, Trigram Models and Lemmatize

Now, form the bigram and trigrams using the Phrases model. This is passed to Phraser() for efficiency in speed of execution.
Next, lemmatize each word to its root form, keeping only nouns, adjectives, verbs and adverbs.
We keep only these POS tags because they are the ones contributing the most to the meaning of the sentences. Here, I use spacy for lemmatization.

```
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# !python3 -m spacy download en  # run in terminal once
def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
    texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    texts = [bigram_mod[doc] for doc in texts]
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    texts_out = []
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    # remove stopwords once more after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
    return texts_out

data_ready = process_words(data_words)  # processed Text Data!
```
**Now that we have the processed data🥳**

### 2.4 Build the Topic Model

```
# Create Dictionary
id2word = corpora.Dictionary(data_ready)

# Create Corpus: Term Document Frequency
corpus = [id2word.doc2bow(text) for text in data_ready]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='symmetric',
                                           iterations=100,
                                           per_word_topics=True)

pprint(lda_model.print_topics())
```
Results:
```[(0,
  '0.012*"state" + 0.012*"israeli" + 0.011*"people" + 0.011*"kill" + '
  '0.009*"attack" + 0.009*"government" + 0.008*"war" + 0.007*"turkish" + '
  '0.006*"soldier" + 0.006*"greek"'),
 (1,
  '0.020*"game" + 0.018*"bike" + 0.017*"write" + 0.012*"article" + '
  '0.009*"rider" + 0.008*"list" + 0.008*"ride" + 0.007*"score" + '
  '0.006*"motorcycle" + 0.006*"helmet"'),
 (2,
  '0.017*"team" + 0.015*"year" + 0.012*"time" + 0.011*"write" + 0.009*"well" + '
  '0.009*"first" + 0.009*"play" + 0.008*"look" + 0.008*"help" + 0.008*"name"'),
 (3,
  '0.014*"people" + 0.012*"write" + 0.010*"believe" + 0.008*"reason" + '
  '0.007*"evidence" + 0.006*"question" + 0.006*"thing" + 0.006*"article" + '
  '0.006*"claim" + 0.005*"faith"')]
  ```
  **The decimal after each word can be considered as the probability that the word belongs to the topic, and the probability sum of all words under the topic is 1.**

### 2.5 Analyze the Text

To find out what is the Dominant topic and its percentage contribution in each document
In LDA models, each document is composed of multiple topics. But, typically only one of the topics is dominant. The below code extracts this dominant topic for each sentence and shows the weight of the topic and the keywords.
This way, you will know which document belongs predominantly to which topic.
```
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)
```


