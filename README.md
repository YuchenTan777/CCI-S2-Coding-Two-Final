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

## 3 Analyze the Text

### 3.1 Dominant topic

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
![image](https://github.com/YuchenTan777/CCI-S2-Coding-Two-Final/blob/main/pic/keywords.png)

### 3.2 Representative sentence

```
# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
sent_topics_sorteddf_mallet.head(10)
```
![image](https://github.com/YuchenTan777/CCI-S2-Coding-Two-Final/blob/main/pic/sentence.png)

## 4 Data Visualization

### 4.1 Frequency Distribution of Word Counts in Documents

When working with a large number of documents, we want to know how big the documents are as a whole and by topic. 
I also calculate mean, median and stedv of it.

```
doc_lens = [len(d) for d in df_dominant_topic.Text]

# Plot
plt.figure(figsize=(16,7), dpi=160)
plt.hist(doc_lens, bins = 1000, color='pink')
plt.text(750, 100, "Mean   : " + str(round(np.mean(doc_lens))))
plt.text(750,  90, "Median : " + str(round(np.median(doc_lens))))
plt.text(750,  80, "Stdev   : " + str(round(np.std(doc_lens))))
plt.text(750,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
plt.text(750,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

plt.gca().set(xlim=(0, 1000), ylabel='Number of Documents', xlabel='Document Word Count')
plt.tick_params(size=16)
plt.xticks(np.linspace(0,1000,9))
plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
plt.show()
```
![image](https://github.com/YuchenTan777/CCI-S2-Coding-Two-Final/blob/main/pic/visualization/Distribution%20of%20Document%20Word%20Counts.png)

#### By Dominant Topic
```
import seaborn as sns
import matplotlib.colors as mcolors
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()] 

fig, axes = plt.subplots(2,2,figsize=(16,14), dpi=160, sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):    
    df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
    doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
    ax.hist(doc_lens, bins = 1000, color=cols[i])
    ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
    sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
    ax.set(xlim=(0, 1000), xlabel='Document Word Count')
    ax.set_ylabel('Number of Documents', color=cols[i])
    ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

fig.tight_layout()
fig.subplots_adjust(top=0.90)
plt.xticks(np.linspace(0,1000,9))
fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
plt.show()
```
![image](https://github.com/YuchenTan777/CCI-S2-Coding-Two-Final/blob/main/pic/visualization/Distribution%20of%20Document%20Word%20Counts%20by%20Dominant%20Topic.png)

### 4.2 Word Clouds of Top Keywords

Though we’ve already seen what are the topic keywords in each topic, a word cloud with the size of the words proportional to the weight is a pleasant sight. 

```
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
```
![image](https://github.com/YuchenTan777/CCI-S2-Coding-Two-Final/blob/main/pic/visualization/word%20cloud.png)

### 4.3 Word Counts of Topic Keywords

When it comes to the keywords in the topics, the importance (weights) of the keywords matters. Along with that, how frequently the words have appeared in the documents is also interesting to look.
And I plot the word counts and the weights of each keyword in the same chart.

```
from collections import Counter
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in data_ready for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()
```
![image](https://github.com/YuchenTan777/CCI-S2-Coding-Two-Final/blob/main/pic/visualization/Word%20Count%20and%20Importance%20of%20Topic%20Keywords.png)

### 4.4 What are the most discussed topics

```
def topics_per_document(model, corpus, start=0, end=1):
    corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus_sel):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)

dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            

# Distribution of Dominant Topics in Each Document
df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

# Total Topic Distribution by actual weight
topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

# Top 3 Keywords for each Topic
topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False) 
                                 for j, (topic, wt) in enumerate(topics) if j < 3]

df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
df_top3words.reset_index(level=0,inplace=True)
```
I'm make two plots:

The number of documents for each topic by assigning the document to the topic that has the most weight in that document.
The number of documents for each topic by by summing up the actual weight contribution of each topic to respective documents.

![image](https://github.com/YuchenTan777/CCI-S2-Coding-Two-Final/blob/main/pic/visualization/number%20of%20documents%20for%20each%20topic.png)

### 4.4 t-SNE Clustering Chart

t-SNE (t-Distributed Stochastic Neighbor Embedding) is a very popular nonlinear dimensionality reduction technique, mainly used to visualize high-dimensional data.

```
# Get topic weights and dominant topics ------------
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook

# Get topic weights
topic_weights = []
for i, row_list in enumerate(lda_model[corpus]):
    topic_weights.append([w for i, w in row_list[0]])

# Array of topic weights    
arr = pd.DataFrame(topic_weights).fillna(0).values

# Keep the well separated points (optional)
arr = arr[np.amax(arr, axis=1) > 0.35]

# Dominant topic number in each doc
topic_num = np.argmax(arr, axis=1)

# tSNE Dimension Reduction
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
tsne_lda = tsne_model.fit_transform(arr)

# Plot the Topic Clusters using Bokeh
output_notebook()
n_topics = 4
mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
              plot_width=900, plot_height=700)
plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
show(plot)
```
![image](https://github.com/YuchenTan777/CCI-S2-Coding-Two-Final/blob/main/pic/visualization/t-SNE%20Clustering%20Chart.png)

### 4.5 pyLDAVis

Finally, pyLDAVis is the most commonly used and a nice way to visualise the information contained in a topic model. 

```
!pip install pyLDAvis==2.1.2

import pyLDAvis.gensim

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
vis
```
**If λ is closer to 1, then words that occur more frequently under the topic are more relevant to the topic.**

**If λ is closer to 0, then the more specific and exclusive words under that topic are more relevant to the topic.**

![image](https://github.com/YuchenTan777/CCI-S2-Coding-Two-Final/blob/main/pic/visualization/pyLDAvis.png)

## 5 Conclusion 

I started from scratch by importing, cleaning and processing the newsgroups dataset to build the LDA model. Then we saw multiple ways to visualize the outputs of topic models including the word clouds and sentence coloring, which intuitively tells you what topic is dominant in each topic. A t-SNE clustering and the pyLDAVis are provide more details into the clustering of the topics.

It was an interesting attempt to help me better understand the LDA model and the flexibility of the various visualizations. At the same time, I found that there are also many people from other countries who use their own languages to build LDA topic models, such as Chinese. This is different from the way text is processed in English in many ways. I think in the future I can try theme models in multiple languages and use sentiment analysis to complete more visual charts and do some more complex representations.

## 6 Reference

[Complete Guide to Topic Modeling](https://nlpforhackers.io/topic-modeling/)

[Topic Modeling and Sentiment Analysis on Amazon Alexa Reviews](https://towardsdatascience.com/topic-modeling-and-sentiment-analysis-on-amazon-alexa-reviews-81e5017294b1)

[Topic Analysis](https://monkeylearn.com/topic-analysis/)

[Topic modeling visualization – How to present the results of LDA models?](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/#6.-What-is-the-Dominant-topic-and-its-percentage-contribution-in-each-document)

[热销商品评论之情感分析案例 - 基于LDA、贝叶斯模型算法实现](https://blog.csdn.net/weixin_42219368/article/details/80840151)

[LDA模型中文文本主题提取丨可视化工具pyLDAvis的使用](https://www.it610.com/article/1296632174824464384.htm)

[LDA on the Texts of Harry Potter-Topic Modeling with Latent Dirichlet Allocation](https://towardsdatascience.com/basic-nlp-on-the-texts-of-harry-potter-topic-modeling-with-latent-dirichlet-allocation-f3c00f77b0f5)

Kwon, H.-J. et al. (2021) ‘Topic Modeling and Sentiment Analysis of Online Review for Airlines’, Information, 12(2), p. 78.




