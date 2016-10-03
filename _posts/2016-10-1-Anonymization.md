---
layout: post
title: "Aline"
---
<center> <img src="http://anchordentalkaty.com/assets/uploads/2015/11/11305192_l.jpg" alt="alt text" width="800px"> </center>
Automated anonymization of documents is one important research subject, especially in the medical area where [creteria](http://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/) are strictly guided. In general, it is a difficult task, and even more so for data outside of medical domain due to the lack of specialized tools. In this post, I explored the possibility of using the [vector embedding](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) [Mikolov 13] approach to anonymize non-medical documents. I arrived at some interesting results with hints from theoretical physics, which I would like share about.


### The project: anonymize text documents

Recently, I worked on a consultant project, as a [Insight Data Science](http://insightdatascience.com/) fellow, partnered with [Bonfire](http://gobonfire.com), a company based in Ontario, whose main product is a website that helps organizations (universities, hospitals, etc) to setup their procurement processes. Public purchasing often involves some very tedious and tiresome procedures, especially the very early step of preparing a bunch of documents (in jargons: [Request For Proposals](https://en.wikipedia.org/wiki/Request_for_proposal), or RFPs). Such a pain can be partially reduced if existing documents on similar purchases can be reused by different users as templates. However, for privacy concern, this requires user identifying information in these documents to be removed. So my major goal here is to **anonymize the approximately 200,000 documents** produced by a couple hundred of users.

It is by no means a simple task as search-and-replace. In writing the documents, users tend to refer themselves in various ways that cannot be easily inducted. For example, users may mention their building names, street names or even years they were founded, like in the example below.

<center> <img src="{{ site.baseurl }}/images/utopia.png" alt="alt text" height="120px"> </center>

Name entity recognition seems to be a good start. However, entity name tagging method is not working very well for documents consisting of unnatural flows of sentences, such as the FAQ style of writing that most of my documents are in. Futhermore, private information is more than just entities.

It boils down to how we define private inforamtion. Here is how "privacy" is [defined on Wikipedia](https://en.wikipedia.org/wiki/Privacy):

> Privacy is the ability of an individual or group to ***seclude*** themselves, ...

In my special case, what I am after is the information that exclusively belongs to a particular user, but not others. I need to find a way to segementate the information within a document according the closeness to a particular user.

### Word vectors and contex binding

The closeness of information to a given word can be naturally quantified through [word vectors](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) in a model called skip-gram [Mikolov 13]. The algorithm trains a vector representation of the vocabulary by binding the context (neighboring words). For any word $w$ and its context $c$, this is can be achieved by minizing the negative log-likelyhood

$$
\begin{equation}
l =  \sum_{\langle w,c\rangle} \big[ \log(1+ e^{-\boldsymbol{v}_w \cdot \boldsymbol{v}_c}) + \sum_{n\in \mathcal{N}_c}\log(1+e^{ \boldsymbol{v}_w\cdot \boldsymbol{v}_n}) \big ]
\end{equation}
$$

where $\boldsymbol{v}$ is a mapping from the vocabulary to a vector space, which is what we ar trying to obtain, and $\langle w,c\rangle$ means a neighbor pair. The second term in the squared brackets is for negative sampling where $\mathcal{N}_c$ is a set of random negative samples of $c$. Mostly, it serves like a regularization term.

The context binding here provides a way to separate out the common words. This can be visualized in the cartoon below. Intuitively, when a common word shows up in multiple contexts, the binding will pull it away from any of the contexts, as a compromise to minimize the negative log-likelyhood. As a result, only words that are unique will be binded closer in the vector space.

<center> <img src="{{ site.baseurl }}/images/wordvec.gif" alt="alt text" width="800px"> </center>

We can also look at an alternative form of Eq. (1). If we ingore the negative sampling term for a moment, and restrict the embedding $\boldsymbol{v}$ to be a unit vector, then

$$
\begin{eqnarray}
& & \arg \min \sum_{\langle w,c\rangle} \big[ \log(1+ e^{-\boldsymbol{v}_w \cdot \boldsymbol{v}_{c}}) \big]   \nonumber\\
& = & \arg \min \sum_{\langle w,c\rangle} \big [ \log( e^{-\boldsymbol{v}_w \cdot \boldsymbol{v}_{c}}) \big ] \nonumber\\
& = & \arg \min (-\sum_{\langle w,c\rangle} \boldsymbol{v}_w \cdot \boldsymbol{v}_{c} )
\end{eqnarray}
$$

This is nothing but the one-dimensional [O(n) model](https://en.wikipedia.org/wiki/N-vector_model) in statistical physics!

The O(n) model or n-vector model is a simplified physics model that explains how macro magnectization is formed as an effect of grids of tiny magnects (or spins) lining up together. Strong magnect field is formed if all the constituent individual tiny magnets are orientated in the same direction and otherwise if they are randomly orientated. Every tiny magnet is affected by and affecting its neighbors at the same time. The physics system will settle around the configuration where the total energy (defined by Eq. (2)) is minimum. Our problem at hand is very similar: we are trying to find a configuration of all the word vectors such that the total "energy" is minimum.

<center> <img src="{{ site.baseurl }}/images/nvectors.png" alt="alt text" height="80px"> </center>

Some additional notes: the special case of O(n) model with $n=1$ is called [Ising model](https://en.wikipedia.org/wiki/Ising_model), which is one of the most inspiring models that brought us rich insights into the nature of strong coupling systems, such as the fundamental interaction among subatomic particles: [quarks](https://en.wikipedia.org/wiki/Quark) and [gluons](https://en.wikipedia.org/wiki/Gluon).

### Implementation: FastText

[FastText](https://github.com/facebookresearch/fastText) is a open source tool developed by [Facebook AI](https://research.facebook.com/ai/) which is based on the skip-gram model that we just discussed, but adding an important ingradient that is highly relevant to our interest here: the subword information, based on [an character-based n-gram algorithm](https://arxiv.org/pdf/1607.04606v1.pdf) [Bojanowski&Grave 16]. Its implication here is two-folded. One, it largely reduces out-of-vocabulary scenarios that word2vec might encounter. Second, it can bind words or typos that are morphologically similar, and hence achieve some functionality of "regular expressions", which is crucial for our anonymization task.

I turned the entire set of documents into one hot word array, after some text preprocessing and clearning, then feeded it to fastText to learn the vector representation. Given a trunk of text, I will be able to calculate the angle separation between any word in the text to a given term, say "Utopia University", which then can be taken as a score measuring its closeness to such a term. It turned out that this simple method worked very well. With a fairy low false positive rate (3~4%), I was able to identify 75~80% of the sensitive words.

However, there is a caveat in this approach. Since we only use context binding, some word vectors that belong to different groups may end up being close because they might largely overlap in some contexts. There is no guarantee that this will not happen.

Actually, we leave out one important term of O(n) model in Eq. (2). With the presence of external magnectic field $\boldsymbol{\mu}$, the correct total energy should be

$$
\begin{equation}
E = - \sum_{\langle w, c\rangle} \boldsymbol{v}_w \cdot \boldsymbol{v}_c - \sum_{w} \boldsymbol{v}_w \cdot \boldsymbol{\mu}
\end{equation}
$$

The physical implication here is straightforward: every tiny magnect goes under the shaddow of its neighbors and the external field. In the problem here, the situation is similar. The words in a labeled document are essentially under the shaddow of a "external field" (the label). So inspired by the O(n) model, we should add the counterpart to the negative log-likelyhood, which is missing in the skip-gram model. That was what I did, adding a term

$$
\begin{equation}
 \sum_w \log(1+ e^{-\boldsymbol{v}_w \cdot \boldsymbol{\mu} }),
 \end{equation}
$$

which I called a **polarization** term, to the skip-gram model.

### Improvement: PolarizedText

To integrate the polarization term (Eq. (4)), I had to modify the source code of fastText and made a new repository called [PolarizedText](). PolarizedText puts each document label into its unique dimension and applies an "external field" to shaddow the text. This avoids the label confusion that might happen during the context binding. Furthermore, the separations of the unique words are maximized since they are orthogonal.

To validate the algorithm, I hand labeled about 300 documents for the words that are subject to removal. The ROC curves of various algorithms are compared below.

<center> <img src="{{ site.baseurl }}/images/roc.png" alt="alt text" width="600px"> </center>


-----

[Mikolov13] Tomáš Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013b. Distributed representations of words and phrases and their compositionality. In Adv. NIPS

[Bojanowski&Grave 16] P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information, arXiv:1607.04606
