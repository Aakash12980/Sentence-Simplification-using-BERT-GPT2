
<h1 align="center">Leveraging BERT-to-GPT2 for Sentence Simplification</h1>
<h3 align="center"> An Encoder-Decoder Transformer model for simplifying English Sentences </h3>  

</br>

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project"> ➤ About The Project</a></li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#model-architecture"> ➤ Model Architecture</a></li>
    <li><a href="#results-and-discussion"> ➤ Results and Discussion</a></li>
    <li><a href="#references"> ➤ References</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ABOUT THE PROJECT -->
<h2 id="about-the-project"> :pencil: About The Project</h2>

<p align="justify"> 
    The project is based on simplification of English sentences. Given an English sentence as input, 
    the model aims to rearrange the words or substitute words or phrases to make it easier to comprehend without losing the underlying information carried by the original sentence.
    This project is an effort to find an approach to achieve good result in sentence simplification task. 
    The project’s output can be useful for other NLP tasks which requires simplified sentences such as Machine Translation, Summarization, Classification, etc. The project leverages Bert model as encoder and GPT-2 model as decoder. 
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREREQUISITES -->
<h2 id="prerequisites"> :fork_and_knife: Prerequisites</h2>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try) <br>

<!--This project is written in Python programming language. <br>-->
The following major packages are used in this project:
* Python v3.0+
* Pandas v1.1.0+
* SK-learn v0.24+
* Pytorch v1.5+
* Transformers(Huggingface) v3.3+

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- :paw_prints:-->
<!-- FOLDER STRUCTURE -->
<h2 id="folder-structure"> :cactus: Folder Structure</h2>

    Base Project Folder
    .
    │
    ├── dataset
    │   ├── src_train
    │   ├── src_valid
    │   ├── src_test
    │   ├── tgt_train
    │   ├── tgt_valid
    │   ├── tgt_test
    │   ├── ref_test
    │   ├── ref_valid
    │   ├── src_file
    │
    ├── best_model
    │   ├── model.pt
    │
    ├── checkpoint
    │   ├── model_ckpt.pt
    │
    ├── outputs
    │   ├── decoded.txt
    │
    ├── run.py
    ├── data.py
    ├── sari.py   
    ├── tokenizer.py
 

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- DATASET -->
<h2 id="dataset"> :floppy_disk: Dataset</h2>
<p> 
    Wiki dataset comprising of parallel corpus of normal sentences and simple sentences is used to train the model. The original dataset consists of around 167k English sentence pairs from the Wikipedia articles. The dataset comprises of mapping of one-to-many, one-to-one and many-to-one sentence pairs. But the dataset was not suitable for the training without preprocessing. Upon tokenizing the sentences, sentences having token length of more than 80 were removed keeping the maximum token length of sentences to 80. The resulting training dataset became 138k from 167k. <br>
    For the evaluation and testing purpose, TurkCorpus is used. The dataset consists of 2k manually prepared sentence pairs with 8 reference sentences and 300 sentences for testing purpose which also has 8 reference sentences.

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- DATASET -->
<h2 id="dataset"> :floppy_disk: Model Architecture</h2>
<p> 
    The project provides an end-to-end pipeline for the simplification task with supervised technique using SOTA transformer models. The model accepts normal sentences as input. The sentences are converted to token embedding using BERTTokenizer. The token embeddings are fed into the encoder-decoder model. Finally, the model outputs token embeddings for simple sentences which are then converted to sentences using GPT2Tokenizer.

<p align="center">
  <img src="images/model architecture.jpg" alt="model architecture" width="75%" height="75%">
</p>    

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- RESULTS AND DISCUSSION -->
<h2 id="results-and-discussion"> :mag: Results and Discussion</h2>

<p align="justify">
The BERT-to-GPT2 model was able to achieve SARI score of 35.17 with BLEU score of 37.39. However, BERT-to-GPT-2 model was able to simplify sentences with promising results in most of the sentences. The model is mostly seen to substitute words with corrsponding simple words maintaing its context. The model is also able to simplify sentences in phrase level as well. Few of the examples of the results are shown in the table below.
<p align="center">
  <img src="images/output by B2G model 1.jpg" alt="Model Outputs" width="75%" height="75%">
</p>
</p>
<p align="justify">
However, my model failed to remember certain words during the inference time. For example, in 
the first instance of table 4.2, the word ‘tarantula’ has been misinterpreted as ‘talisman’ which 
might create a lot of confusion to the readers. Also, in the second example of table 4.2, the year 
1982 has been missed which is crucial for keeping the exact information of the whole sentence. 
The model failed to provide good results in some of the examples.
<p align="center">
  <img src="images/output by B2G model 2.jpg" alt="Model Outputs" width="75%" height="75%">
</p>

</p>

<p align="justify">
There are various reasons for the poor result for the model’s outputs. Firstly, the dataset itself does not have good pairs of sentences. Most of the sentences are totally similar in the dataset. The model could have performed better if some gold-level dataset, such as Newsela dataset, were used during training of the model. Secondly, the hyper parameters of the model need to be adjusted properly to get the better result. Since it is computationally very expensive to train the model, therefore it is hard to tune the hyperparameters. 
</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- REFERENCES -->
<h2 id="references"> :books: References</h2>

<ul>
  <li>
    <p>Raman Chandrasekar and Bangalore Srinivas. 1997. Automatic induction of rules for text simplification. In Knowledge Based Systems.
    </p>
  </li>
  <li>
    <p>
      David Vickrey and Daphne Koller. 2008. Sentence simplification for semantic role labeling. In Proceedings of ACL.
    </p>
  </li>
  <li>
    <p>
      Lijun Feng. 2008. Text simplification: A survey. CUNY Technical Report.
    </p>
  </li>
  <li>
    <p>
      Xu, Wei, CourtneyNapoles, ElliePavlick, QuanzeChen, and ChrisCallison-Burch. 2016. Optimizing statistical machine translation for text simplification.
    </p>
  </li>
  <li>
    <p>
      Sanqiang Zhao, Rui Meng, Daqing He, Saptono Andi, and Parmanto Bambang. Integrating transformer and paraphrase rules for sentence simplification. 2018.
    </p>
  </li>
  <li>
    <p>
      J. Qiang et. al, 16 Aug, 2019. A Simple BERT-based Approach for Lexical Simplification.
    </p>
  </li>
  <li>
    <p>
      Xingxing Zhang and Mirella Lapata. Sentence simplification with deep reinforcement learning, 2017.
    </p>
  </li>
  <li>
    <p>
      Raman Chandrasekar and Bangalore Srinivas. 1997. Automatic induction of rules for text simplification. In Knowledge Based Systems.
    </p>
  </li>
  <li>
    <p>
      Akhilesh Sudhakar, Bhargav Upadhyay, and Arjun Maheswaran. Transforming delete, retrieve, generate approach for controlled text style transfer, 2019.
    </p>
  </li>
</ul>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<br>
✤ <i> This was the part of mini project for my fifth semester of Computer Science, at Deerwalk College. </a><i>