---
layout: model
title: English financial Word Embeddings (Roberta, Financial Phrasebank Corpus)
author: John Snow Labs
name: roberta_embeddings_financial
date: 2022-05-04
tags: [roberta, embeddings, en, open_source, financial]
task: Embeddings
language: en
nav_key: models
edition: Spark NLP 3.4.2
spark_version: 3.0
supported: true
annotator: RoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Financial Pretrained Roberta Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `abhilash1910/financial_roberta` is a English Financial model orginally trained upon Financial Phrasebank Corpus.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_financial_en_3.4.2_3.0_1651678861262.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_financial_en_3.4.2_3.0_1651678861262.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("document")

tokenizer = Tokenizer() \
.setInputCols("document") \
.setOutputCol("token")

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_financial","en") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["I Love Spark-NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
  .setInputCol("text") 
  .setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_financial","en") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("I Love Spark-NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_financial|
|Compatibility:|Spark NLP 3.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|en|
|Size:|324.5 MB|
|Case sensitive:|true|