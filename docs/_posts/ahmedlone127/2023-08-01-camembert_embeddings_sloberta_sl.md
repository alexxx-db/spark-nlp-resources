---
layout: model
title: Slovenian CamemBert Embeddings (from EMBEDDIA)
author: John Snow Labs
name: camembert_embeddings_sloberta
date: 2023-08-01
tags: [sl, open_source, camembert, embeddings, onnx]
task: Embeddings
language: sl
edition: Spark NLP 5.0.2
spark_version: 3.0
supported: true
engine: onnx
annotator: CamemBertEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBert Embeddings model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `sloberta` is a Slovenian model orginally trained by `EMBEDDIA`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_embeddings_sloberta_sl_5.0.2_3.0_1690926104653.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/camembert_embeddings_sloberta_sl_5.0.2_3.0_1690926104653.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = CamemBertEmbeddings.pretrained("camembert_embeddings_sloberta","sl") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Obožujem Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val embeddings = CamemBertEmbeddings.pretrained("camembert_embeddings_sloberta","sl")
    .setInputCols(Array("document", "token"))
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Obožujem Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("sl.embed.camembert").predict("""Obožujem Spark NLP""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_embeddings_sloberta|
|Compatibility:|Spark NLP 5.0.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[embeddings]|
|Language:|sl|
|Size:|263.5 MB|
|Case sensitive:|true|