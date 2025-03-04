---
layout: model
title: Russian RoBERTa Embeddings
author: John Snow Labs
name: roberta_embeddings_ruRoberta_large
date: 2024-09-18
tags: [roberta, embeddings, ru, open_source, onnx]
task: Embeddings
language: ru
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaEmbeddings
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBERTa Embeddings model, uploaded to Hugging Face, adapted and imported into Spark NLP. `ruRoberta-large` is a Russian model orginally trained by `sberbank-ai`.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_embeddings_ruRoberta_large_ru_5.5.0_3.0_1726651837817.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_embeddings_ruRoberta_large_ru_5.5.0_3.0_1726651837817.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_ruRoberta_large","ru") \
.setInputCols(["document", "token"]) \
.setOutputCol("embeddings")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, embeddings])

data = spark.createDataFrame([["Я люблю искра NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val tokenizer = new Tokenizer() 
.setInputCols(Array("document"))
.setOutputCol("token")

val embeddings = RoBertaEmbeddings.pretrained("roberta_embeddings_ruRoberta_large","ru") 
.setInputCols(Array("document", "token")) 
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, embeddings))

val data = Seq("Я люблю искра NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("ru.embed.ruRoberta_large").predict("""Я люблю искра NLP""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_embeddings_ruRoberta_large|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[roberta]|
|Language:|ru|
|Size:|1.3 GB|

## References

References

- https://huggingface.co/sberbank-ai/ruRoberta-large
- https://sberdevices.ru/