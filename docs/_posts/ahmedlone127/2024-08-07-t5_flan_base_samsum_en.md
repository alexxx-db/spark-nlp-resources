---
layout: model
title: English T5ForConditionalGeneration Cased model (from philschmid)
author: John Snow Labs
name: t5_flan_base_samsum
date: 2024-08-07
tags: [open_source, t5, flan, en, onnx]
task: Text Generation
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
engine: onnx
annotator: T5Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5ForConditionalGeneration model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. flan-t5-base-samsum is a English model originally trained by philschmid.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/t5_flan_base_samsum_en_5.4.2_3.0_1723032443785.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/t5_flan_base_samsum_en_5.4.2_3.0_1723032443785.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
.setInputCols("text") \
.setOutputCols("document")

t5 = T5Transformer.pretrained("t5_flan_base_samsum","en") \
.setInputCols("document") \
.setOutputCol("answers")

pipeline = Pipeline(stages=[documentAssembler, t5])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
.setInputCols("text")
.setOutputCols("document")

val t5 = T5Transformer.pretrained("t5_flan_base_samsum","en") 
.setInputCols("document")
.setOutputCol("answers")

val pipeline = new Pipeline().setStages(Array(documentAssembler, t5))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|t5_flan_base_samsum|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|en|
|Size:|1.0 GB|

## References

References

https://huggingface.co/philschmid/flan-t5-base-samsum