---
layout: model
title: French camembert_base_finetuned_nli_repnum_wl_rua_wl CamemBertForSequenceClassification from waboucay
author: John Snow Labs
name: camembert_base_finetuned_nli_repnum_wl_rua_wl
date: 2024-01-19
tags: [camembert, fr, open_source, sequence_classification, onnx]
task: Text Classification
language: fr
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: CamemBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`camembert_base_finetuned_nli_repnum_wl_rua_wl` is a French model originally trained by waboucay.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/camembert_base_finetuned_nli_repnum_wl_rua_wl_fr_5.2.4_3.0_1705699198641.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/camembert_base_finetuned_nli_repnum_wl_rua_wl_fr_5.2.4_3.0_1705699198641.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")  
    
sequenceClassifier = CamemBertForSequenceClassification.pretrained("camembert_base_finetuned_nli_repnum_wl_rua_wl","fr")\
            .setInputCols(["document","token"])\
            .setOutputCol("class")

pipeline = Pipeline().setStages([document_assembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)

```
```scala

val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document") 
    .setOutputCol("token")  
    
val sequenceClassifier = CamemBertForSequenceClassification.pretrained("camembert_base_finetuned_nli_repnum_wl_rua_wl","fr")
            .setInputCols(Array("document","token"))
            .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|camembert_base_finetuned_nli_repnum_wl_rua_wl|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|fr|
|Size:|410.0 MB|

## References

https://huggingface.co/waboucay/camembert-base-finetuned-nli-repnum_wl-rua_wl