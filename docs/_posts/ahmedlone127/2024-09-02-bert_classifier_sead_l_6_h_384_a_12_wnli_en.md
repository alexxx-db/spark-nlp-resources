---
layout: model
title: English BertForSequenceClassification Cased model (from course5i)
author: John Snow Labs
name: bert_classifier_sead_l_6_h_384_a_12_wnli
date: 2024-09-02
tags: [en, open_source, bert, sequence_classification, classification, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `SEAD-L-6_H-384_A-12-wnli` is a English model originally trained by `course5i`.

## Predicted Entities

`0`, `1`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_sead_l_6_h_384_a_12_wnli_en_5.5.0_3.0_1725293789811.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_sead_l_6_h_384_a_12_wnli_en_5.5.0_3.0_1725293789811.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_sead_l_6_h_384_a_12_wnli","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
      .setInputCols(Array("text"))
      .setOutputCols(Array("document"))

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_sead_l_6_h_384_a_12_wnli","en")
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.classify.bert.wnli_glue.6l_384d_a12a").predict("""PUT YOUR STRING HERE""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_sead_l_6_h_384_a_12_wnli|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|84.2 MB|

## References

References

- https://huggingface.co/course5i/SEAD-L-6_H-384_A-12-wnli
- https://arxiv.org/abs/1910.01108
- https://arxiv.org/abs/1909.10351
- https://arxiv.org/abs/2002.10957
- https://arxiv.org/abs/1810.04805
- https://arxiv.org/abs/1804.07461
- https://arxiv.org/abs/1905.00537
- https://www.adasci.org/journals/lattice-35309407/?volumes=true&open=621a3b18edc4364e8a96cb63