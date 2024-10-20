---
layout: model
title: Korean ElectraForSequenceClassification Base Cased model (from JminJ)
author: John Snow Labs
name: electra_classifier_kc_base_bad_sentence
date: 2023-11-01
tags: [ko, open_source, electra, sequence_classification, classification, onnx]
task: Text Classification
language: ko
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained ElectraForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `kcElectra_base_Bad_Sentence_Classifier` is a Korean model originally trained by `JminJ`.

## Predicted Entities

`bad_sen`, `ok_sen`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_classifier_kc_base_bad_sentence_ko_5.1.4_3.4_1698805681680.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/electra_classifier_kc_base_bad_sentence_ko_5.1.4_3.4_1698805681680.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("electra_classifier_kc_base_bad_sentence","ko") \
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

val seq_classifier = BertForSequenceClassification.pretrained("electra_classifier_kc_base_bad_sentence","ko")
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("ko.classify.electra.base.kc.by_jminj").predict("""PUT YOUR STRING HERE""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_classifier_kc_base_bad_sentence|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|ko|
|Size:|466.8 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

References

- https://huggingface.co/JminJ/kcElectra_base_Bad_Sentence_Classifier
- https://github.com/smilegate-ai/korean_unsmile_dataset
- https://github.com/kocohub/korean-hate-speech
- https://github.com/Beomi/KcELECTRA
- https://github.com/monologg/KoELECTRA
- https://github.com/JminJ/Bad_text_classifier
- https://github.com/Beomi/KcELECTRA
- https://github.com/monologg/KoELECTRA
- https://github.com/smilegate-ai/korean_unsmile_dataset
- https://github.com/kocohub/korean-hate-speech
- https://arxiv.org/abs/2003.10555