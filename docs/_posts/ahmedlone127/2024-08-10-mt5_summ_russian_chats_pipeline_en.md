---
layout: model
title: English mt5_summ_russian_chats_pipeline pipeline T5Transformer from marlechka
author: John Snow Labs
name: mt5_summ_russian_chats_pipeline
date: 2024-08-10
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mt5_summ_russian_chats_pipeline` is a English model originally trained by marlechka.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mt5_summ_russian_chats_pipeline_en_5.4.2_3.0_1723317553441.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mt5_summ_russian_chats_pipeline_en_5.4.2_3.0_1723317553441.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mt5_summ_russian_chats_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mt5_summ_russian_chats_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mt5_summ_russian_chats_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|2.2 GB|

## References

https://huggingface.co/marlechka/mt5_summ_ru_chats

## Included Models

- DocumentAssembler
- T5Transformer