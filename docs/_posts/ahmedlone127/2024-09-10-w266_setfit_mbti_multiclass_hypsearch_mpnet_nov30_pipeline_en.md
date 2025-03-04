---
layout: model
title: English w266_setfit_mbti_multiclass_hypsearch_mpnet_nov30_pipeline pipeline MPNetEmbeddings from shrinivasbjoshi
author: John Snow Labs
name: w266_setfit_mbti_multiclass_hypsearch_mpnet_nov30_pipeline
date: 2024-09-10
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MPNetEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`w266_setfit_mbti_multiclass_hypsearch_mpnet_nov30_pipeline` is a English model originally trained by shrinivasbjoshi.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/w266_setfit_mbti_multiclass_hypsearch_mpnet_nov30_pipeline_en_5.5.0_3.0_1725995240193.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/w266_setfit_mbti_multiclass_hypsearch_mpnet_nov30_pipeline_en_5.5.0_3.0_1725995240193.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("w266_setfit_mbti_multiclass_hypsearch_mpnet_nov30_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("w266_setfit_mbti_multiclass_hypsearch_mpnet_nov30_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|w266_setfit_mbti_multiclass_hypsearch_mpnet_nov30_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|407.3 MB|

## References

https://huggingface.co/shrinivasbjoshi/w266-setfit-mbti-multiclass-hypsearch-mpnet-nov30

## Included Models

- DocumentAssembler
- MPNetEmbeddings