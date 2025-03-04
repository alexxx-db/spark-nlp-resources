---
layout: model
title: English deberta_embeddings_m5_mlm_pipeline pipeline DeBertaEmbeddings from S2312dal
author: John Snow Labs
name: deberta_embeddings_m5_mlm_pipeline
date: 2024-09-02
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

Pretrained DeBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deberta_embeddings_m5_mlm_pipeline` is a English model originally trained by S2312dal.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_embeddings_m5_mlm_pipeline_en_5.5.0_3.0_1725261128263.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_embeddings_m5_mlm_pipeline_en_5.5.0_3.0_1725261128263.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("deberta_embeddings_m5_mlm_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("deberta_embeddings_m5_mlm_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_embeddings_m5_mlm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|689.7 MB|

## References

https://huggingface.co/S2312dal/M5_MLM

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaEmbeddings