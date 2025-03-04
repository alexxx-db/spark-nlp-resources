---
layout: model
title: Tagalog sent_bert_tagalog_base_cased_wwm_pipeline pipeline BertSentenceEmbeddings from jcblaise
author: John Snow Labs
name: sent_bert_tagalog_base_cased_wwm_pipeline
date: 2024-09-17
tags: [tl, open_source, pipeline, onnx]
task: Embeddings
language: tl
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_bert_tagalog_base_cased_wwm_pipeline` is a Tagalog model originally trained by jcblaise.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_bert_tagalog_base_cased_wwm_pipeline_tl_5.5.0_3.0_1726588105608.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_bert_tagalog_base_cased_wwm_pipeline_tl_5.5.0_3.0_1726588105608.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_bert_tagalog_base_cased_wwm_pipeline", lang = "tl")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_bert_tagalog_base_cased_wwm_pipeline", lang = "tl")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_bert_tagalog_base_cased_wwm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|tl|
|Size:|407.4 MB|

## References

https://huggingface.co/jcblaise/bert-tagalog-base-cased-WWM

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- BertSentenceEmbeddings