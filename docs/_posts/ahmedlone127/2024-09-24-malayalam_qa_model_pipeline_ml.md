---
layout: model
title: Malayalam malayalam_qa_model_pipeline pipeline BertForQuestionAnswering from Anitha2020
author: John Snow Labs
name: malayalam_qa_model_pipeline
date: 2024-09-24
tags: [ml, open_source, pipeline, onnx]
task: Question Answering
language: ml
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`malayalam_qa_model_pipeline` is a Malayalam model originally trained by Anitha2020.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/malayalam_qa_model_pipeline_ml_5.5.0_3.0_1727163232993.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/malayalam_qa_model_pipeline_ml_5.5.0_3.0_1727163232993.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("malayalam_qa_model_pipeline", lang = "ml")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("malayalam_qa_model_pipeline", lang = "ml")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|malayalam_qa_model_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ml|
|Size:|890.5 MB|

## References

https://huggingface.co/Anitha2020/Malayalam_QA_model

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering