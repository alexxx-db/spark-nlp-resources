---
layout: model
title: Mongolian whisper_small_mongolian_12_pipeline pipeline WhisperForCTC from bayartsogt
author: John Snow Labs
name: whisper_small_mongolian_12_pipeline
date: 2024-09-12
tags: [mn, open_source, pipeline, onnx]
task: Automatic Speech Recognition
language: mn
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained WhisperForCTC, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`whisper_small_mongolian_12_pipeline` is a Mongolian model originally trained by bayartsogt.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/whisper_small_mongolian_12_pipeline_mn_5.5.0_3.0_1726151376502.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/whisper_small_mongolian_12_pipeline_mn_5.5.0_3.0_1726151376502.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("whisper_small_mongolian_12_pipeline", lang = "mn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("whisper_small_mongolian_12_pipeline", lang = "mn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|whisper_small_mongolian_12_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|mn|
|Size:|1.7 GB|

## References

https://huggingface.co/bayartsogt/whisper-small-mn-12

## Included Models

- AudioAssembler
- WhisperForCTC