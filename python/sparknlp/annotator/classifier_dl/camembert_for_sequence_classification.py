#  Copyright 2017-2022 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Contains classes for CamemBertForSequenceClassification."""

from sparknlp.common import *


class CamemBertForSequenceClassification(AnnotatorModel,
                                         HasCaseSensitiveProperties,
                                         HasBatchedAnnotate,
                                         HasClassifierActivationProperties,
                                         HasEngine):
    """CamemBertForSequenceClassification can load CamemBERT Models with a sequence
    classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for multi-class document classification tasks.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> sequence_classifier = CamemBertForSequenceClassification.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("label")

    The default model is ``"camembert_base_sequence_classifier_allocine"``, if no
    name is provided.

    For available pretrained models please see the `Models Hub
    <https://nlp.johnsnowlabs.com/models?task=Text+Classification>`__.
    To see which models are compatible and how to import them see
    `Import Transformers into Spark NLP 🚀
    <https://github.com/JohnSnowLabs/spark-nlp/discussions/5669>`_.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT, TOKEN``    ``CATEGORY``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Batch size. Large values allows faster processing but requires more
        memory, by default 8
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default
        True
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    maxSentenceLength
        Max sentence length to process, by default 128
    coalesceSentences
        Instead of 1 class per sentence (if inputCols is `sentence`) output
        1 class per document by averaging probabilities in all sentences, by
        default False.
    activation
        Whether to calculate logits via Softmax or Sigmoid, by default
        `"softmax"`.

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> tokenizer = Tokenizer() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("token")
    >>> sequenceClassifier = CamemBertForSequenceClassification.pretrained() \\
    ...     .setInputCols(["token", "document"]) \\
    ...     .setOutputCol("label") \\
    ...     .setCaseSensitive(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     tokenizer,
    ...     sequenceClassifier
    ... ])
    >>> data = spark.createDataFrame([["j'ai adoré ce film lorsque j'étais enfant.", "Je déteste ça."]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("class.result").show(truncate=False)
    +------+
    |result|
    +------+
    |[pos] |
    |[neg] |
    +------+
    """
    name = "CamemBertForSequenceClassification"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN]

    outputAnnotatorType = AnnotatorType.CATEGORY

    maxSentenceLength = Param(Params._dummy(),
                              "maxSentenceLength",
                              "Max sentence length to process",
                              typeConverter=TypeConverters.toInt)

    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    coalesceSentences = Param(Params._dummy(), "coalesceSentences",
                              "Instead of 1 class per sentence (if inputCols is '''sentence''') output 1 class per document by averaging probabilities in all sentences.",
                              TypeConverters.toBoolean)

    def getClasses(self):
        """
        Returns labels used to train this model
        """
        return self._call_java("getClasses")

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMaxSentenceLength(self, value):
        """Sets max sentence length to process, by default 128.

        Parameters
        ----------
        value : int
            Max sentence length to process
        """
        return self._set(maxSentenceLength=value)

    def setCoalesceSentences(self, value):
        """Instead of 1 class per sentence (if inputCols is '''sentence''') output 1
        class per document by averaging probabilities in all sentences, by default True.

        Due to max sequence length limit in almost all transformer models such as BERT
        (512 tokens), this parameter helps feeding all the sentences into the model and
        averaging all the probabilities for the entire document instead of probabilities
        per sentence.

        Parameters
        ----------
        value : bool
            If the output of all sentences will be averaged to one output
        """
        return self._set(coalesceSentences=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.classifier.dl.CamemBertForSequenceClassification",
                 java_model=None):
        super(CamemBertForSequenceClassification, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=True,
            coalesceSentences=False,
            activation="softmax"
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        CamemBertForSequenceClassification
            The restored model
        """
        from sparknlp.internal import _CamemBertForSequenceClassificationLoader
        jModel = _CamemBertForSequenceClassificationLoader(folder, spark_session._jsparkSession)._java_obj
        return CamemBertForSequenceClassification(java_model=jModel)

    @staticmethod
    def pretrained(name="camembert_base_sequence_classifier_allocine", lang="fr", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default
            "camembert_base_sequence_classifier_allocine"
        lang : str, optional
            Language of the pretrained model, by default "fr"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        CamemBertForSequenceClassification
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(CamemBertForSequenceClassification, name, lang, remote_loc)