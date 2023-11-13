from pykeen.datasets import FB15k237
from pykeen.evaluation import ClassificationEvaluator
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import *

num = "zscoreUpper40"

from pykeen.pipeline import pipeline
from pykeen.datasets import get_dataset
from pykeen.datasets import PathDataset
from pykeen.predict import predict_target

myDataset = PathDataset(
    training_path="src/pykeen/datasets/CS/train_v" + str(num) + ".txt",
    testing_path="src/pykeen/datasets/CS/test_v" + str(num) + ".txt",
    validation_path="src/pykeen/datasets/CS/valid_v" + str(num) + ".txt",
    eager=True)

# Define model
model = TransE(
    triples_factory=myDataset.training,
)

# Train your model (code is omitted for brevity)
# result = pipeline(dataset=myDataset, model=model, lr_scheduler_kwargs=dict(gamma=0.0916875853238321), training_kwargs=dict(num_epochs=15, batch_size=128))
result = pipeline(dataset=myDataset, model=model, training_kwargs=dict(num_epochs=80, batch_size=16))

# Define evaluator
evaluator = RankBasedEvaluator()

# Evaluate your model with not only testing triples,
# but also filter on validation triples
results = evaluator.evaluate(
    model=model,
    mapped_triples=myDataset.testing.mapped_triples,
    additional_filter_triples=[
        myDataset.training.mapped_triples,
        myDataset.validation.mapped_triples,
    ],
)

# ClassificationEvaluator metric
# print("accuracy_score", results.get_metric("accuracy_score"))
# print("average_precision_score", results.get_metric("average_precision_score"))

# RankBasedEvaluator metric
print("hits_at_3", results.get_metric("hits@3"))
print("hits_at_5", results.get_metric("hits@5"))
print("hits_at_10", results.get_metric("hits_at_10"))