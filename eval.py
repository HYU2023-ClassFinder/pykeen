from pykeen.datasets import FB15k237
from pykeen.evaluation import ClassificationEvaluator
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import *

num = "4"

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
model = TransR(
    triples_factory=myDataset.training,
)

f = open("hitRateData7.txt", "a")

for i in list(range(70, 80+1)):
    # Train your model (code is omitted for brevity)
    result = pipeline(dataset=myDataset, model=model, training_kwargs=dict(num_epochs=100*i))

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

    # # ClassificationEvaluator metric
    # # print("accuracy_score", results.get_metric("accuracy_score"))
    # # print("average_precision_score", results.get_metric("average_precision_score"))

    # # RankBasedEvaluator metric
    # print("hits_at_3", results.get_metric("hits@3"))
    # print("hits_at_5", results.get_metric("hits@5"))
    # print("hits_at_10", results.get_metric("hits_at_10"))
    print(str(i*100) + " " + str(results.get_metric("hits@3")) + " " + str(results.get_metric("hits@5")) + " " + str(results.get_metric("hits@10")))
    f.write(str(i*100) + " " + str(results.get_metric("hits@3")) + " " + str(results.get_metric("hits@5")) + " " + str(results.get_metric("hits@10")) + "\n")

f.close()