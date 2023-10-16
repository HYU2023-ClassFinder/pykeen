import json
from pykeen.hpo import hpo_pipeline
from pykeen.hpo import hpo_pipeline_from_path
from pykeen.datasets import PathDataset

num = "zscoreUpper40"

myDataset = PathDataset(
    training_path="src/pykeen/datasets/CS/train_v" + str(num) + ".txt",
    testing_path="src/pykeen/datasets/CS/test_v" + str(num) + ".txt",
    validation_path="src/pykeen/datasets/CS/valid_v" + str(num) + ".txt",
    eager=True)

hpo_pipeline_result = hpo_pipeline(
    n_trials=10000,
    dataset=myDataset,
    model='TransE',
    model_kwargs=dict(embedding_dim=20, scoring_fct_norm=1),
    optimizer='SGD',
    optimizer_kwargs=dict(lr=0.01),
    lr_scheduler='ExponentialLR',
    lr_scheduler_kwargs_ranges=dict(
        gamma=dict(type=float, low=0.001, high=1.0, log=True),
    ),
    loss='CrossEntropyLoss',
    training_loop='slcwa',
    training_kwargs_ranges=dict(
        num_epochs=dict(type=int, low=1, high=100, step=1), 
        batch_size=dict(type=int, scale='power', base=2, low=4, high=7)
    ),
    negative_sampler='basic',
    negative_sampler_kwargs_ranges=dict(
        num_negs_per_pos=dict(type=int, low=1, high=100),
    ),
    evaluator_kwargs=dict(filtered=True),
    evaluation_kwargs=dict(batch_size=128),
    stopper='early',
    stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
)

hpo_pipeline_result.save_to_directory('')