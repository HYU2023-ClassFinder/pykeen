from pykeen.triples import TriplesFactory
# C:/Users/SAMSUNG/Desktop/gp/pykeen/doctests/test_pre_stratified_transe/training_triples
training_path = TriplesFactory.from_path("C:/Users/SAMSUNG/Desktop/gp/pykeen/doctests")

training_triples_factory = training_path

# Pick a model
from pykeen.models import TransE
model = TransE(triples_factory=training_triples_factory)

# Pick an optimizer from Torch
from torch.optim import Adam
optimizer = Adam(params=model.get_grad_params())

# Pick a training approach (sLCWA or LCWA)
from pykeen.training import SLCWATrainingLoop
training_loop = SLCWATrainingLoop(model=model, optimizer=optimizer)

# Train like Cristiano Ronaldo
training_loop.train(num_epochs=50, batch_size=256)

# Pick an evaluator
from pykeen.evaluation import RankBasedEvaluator
evaluator = RankBasedEvaluator(model)

# Get triples to test
mapped_triples = training_triples_factory = "C:/Users/SAMSUNG/Desktop/gp/pykeen/src/pykeen/datasets/CS/test.txt"

# Evaluate
results = evaluator.evaluate(mapped_triples, batch_size=1024)
print(results)