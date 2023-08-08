# -*- coding: utf-8 -*-

filename = "tagClaims_soohan_v1"

"""The DBpedia datasets from [shi2017b]_.

- GitHub Repository: https://github.com/bxshi/ConMask
- Paper: https://arxiv.org/abs/1711.03438
"""

# from docdata import parse_docdata

# from pykeen.datasets.base import UnpackedRemoteDataset

# __all__ = [
#     "DBpedia50",
# ]

# BASE = "https://raw.githubusercontent.com/ZhenfengLei/KGDatasets/master/DBpedia50"
# TEST_URL = f"{BASE}/test.txt"
# TRAIN_URL = f"{BASE}/train.txt"
# VALID_URL = f"{BASE}/valid.txt"


# # [docs]@parse_docdata
# class DBpedia50(UnpackedRemoteDataset):
#     """The DBpedia50 dataset.

#     ---
#     name: DBpedia50
#     citation:
#         author: Shi
#         year: 2017
#         link: https://arxiv.org/abs/1711.03438
#     statistics:
#         entities: 24624
#         relations: 351
#         training: 32203
#         testing: 2095
#         validation: 123
#         triples: 34421
#     """

#     def __init__(self, **kwargs):
#         """Initialize the DBpedia50 small dataset from [shi2017b]_.

#         :param kwargs: keyword arguments passed to :class:`pykeen.datasets.base.UnpackedRemoteDataset`.
#         """
#         super().__init__(
#             training_url=TRAIN_URL,
#             testing_url=TEST_URL,
#             validation_url=VALID_URL,
#             load_triples_kwargs={
#                 # as pointed out in https://github.com/pykeen/pykeen/issues/275#issuecomment-776412294,
#                 # the columns are not ordered properly.
#                 "column_remapping": [0, 2, 1],
#             },
#             **kwargs,
#         )

# if __name__ == "__main__":
#     # DBpedia50.cli()

#     from pykeen.pipeline import pipeline
#     result = pipeline(dataset="nations", model="pairre", training_kwargs=dict(num_epochs=0))

#     from pykeen.datasets import get_dataset
#     from pykeen.predict import predict_target
#     dataset = get_dataset(dataset="nations")
#     pred = predict_target(
#         model=result.model,
#         head="uk",
#         relation="embassy",
#         triples_factory=result.training,
#     )
#     pred_filtered = pred.filter_triples(dataset.training)
#     pred_annotated = pred_filtered.add_membership_columns(validation=dataset.validation, testing=dataset.testing)

#     print(pred_annotated.df)

if __name__ == "__main__":
    # DBpedia50.cli()

    from pykeen.pipeline import pipeline, PipelineResult
    

    from pykeen.datasets import get_dataset
    from pykeen.datasets import PathDataset
    from pykeen.predict import predict_target
    from pandas import DataFrame, Series
    from typing import Iterable, Tuple, Hashable
    from typing_extensions import TypeAlias

    Label: TypeAlias = Hashable | None

    myDataset = PathDataset(
        training_path="src/pykeen/datasets/CS/"+filename+"_train.txt",
        testing_path="src/pykeen/datasets/CS/"+filename+"_test.txt",
        validation_path="src/pykeen/datasets/CS/"+filename+"_valid.txt",
        eager=True,
        create_inverse_triples=True)

    # _head = "Q19399674"
    heads = map(lambda x: x.replace('\n', ''), open(filename+"_heads.txt").readlines())
    output_links = open(filename+"_result.csv", "w")
    output_highest_candidates = open(filename+"_highest_candidates.csv", "w")
    output_threshold_050 = open(filename+"_threshold_0.50.csv", "w")
    output_threshold_075 = open(filename+"_threshold_0.75.csv", "w")
    output_threshold_100 = open(filename+"_threshold_1.00.csv", "w")
    output_threshold_125 = open(filename+"_threshold_1.25.csv", "w")
    output_threshold_150 = open(filename+"_threshold_1.50.csv", "w")
    output_loss = open(filename+"_loss.txt", "w")

    output_links.write("idx,row,score,head,tail_label,in_validation,in_testing\n")
    output_highest_candidates.write("idx,row,score,head,tail_label,in_validation,in_testing\n")
    output_threshold_050.write("idx,row,score,head,tail_label,in_validation,in_testing\n")
    output_threshold_075.write("idx,row,score,head,tail_label,in_validation,in_testing\n")
    output_threshold_100.write("idx,row,score,head,tail_label,in_validation,in_testing\n")
    output_threshold_125.write("idx,row,score,head,tail_label,in_validation,in_testing\n")
    output_threshold_150.write("idx,row,score,head,tail_label,in_validation,in_testing\n")
    # _relation = ["P361_part_of", "P279_subclass_of", "P2579_studied_by", "P366_has_use", "P2283_uses"]
    _relation = ["preceded_by"]


    for _rela in _relation:
        result: PipelineResult = pipeline(dataset=myDataset, model="pairre", training_kwargs=dict(num_epochs=100))
        print(result.losses)
        output_loss.write(str(result.losses))
        for head in heads:
            try:
                print("head: "+head)
                pred = predict_target(
                    model=result.model,
                    head=head,
                    relation=_rela,
                    triples_factory=result.training,
                )
                pred_filtered = pred.filter_triples(myDataset.training)
                pred_annotated = pred_filtered.add_membership_columns(validation=myDataset.validation, testing=myDataset.testing)

                # print(pred_annotated.df)

                csdf: DataFrame = pred_annotated.df

                print("==== LINK CANDIDATES ====")
                print(f"{'idx':>6} {'tail_id':>8} {'score':>16} {'head':>24} {'tail_label':>24} {'in_validation':>16} {'in_testing':>16}")
                rows: Iterable[Tuple[Label, Series]] = csdf.iterrows()
                for i in range(min(5, csdf.shape[0])):
                    (idx, row) = next(rows)
                    print(f"{idx:>6} {row['tail_id']:>8} {row['score']:>16} {head:>24} {row['tail_label']:>24} {row['in_validation']:>16} {row['in_testing']:>16}")
                    output_highest_candidates.write(f"{idx},{row['tail_id']},{row['score']},{head},{row['tail_label']},{row['in_validation']},{row['in_testing']}\n")
                
                print("==== LINKS PREDICTED ====")
                print(f"{'idx':>6} {'tail_id':>8} {'score':>16} {'head':>24} {'tail_label':>24} {'in_validation':>16} {'in_testing':>16}")
                for idx, row in csdf.iterrows():
                    if(row['score'] >= -0.5):
                        output_threshold_050.write(f"{idx},{row['tail_id']},{row['score']},{head},{row['tail_label']},{row['in_validation']},{row['in_testing']}\n")
                    if(row['score'] >= -0.75):
                        output_threshold_075.write(f"{idx},{row['tail_id']},{row['score']},{head},{row['tail_label']},{row['in_validation']},{row['in_testing']}\n")
                    if(row['score'] >= -1.0):
                        output_threshold_100.write(f"{idx},{row['tail_id']},{row['score']},{head},{row['tail_label']},{row['in_validation']},{row['in_testing']}\n")
                    if(row['score'] >= -1.25):
                        output_threshold_125.write(f"{idx},{row['tail_id']},{row['score']},{head},{row['tail_label']},{row['in_validation']},{row['in_testing']}\n")
                    if(row['score'] >= -1.5):
                        output_threshold_150.write(f"{idx},{row['tail_id']},{row['score']},{head},{row['tail_label']},{row['in_validation']},{row['in_testing']}\n")
                    if(row['in_validation'] == True or row['in_testing'] == True):
                        # print(idx, row)
                        print(f"{idx:>6} {row['tail_id']:>8} {row['score']:>16} {head:>24} {row['tail_label']:>24} {row['in_validation']:>16} {row['in_testing']:>16}")
                        output_links.write(f"{idx},{row['tail_id']},{row['score']},{head},{row['tail_label']},{row['in_validation']},{row['in_testing']}\n")
            except KeyError:
                print("KeyError for head "+head)
                # print(idx, row)
            
            # csdf.to_csv(_head+"_"+_rela+".txt")
# from pykeen.triples import TriplesFactory
# from pykeen.pipeline import pipeline
# from pykeen.datasets.nations import NATIONS_TRAIN_PATH, NATIONS_TEST_PATH
# training = TriplesFactory.from_path(
#     "C:/Users/SAMSUNG/Desktop/gp/pykeen/src/pykeen/datasets/CS/train.txt",
#     create_inverse_triples=True,
# )
# testing = TriplesFactory.from_path(
#     "C:/Users/SAMSUNG/Desktop/gp/pykeen/src/pykeen/datasets/CS/test.txt",
#     entity_to_id=training.entity_to_id,
#     relation_to_id=training.relation_to_id,
#     create_inverse_triples=True,
# )
# result = pipeline(
#     training=training,
#     testing=testing,
#     model='TransE',
#     epochs=5,  # short epochs for testing - you should go higher
# )
# result.save_to_directory('doctests/test_pre_stratified_transe')