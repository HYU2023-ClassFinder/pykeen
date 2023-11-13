# -*- coding: utf-8 -*-
'''
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
'''

# solution.py에서는 학습을 시키고 prediction 결과를 csv 파일로 저장합니다.
# 100번의 예측을 반복해, 그 평균 점수와 100회의 예측 중 몇 번 등장했는지를 기록합니다.
# 가능한 모든 head에 대해서 반복합니다.

from collections import Counter
import itertools

if __name__ == "__main__":
    # DBpedia50.cli()

    from pykeen.pipeline import pipeline
    

    from pykeen.datasets import get_dataset
    from pykeen.datasets import PathDataset
    from pykeen.predict import predict_target
    
    # zscoreUpper40 dataset을 사용합니다.
    num = "zscoreUpper40"
    myDataset = PathDataset(
        training_path="src/pykeen/datasets/CS/train_v" + str(num) + ".txt",
        testing_path="src/pykeen/datasets/CS/test_v" + str(num) + ".txt",
        validation_path="src/pykeen/datasets/CS/valid_v" + str(num) + ".txt",
        eager=True,
        create_inverse_triples = False)

    # head는 저희가 수집한 강의 데이터셋의 태그 중, head로 쓰일 수 있는 tag를 모은 것입니다.
    head = ['Q3050461', 'Q1936256', 'Q11205', 'Q28865', 'Q23808', 'Q848010', 'Q132364', 'Q1427251', 'Q9143', 'Q1635410', 'Q208163', 'Q11476', 'Q845566', 'Q2493', 'Q1301371', 'Q176645', 'Q3510521', 'Q101333', 'Q1051282', 'Q2374463', 'Q857102', 'Q163310', 'Q116777014', 'Q2321565', 'Q1070689', 'Q373045', 'Q1152135', 'Q833585', 'Q395', 'Q179976', 'Q189436', 'Q478798', 'Q796212', 'Q208042', 'Q205084', 'Q212108', 'Q5300', 'Q37437', 'Q80006', 'Q131476', 'Q211496', 'Q13649246', 'Q1363085', 'Q638608', 'Q2878974', 'Q15777', 'Q21198', 'Q8078', 'Q43260', 'Q1026367', 'Q844240', 'Q1034415', 'Q1141518', 'Q219320', 'Q7397', 'Q483130', 'Q232661', 'Q319400', 'Q272683', 'Q50423863', 'Q827335', 'Q9135', 'Q12483', 'Q117879', 'Q121416', 'Q729271', 'Q7600677', 'Q79798', 'Q3968', 'Q938438', 'Q752532', 'Q378859', 'Q85810444', 'Q839721', 'Q200125', 'Q192776', 'Q583461', 'Q178354', 'Q4479242', 'Q1027879', 'Q9492']   
    # _relation = ["P361_part_of", "P279_subclass_of", "P2579_studied_by", "P366_has_use", "P2283_uses"]
    
    # relation은 is_preceded_by로, head와 저희가 예측하려는 tail간의 relationship을 의미합니다.
    relation = "is_preceded_by"
    '''
    # _model = ["AutoSF",
    # "BoxE",
    # "CompGCN",
    # "ComplEx",
    # # "ComplExLiteral", #AttributeError: 'TriplesFactory' object has no attribute 'literal_shape'
    # # "ConvE", # too slow
    # # "ConvKB", # too slow
    # "CP",
    # "CrossE",
    # "DistMA",
    # "DistMult",
    # # "DistMultLiteral", #AttributeError: 'TriplesFactory' object has no attribute 'literal_shape'
    # # "DistMultLiteralGated", #AttributeError: 'TriplesFactory' object has no attribute 'literal_shape'
    # "ERMLP",
    # "ERMLPE",
    # "HolE",
    # "KG2E",
    # "FixedModel",
    # "MuRE",
    # "NodePiece",
    # "NTN",
    # "PairRE",
    # "ProjE",
    # "QuatE",
    # "RESCAL",
    # # "RGCN", #ValueError: RGCN internally creates inverse triples. It thus expects a triples factory without them.
    # "RotatE",
    # "SimplE",
    # "SE",
    # "TorusE",
    # "TransD",
    # "TransE",
    # "TransF",
    # "TransH",
    # "TransR",
    # "TuckER",
    # "UM",]
    '''

    # 모델은 최대한 단순한 모델인 TransE를 사용합니다.
    model = ["TransE"]
    TRY = 100

    for _head in head:
        trueRelations = []
        prePredictedTail = []

        try:
            for i in list(range(TRY)):
                for _model in model:
                    print("_____________________________________________________________")
                    print(_model + " " + _head + " " + str(i))
                    print("_____________________________________________________________")
                    result = pipeline(dataset=myDataset, 
                                    model=_model, 
                                    optimizer_kwargs=dict(lr=0.01), 
                                    training_kwargs=dict(num_epochs=80, batch_size=16))
                    
                    pred = predict_target(
                        model=result.model,
                        head=_head,
                        relation=relation,
                        triples_factory=result.training,
                    )

                pred_filtered = pred.filter_triples(myDataset.training)
                pred_annotated = pred_filtered.add_membership_columns(validation=myDataset.validation, testing=myDataset.testing)
                '''
                # print(pred_annotated.df.nlargest(20, 'score'))

                # csdf = pred_annotated.df
                
                # for idx, row in csdf.iterrows():
                #     if(row['in_validation'] == True or row['in_testing'] == True):
                #         print(idx, row)
                #         trueRelations.append([idx, row])
                
                # csdf.to_csv("results/"+_head+"_"+_model+"_"+relation+".txt")
                
                # f = open(_head+"_"+relation+".txt", "w")
                # for trueRelation in trueRelations:
                #     f.write(str(trueRelation[0]) + '\n' + str(trueRelation[1]))
                # f.close()

                # f2 = open("summary_"+_head+"_"+relation+".txt", "w")
                # tailId = []
                # for trueRelation in trueRelations:
                #     tailId.append(trueRelation[1]["tail_label"])
                # tailIdSet = list(set(tailId))

                # for tail in tailIdSet:
                #     f2.write(tail + ":  " + str(tailId.count(tail)) + "/" + str(len(model)) + "\n")
                # f2.close()
                '''
                for idx, row in pred_annotated.df.nlargest(20, 'score').iterrows():
                    prePredictedTail.append((row['tail_label'], row['score']))

            temp = []
            # predictedTail의 첫 번째 원소가 tail entity, 두 번째 원소가 평균 score, 세 번째 원소가 등장 횟수입니다.
            predictedTail = []
            for p in prePredictedTail:
                if(p[0] not in list(itertools.chain(*predictedTail))):
                    predictedTail.append([p[0], p[1], 1])
                    temp.append(p[0])
                else:
                    predictedTail[temp.index(p[0])][2] = predictedTail[temp.index(p[0])][2]+1
                    predictedTail[temp.index(p[0])][1] = predictedTail[temp.index(p[0])][2]+p[1]
                
            for p in predictedTail:
                p[1] = p[1] / p[2]

            predictedTail.sort(key=lambda x : -x[2])
            
            f3 = open("predictedTail/" + _head+"_"+relation+"_predictedTail_try" + str(TRY) + "_using_" + _model + ".txt", "w")
            for p in predictedTail:
                f3.write(str(p[0]) + '\t' + str(p[1]) + '\t' + str(p[2]) + '\t' + "\n")
            f3.close()
        
        except KeyError:
            continue

    # from pykeen import predict
    # model = "PairRE"
    # result = pipeline(dataset=myDataset, model=model, training_kwargs=dict(num_epochs=10))
    # pred = predict.predict_all(model=result.model).process()
    # pred_filtered = pred.filter_triples(myDataset.training)
    # pred_annotated = pred_filtered.add_membership_columns(validation=myDataset.validation, testing=myDataset.testing)
    # print(pred_annotated.df)

    # csdf = pred_annotated.df

    # for idx, row in csdf.iterrows():
    #     if(row['in_validation'] == True or row['in_testing'] == True):
    #         print(idx, row)
    #         trueRelations.append([idx, row])

    # csdf.to_csv("results/all.txt")

    # f = open("all_"+_head+"_"+_relation+".txt", "w")
    # for trueRelation in trueRelations:
    #     f.write(str(trueRelation[0]) + '\n' + str(trueRelation[1]))
    # f.close()

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