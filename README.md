# pykeen
pykeen은 저희가 사용한 Knowledge Graph Embedding 라이브러리입니다.
저희는 pykeen이 제공하는 모델 중 TransE를 선택했습니다.
TransE를 선택한 이유는 모델 파라미터가 적기 때문에, 데이터셋이 부족한 환경에서도 오버피팅이 일어날 가능성이 적기 때문입니다.

학습을 시키기에 앞서, 
zscoreCutter.py로 z-score를 기준으로 dataset 중 드물게 나타나는 entity를 제거하고,
trainingAndTestSplitter.py로 training, valid, test dataset을 분리했습니다.

solution.py에서 학습을 시키고 prediction 결과를 csv 파일로 저장했습니다.

코드를 읽기 전에 https://pykeen.readthedocs.io/en/stable 를 읽어보시는 것을 추천드립니다.