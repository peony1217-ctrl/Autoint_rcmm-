AutoInt Recommendation System (MovieLens)  
AutoInt 및 AutoInt-MLP 모델을 활용한 MovieLens 기반 추천 시스템 프로젝트입니다. 기존 코드에서 모델정의 변경으로 성능 향상을 시도했습니다.  

AUTOINT/  
│  
├── data/                         # 데이터 및 전처리 결과  
│   ├── ml-1m/                    # MovieLens 1M 원본 데이터  
│   ├── field_dims.npy            # 각 feature별 embedding dimension  
│   ├── ml-1m.npy                 # 모델 학습용 numpy 데이터  
│   ├── movielens_rcmm_v1.csv     # 추천 결과 (version 1) - 사이즈 커서 안올림  
│   ├── movielens_rcmm_v2.csv     # 추천 결과 (version 2) - 사이즈 커서 안올림  
│   ├── movies_prepro.csv         # 영화 데이터 전처리 결과 - 사이즈 커서 안올림  
│   ├── ratings_prepro.csv        # 평점 데이터 전처리 결과 - 사이즈 커서 안올림  
│   ├── users_prepro.csv          # 사용자 데이터 전처리 결과 - 사이즈 커서 안올림  
│  
├── model/                        # 학습된 모델 및 가중치  
│   ├── autoInt_model.keras  
│   ├── autoInt_model_weights.weights.h5  
│   ├── autoIntMLP_model_weights.weights.h5  
│   ├── autoIntMLP1_model_weights.weights.h5 - 튜닝 적용 가중지  
│   ├── label_encoders.pkl  
│   └── label_encoders1.pkl - - 튜닝 적용 가중지  
│  
├── notebook/                     # 실험 및 학습용 노트북  
│   ├── autoint_train.ipynb       # AutoInt 학습  
│   ├── autoint_mlp_train.ipynb   # AutoInt-MLP 학습 - 기존코드에서 튜닝 적용 후 새로운 이름으로 가중치 저장  
│   ├── data_EDA.ipynb            # 데이터 탐색  
│   ├── data_prepro.ipynb         # 데이터 전처리  
│   └── model_load_test.ipynb     # 모델 로딩 테스트  
│  
├── autoint.py                    # AutoInt 모델 정의  
├── autointmlp.py                 # AutoInt-MLP 모델 정의 - 기존코드에서 튜닝 적용  
├── show_st.py                    # Streamlit 추천 UI  
├── show_st_plus.py               # Streamlit 확장 UI - 기존코드에서 튜닝 적용  
├── requirements.txt              # Python 패키지 목록  
└── README.md  

