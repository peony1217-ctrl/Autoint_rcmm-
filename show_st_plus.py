# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import os
# import joblib
# from autointmlp import AutoIntMLPModel, predict_model
# from tensorflow.keras.models import load_model

# # ---------------------------
# # 페이지 설정
# # ---------------------------
# st.set_page_config(
#     page_title="🎬 영화 추천 시스템",
#     page_icon="🎥",
#     layout="centered",
# )

# # ---------------------------
# # 커스텀 스타일 적용
# # ---------------------------
# st.markdown(
#     """
#     <style>
#     /* 배경 색상 */
#     .stApp {
#         background-color: #fff9e6;  /* 연노란색 */
#     }
#     /* 제목 폰트 크기 */
#     h1 {
#         font-size: 36px;
#     }
#     h2 {
#         font-size: 28px;
#     }
#     /* 추천 결과 표 스타일 */
#     .dataframe th {
#         background-color: #ffe680;
#         color: #000000;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # ---------------------------
# # 데이터 로드
# # ---------------------------
# @st.cache_resource
# def load_data():
#     project_path = os.path.abspath(os.getcwd())
#     data_dir_nm = 'data'
#     movielens_dir_nm = 'ml-1m'
#     model_dir_nm = 'model'
#     data_path = f"{project_path}/{data_dir_nm}"
#     model_path = f"{project_path}/{model_dir_nm}"
#     field_dims = np.load(f'{data_path}/field_dims.npy')
#     dropout= 0.4
#     embed_dim= 16
    
#     ratings_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/ratings_prepro.csv')
#     movies_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/movies_prepro.csv')
#     user_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/users_prepro.csv')

#     model = AutoIntMLPModel(
#         field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True,
#         dnn_hidden_units=(32, 32), dnn_activation='relu',
#         l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=dropout, init_std=0.0001
#     )
    
#     # 모델 초기화
#     model(tf.constant([[0] * len(field_dims)], dtype=tf.int64))
#     model.load_weights(f'{model_path}/autoIntMLP_model_weights.weights.h5') 
#     label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')
    
#     return user_df, movies_df, ratings_df, model, label_encoders

# # ---------------------------
# # 사용자-영화 데이터 처리 함수
# # ---------------------------
# def get_user_seen_movies(ratings_df):
#     user_seen_movies = ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()
#     return user_seen_movies

# def get_user_non_seed_dict(movies_df, user_df, user_seen_movies):
#     unique_movies = movies_df['movie_id'].unique()
#     unique_users = user_df['user_id'].unique()
#     user_non_seen_dict = dict()

#     for user in unique_users:
#         user_seen_movie_list = user_seen_movies[user_seen_movies['user_id'] == user]['movie_id'].values[0]
#         user_non_seen_movie_list = list(set(unique_movies) - set(user_seen_movie_list))
#         user_non_seen_dict[user] = user_non_seen_movie_list
        
#     return user_non_seen_dict

# def get_user_info(user_id):
#     return users_df[users_df['user_id'] == user_id]

# def get_user_past_interactions(user_id):
#     return ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)].merge(movies_df, on='movie_id')

# def get_recom(user, user_non_seen_dict, user_df, movies_df, r_year, r_month, model, label_encoders):
#     user_non_seen_movie = user_non_seen_dict.get(user)
#     user_id_list = [user for _ in range(len(user_non_seen_movie))]
#     r_decade = str(r_year - (r_year % 10)) + 's'
    
#     user_non_seen_movie = pd.merge(pd.DataFrame({'movie_id':user_non_seen_movie}), movies_df, on='movie_id')
#     user_info = pd.merge(pd.DataFrame({'user_id':user_id_list}), user_df, on='user_id')
#     user_info['rating_year'] = r_year
#     user_info['rating_month'] = r_month
#     user_info['rating_decade'] = r_decade
    
#     merge_data = pd.concat([user_non_seen_movie, user_info], axis=1)
#     merge_data.fillna('no', inplace=True)
#     merge_data = merge_data[['user_id', 'movie_id','movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade', 
#                              'genre1','genre2', 'genre3', 'gender', 'age', 'occupation', 'zip']]
    
#     for col, le in label_encoders.items():
#         merge_data[col] = le.fit_transform(merge_data[col])
    
#     recom_top = predict_model(model, merge_data)
#     recom_top = [r[0] for r in recom_top]
#     origin_m_id = label_encoders['movie_id'].inverse_transform(recom_top)
    
#     return movies_df[movies_df['movie_id'].isin(origin_m_id)]

# # ---------------------------
# # 데이터 준비
# # ---------------------------
# users_df, movies_df, ratings_df, model, label_encoders = load_data()
# user_seen_movies = get_user_seen_movies(ratings_df)
# user_non_seen_dict = get_user_non_seed_dict(movies_df, users_df, user_seen_movies)

# # ---------------------------
# # 상단 타이틀
# # ---------------------------
# st.markdown("## 🎬 영화 추천 결과 살펴보기 🎬", unsafe_allow_html=True)

# # ---------------------------
# # 입력창
# # ---------------------------
# st.header("사용자 정보를 넣어주세요.")
# user_id = st.number_input("👤 사용자 ID 입력", 
#                           min_value=users_df['user_id'].min(), 
#                           max_value=users_df['user_id'].max(), 
#                           value=users_df['user_id'].min())

# r_year = st.number_input("📅 추천 타겟 연도 입력", 
#                          min_value=ratings_df['rating_year'].min(), 
#                          max_value=ratings_df['rating_year'].max(), 
#                          value=ratings_df['rating_year'].min())

# r_month = st.number_input("🗓 추천 타겟 월 입력", 
#                           min_value=ratings_df['rating_month'].min(), 
#                           max_value=ratings_df['rating_month'].max(), 
#                           value=ratings_df['rating_month'].min())

# # ---------------------------
# # 추천 결과 버튼
# # ---------------------------
# if st.button("🍿 추천 결과 보기"):
#     st.subheader("사용자 기본 정보")
#     user_info = get_user_info(user_id)
#     st.dataframe(user_info)

#     st.subheader("사용자가 과거에 봤던 영화 (평점 4점 이상)")
#     user_interactions = get_user_past_interactions(user_id)
#     st.dataframe(user_interactions)

#     st.subheader("추천 결과 🎯")
#     recommendations = get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders)
#     st.dataframe(recommendations)

# 튜닝 후 코드
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib
from autointmlp import AutoIntMLPModel, predict_model

# ---------------------------
# 페이지 설정
# ---------------------------
st.set_page_config(
    page_title="뽀짝 영화 추천 시스템",
    page_icon="🍿",
    layout="centered",
)

# ---------------------------
# 귀염뽀짝 커스텀 스타일 적용
# ---------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #FFFDF5; }
    
    /* 전광판 효과 */
    .marquee {
        background-color: #FF4B4B;
        color: white;
        padding: 10px;
        font-weight: bold;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 25px;
        font-size: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }

    /* 버튼 스타일 */
    div.stButton > button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #FFD700;
        color: #FF4B4B;
        transform: scale(1.02);
    }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# 데이터 및 모델 로드 함수
# ---------------------------
@st.cache_resource
def load_data():
    project_path = r"C:\Users\Admin\autoint"
    data_path = os.path.join(project_path, 'data')
    weights_path = r"C:\Users\Admin\autoint\model\autoIntMLP1_model_weights.weights.h5"
    encoder_path = r"C:\Users\Admin\autoint\model\label_encoders1.pkl"
    
    field_dims = np.load(os.path.join(data_path, 'field_dims.npy'))
    embed_dim = 32
    
    model = AutoIntMLPModel(
        field_dims=field_dims,
        embedding_size=embed_dim,
        att_layer_num=3,
        att_head_num=4,
        att_res=True,
        dnn_hidden_units=(256, 128, 64),
        dnn_activation='relu',
        dnn_use_bn=True,
        dnn_dropout=0.2
    )
    
    # 모델 빌드 (더미 데이터)
    model(tf.constant([[0] * len(field_dims)], dtype=tf.int64))
    
    load_status = True
    error_msg = ""
    try:
        model.load_weights(weights_path)
    except Exception as e:
        load_status = False
        error_msg = str(e)
    
    ratings_df = pd.read_csv(os.path.join(data_path, 'ml-1m', 'ratings_prepro.csv'))
    movies_df = pd.read_csv(os.path.join(data_path, 'ml-1m', 'movies_prepro.csv'))
    user_df = pd.read_csv(os.path.join(data_path, 'ml-1m', 'users_prepro.csv'))
    label_encoders = joblib.load(encoder_path)
    
    return user_df, movies_df, ratings_df, model, label_encoders, load_status, error_msg

# ---------------------------
# 데이터 처리 함수들
# ---------------------------
def get_user_seen_movies(ratings_df):
    return ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()

def get_user_non_seed_dict(movies_df, user_df, user_seen_movies):
    unique_movies = movies_df['movie_id'].unique()
    unique_users = user_df['user_id'].unique()
    user_non_seen_dict = {}
    for user in unique_users:
        seen_list = user_seen_movies[user_seen_movies['user_id'] == user]['movie_id'].values
        user_seen_movie_list = seen_list[0] if len(seen_list) > 0 else []
        user_non_seen_dict[user] = list(set(unique_movies) - set(user_seen_movie_list))
    return user_non_seen_dict

def get_recom(user, user_non_seen_dict, user_df, movies_df, r_year, r_month, model, label_encoders):
    # 1. 안 본 영화 목록 가져오기
    user_non_seen_movie_ids = user_non_seen_dict.get(user, [])
    if not user_non_seen_movie_ids:
        return pd.DataFrame()

    # 2. 추천 시점 정보 설정
    r_decade = str(r_year - (r_year % 10)) + 's'
    user_info = user_df[user_df['user_id'] == user].iloc[0]
    
    # 3. 예측용 데이터프레임 생성 (학습 코드와 동일한 구조)
    # movies_df에 movie_id가 문자열인지 숫자열인지 맞춰주는 것이 중요합니다.
    merge_data = movies_df[movies_df['movie_id'].isin(user_non_seen_movie_ids)].copy()
    
    # 사용자 및 시간 피처 주입
    merge_data['user_id'] = user
    merge_data['rating_year'] = r_year
    merge_data['rating_month'] = r_month
    merge_data['rating_decade'] = r_decade
    
    for col in ['gender', 'age', 'occupation', 'zip']:
        merge_data[col] = user_info[col]

    # 4. [매우 중요] 학습 시 사용했던 14개 컬럼 순서 및 이름 일치
    input_cols = [
        'user_id', 'movie_id', 'movie_decade', 'movie_year', 
        'rating_year', 'rating_month', 'rating_decade', 
        'genre1', 'genre2', 'genre3', 'gender', 'age', 'occupation', 'zip'
    ]
    
    # 부족한 컬럼이 있다면 'no'로 채워주기 (학습 데이터의 빈값 처리 방식)
    for col in input_cols:
        if col not in merge_data.columns:
            merge_data[col] = 'no'

    # 순서 재배치
    merge_data = merge_data[input_cols]
    
    # 5. 인코딩 처리 (학습 코드의 LabelEncoder 활용)
    # 모든 데이터를 문자열로 변환한 뒤 인코딩 (학습 때 str로 불렀기 때문)
    for col in input_cols:
        if col in label_encoders:
            le = label_encoders[col]
            merge_data[col] = merge_data[col].astype(str)
            # 인코더에 없는 값 처리 (첫 번째 클래스로 대체)
            known_classes = set(le.classes_)
            merge_data[col] = merge_data[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            merge_data[col] = le.transform(merge_data[col])
    
    # 6. 예측 및 결과 도출
    try:
        # 모델 입력 시 정수형 텐서로 변환
        preds = model.predict(merge_data.values.astype(np.int64), verbose=0)
        
        # 예측값과 영화 ID 결합 후 상위 10개 추출
        merge_data['pred_prob'] = preds
        top_10 = merge_data.sort_values(by='pred_prob', ascending=False).head(10)
        
        # 인코딩된 movie_id를 다시 원래 ID로 복원 (inverse_transform)
        # 이미 merge_data에 원래 정보를 가지고 있으므로 index 기반으로 찾거나 원본 join
        recom_movie_indices = top_10['movie_id'].values
        # 만약 movie_id가 이미 인코딩된 상태라면 아래 줄 사용
        origin_m_ids = label_encoders['movie_id'].inverse_transform(recom_movie_indices)
        
        return movies_df[movies_df['movie_id'].astype(str).isin(origin_m_ids.astype(str))]
    except Exception as e:
        st.error(f"⚠️ 예측 도중 오류 발생: {e}")
        return pd.DataFrame()
    
# ---------------------------
# 메인 UI 실행
# ---------------------------
users_df, movies_df, ratings_df, model, label_encoders, load_ok, err_text = load_data()
user_seen_movies = get_user_seen_movies(ratings_df)
user_non_seen_dict = get_user_non_seed_dict(movies_df, users_df, user_seen_movies)

# --- 상단 레이아웃 ---
st.markdown('<div class="marquee">✨ WELCOME TO THE BEST CINEMA ✨</div>', unsafe_allow_html=True)

col_t1, col_t2 = st.columns([1, 4])
with col_t1:
    st.image("https://cdn-icons-png.flaticon.com/512/3163/3163478.png", width=100)
with col_t2:
    st.title("오늘은 어떤 영화를 볼까요?")
    st.write("당신의 취향을 탕탕! 저격할 영화를 찾아드려요 🔫🍿")

st.divider()

# --- 입력 구역 ---
st.subheader("📝 티켓 정보를 입력해주세요")
c1, c2, c3 = st.columns(3)
with c1:
    user_id = st.number_input("👤 사용자 ID", min_value=int(users_df['user_id'].min()), max_value=int(users_df['user_id'].max()))
with c2:
    r_year = st.number_input("📅 추천 연도", min_value=2000, max_value=2025, value=2000)
with c3:
    r_month = st.number_input("🗓 추천 월", min_value=1, max_value=12, value=1)

st.write("")

# --- 추천 실행 ---
if st.button("📽️ 영화을 추천해 드릴게요!"):
    st.balloons()
    
    tab1, tab2 = st.tabs(["👤 사용자 프로필", "🎯 추천 결과"])
    
    with tab1:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("##### **내 기본 정보**")
            st.dataframe(users_df[users_df['user_id'] == user_id], use_container_width=True)
        with col_right:
            st.markdown("##### **내가 좋아한 영화**")
            past_m = ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)].merge(movies_df, on='movie_id')
            d_cols = ['title']
            if 'genre1' in past_m.columns: d_cols.append('genre1')
            st.dataframe(past_m[d_cols].head(5), use_container_width=True)

    with tab2:
        st.markdown("##### **🎬 당신을 위한 오늘의 추천 TOP 10**")
        with st.spinner('영사기를 돌리는 중... 🎞️'):
            recommendations = get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders)
            if not recommendations.empty:
                r_cols = ['title']
                if 'genre1' in recommendations.columns: r_cols.append('genre1')
                st.table(recommendations[r_cols].reset_index(drop=True))
                st.success("맛있게 관람하세요! 🍿🥤")
            else:
                st.warning("추천 목록을 불러오지 못했습니다 😢")

# --- 하단 정보 (성공 메시지 이동) ---
st.write("")
st.write("")
st.divider()
if load_ok:
    st.caption("✅ 시스템 상태: 모델 가중치 및 데이터 로드 성공")
else:
    st.error(f"❌ 시스템 상태: 가중치 로드 실패 ({err_text})")
st.caption("✨ Movie Recommendation System | AutoInt + MLP Architecture ✨")