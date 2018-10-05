import math
import io

#파일 압축 용도
import gzip
import pickle
import zlib

# 데이터, 배열
import pandas as pd
import numpy as np

# 범주형 수치형 변환
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb


np.random.seed(2016)
transformers={}

products = (
    "ind_ahor_fin_ult1",
    "ind_aval_fin_ult1",
    "ind_cco_fin_ult1" ,
    "ind_cder_fin_ult1",
    "ind_cno_fin_ult1" ,
    "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1",
    "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1",
    "ind_deme_fin_ult1",
    "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1",
    "ind_fond_fin_ult1",
    "ind_hip_fin_ult1" ,
    "ind_plan_fin_ult1",
    "ind_pres_fin_ult1",
    "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1",
    "ind_valo_fin_ult1",
    "ind_viv_fin_ult1" ,
    "ind_nomina_ult1"  ,
    "ind_nom_pens_ult1",
    "ind_recibo_ult1"  ,
)

dtypes = {
    "fecha_dato": str,
    "ncodpers": int,
    "conyuemp": str, # Spouse index. 1 if the customer is spouse of an employee
}


# 라벨 인코더 함수
def label_encode(df, features, name):
    # 데이터 프레임 df의 변수 name값을 모두 string으로 변환
    df[name] = df[name].astype('str')

    #이미 라벨 인코더 했던 변수는 trasformer[name]에 있는 라벨 인코더를 재활용
    if name in transformers:
        df[name] = transformers[name].transform(df[name])

    #처음 나오는 변수는 transformer에 라벨인코더를 저장하고 fit_transfrom으로 라벨인코딩
    else:
        transformers[name] = LabelEncoder()
        df[name] = transformers[name].fit_transform(df[name])
        # 라벨인코딩한 변수는 features 리스트에 추가
        features.append(name)



# 자체 구현 one hot encoder
def custom_one_hot(df, features, name, names, dtype = np.int8, check = False):
    for n, val in names.items():
        # 신규 변수명을 "변수명_숫자" 지정
        new_name = "%s_%s" % (name, n)
        #기존 변수에서 해당고유값 있으면 1 그외는 0 이진변수 생성
        df[new_name] = df[name].map(lambda x : 1 if x == val else 0).astype(dtype)
        features.append(new_name)

# 라벨인코더, 원핫인코더, 유틸.py의 빈도 추출, 날짜 변환을 이용해
# 변수에 대한 전처리와 피쳐 엔지니어링 수행
def apply_transforms(train_df):

    # 학습에 사용할 변수를 저장할 features 리스트 생성
    features = []

    # 두 변수를 label_encode()
    label_encode(train_df, features, "canal_entrada")
    label_encode(train_df, features, "pais_residencia")

    # age의 결측값을 0.0으로 대체하고, 모든 값을 정수로 변환.(내 생각 - renta는 값의 범위가 크기에 log를 위해 1로 채워넣음)
    train_df["age"] = train_df["age"].fillna(0.0).astype(np.int16)
    features.append("age")

    # renta 결측값을 1.0으로 대체 하고 log를 씌워 정수 변환
    train_df["renta"].fillna(1.0, inplace=True)
    train_df["renta"] = train_df["renta"].map(math.log)
    features.append("renta")

    # 고빈도 100개의 순위 추출
    train_df["renta_top"] = encode_top(train_df["renta"])
    features.append("renta_top")

    #결측값, 음수를 0으로 대체, 나머지는 +1.0후 정수 변환
    train_df["antiguedad"] = train_df["antiguedad"].map(lambda x : 0.0 if x < 0 or math.isnan(x) else  x + 1.0).astype(np.int16)
    features.append("antiguedad")

    # 결측값을 0.0으로 대체하고, 정수로 변환
    train_df["tipodom"] = train_df["tipodom"].fillna(0.0).astype(np.int8)
    features.append("tipodom")

    train_df["cod_prov"] = train_df["cod_prov"].fillna(0.0).astype(np.int8)
    features.append("cod_prov")

    # fecha_dato에서 월/년도 추출하여 정수값으로 변환
    train_df["fecha_dato_month"] = train_df["fecha_dato"].map(lambda x : int(x.split("-")[1])).astype(np.int8)
    features.append("fecha_dato_month")
    train_df["fecha_dato_year"] = train_df["fecha_dato"].map(lambda x : int(x.split("-")[0])).astype(np.int16)
    features.append("fecha_dato_year")


    # 결측값을 0.0으로 대체하고, fecha_alta에서 월/년도를 추출하여 정수값으로 변환
    # x.__class__를 통해 결측값 탐지.  x.__class__는 결측값일 경우 float을 반환
    train_df["fecha_alta_month"] = train_df["fecha_alta"].map(lambda x : 0.0 if x.__class__ is float else float(x.split("-")[1])).astype(np.int8)
    features.append("fecha_alta_month")
    train_df["fecha_alta_year"] = train_df["fecha_alta"].map(lambda x : 0.0 if x.__class__ is float else float(x.split("-")[0])).astype(np.int8)
    features.append("fecha_alta_year")

    #날짜 데이터를 월 기준 수치형 변수로 변환
    train_df["fecha_dato_float"] = train_df["fecha_dato"].map(date_to_float)
    train_df["fecha_alta_float"] = train_df["fecha_dato"].map(date_to_float)

    # fecha_dato와 fecha_alto의 월 기준 수치형 변수의 차이값을 파생 변수로 생성
    train_df["dato_minus_alta"] = train_df["fecha_dato_float"] - train_df["fecha_alta_float"]
    features.append("dato_minus_alta")

    # 날짜 데이터를 월 기준 수치형 변수로 변환 (1 ~ 18 사이 값으로 제한)
    train_df["int_date"] = train_df["fecha_dato"].map(date_to_int).astype(np.int8)

    # 원핫 인코딩 수행
    custom_one_hot(train_df, features, "indresi", {"n" : "N"})
    custom_one_hot(train_df, features, "indext", {"s" : "S"})
    custom_one_hot(train_df, features, "conyuemp", {"n" : "N"})
    custom_one_hot(train_df, features, "sexo", {"h" : "H", "v":"V"})
    custom_one_hot(train_df, features, "ind_empleado", {"a" : "A", "b":"B", "f": "F", "n":"N"})
    custom_one_hot(train_df, features, "ind_nuevo", {"new" : "1"})
    custom_one_hot(train_df, features, "segmento", {"top" : "01 - TOP", "particulares" : "02 - PARTICULARES", "universitario" : "03 - UNIVERSITARIO"})
    custom_one_hot(train_df, features, "indfall", {"s" : "S"})
    custom_one_hot(train_df, features, "indrel", {"1" : 1, "99" : 99})
    custom_one_hot(train_df, features, "tiprel_1mes", {"a" : "A", "i":"I", "p":"P", "r":"R"})

    # 결측값을 0.0으로 대체 하고 그외는 +1.0 더하고 정수 변환
    train_df["ind_actividad_cliente"] = train_df["ind_actividad_cliente"].map(lambda x : 0.0 if math.isnan(x) else x + 1.0).astype(np.int8)
    features.append("ind_actividad_cliente")


    # 결측값을 0.0으로 대체하고, "P"를 5로 대체하고, 정수 변환
    train_df["indrel_1mes"] = train_df["indrel_1mes"].map(lambda x : 5.0 if x =="P" else x).astype(float).fillna(0.0).astype(np.int8)
    features.append("indrel_1mes")



    # 전처리, 피처 엔지니어링이 1차적으로 완료된 데이터 프레임 train_df와 학습에 필요한 변수리스트 features를 튜플 형태로 반환
    return train_df, tuple(features)



def make_prev_df(train_df, step):
    # 새로운 데이터 프레임에 ncodpers를 추가, int_date를 step만큼 이동시킨 값
    prev_df = pd.DataFrame()
    prev_df["ncodpers"] = train_df["ncodpers"]

    prev_df["int_date"] = train_df["int_date"].map(lambda x : x + step).astype(np.int8)

    # "변수명_prev1" 형태의 lag 변수를 생성
    prod_features = ["%s_prev%s" % (prod, step) for prod in products]
    for prod, prev in zip(products, prod_features):
        prev_df[prev] = train_df[prod]

    return prev_df, tuple(prod_features)



# lag 변수를 훈련 데이터에 통합
def join_with_prev(df, prev_df, how):
    # merge를 통해 join
    df = df.merge(prev_df, on=["ncodpers", "int_date"], how = how)
    # 24개 금융변수를 소수형으로 변환
    for f in set(prev_df.columns.values.tolist()) - set(["ncodpers", "int_date"]):
        df[f] = df[f].astype(np.float16)
    return df


def load_data():
    fname = "/home/jeongchanwoo/workspace/git/study/Kaggle_data/santander-product-recommendation/input/8th.clean.all.csv"
    train_df = pd.read_csv(fname, dtype=dtypes)

    for prod in products:
        train_df[prod] = train_df[prod].fillna(0.0).astype(np.int8)

    # 48개 변수 피쳐 엔지니어링
    train_df, features = apply_transforms(train_df)


    # lag_5 변수 생성
    prev_dfs = []
    prod_features = None

    user_features = frozenset([1,2])

    # 1~5까지의 step에 대해 lag-n 데이터 생성
    for step in range(1,6):
        prev1_train_df, prod1_features = make_prev_df(train_df, step)

        # 생성한 lag는 prev_dfs에 저장
        prev_dfs.append(prev1_train_df)
        # features에 lag-1,2만 추가
        if step in user_features:
            features += prod1_features

        # prod_features에 lag-1 변수명만 저장
        if step == 1:
            prod_features = prod1_features

    return train_df, prev_dfs, features, prod_features



def make_data():
    train_df, prev_dfs, features, prod_features = load_data()
    # lag-5 변수 통합
    for i, prev_df in enumerate(prev_dfs):
        how = "inner" if i==0 else "left"
        train_df = join_with_prev(train_df, prev_df, how=how)

    # 24개 금융변수에 대해 lag 별로 기초통계량을 변수화
    for prod in products:
        # [1~3], [1~5],[2~5] 3개 구간에 대해 표준편차
        for begin, end in [(1,3), (1,5), (2,5)]:
            prods = ["%s_prev%s" % (prod, i ) for i in range(begin - end + 1)]
            mp_df = train_df.as_matrix(columns = prods)
            stdf = "%s_std_%s_%s" % (prod, begin, end)

            # np.nanstd로 표준편차 구하고, features에 신규 파생변수 이름 추가
            train_df[stdf] = np.nanstd(mp_df, axis=1)
            features += (stdf,)


        # [2~3], [2~5]에 대해 최솟값 최댓값 구함
        for begin, end in [(2,3), (2,5)]:
            prods = ["%s_prev%s" % (prod, i) for i in range(begin, end+1)]
            mp_df = train_df.as_matrix(columns = prods)

            minf = "%s_min_%s_%s" % (prod, begin,end)
            train_df[minf] = np.nanmin(mp_df, axis=1).astype(np.int8)

            maxf = "%s_max_%s_%s" % (prod, begin,end)
            train_df[maxf] = np.nanmax(mp_df, axis=1).astype(np.int8)

            features += (minf, maxf,)


    # 고객 식별 번호(ncodpers), 정수 표현 날짜(inf_date), 실제 날짜(fecha_dato), 24개 금융변수(prodcuts), 전처리/피처 엔지니어링 변수(features)가 주요변수

    leave_columns = ["ncodpers", "int_date", "fecha_dato"] + list(products) + list(features)

    # 중복값 확인
    assert len(leave_columns) == len(set(leave_columns))

    # train_df에서 주요변수 추출
    train_df = train_df[leave_columns]

    return train_df, features, prod_features


## Utils

# 빈도 상위 100개 데이터 순위 변수 추출
def encode_top(s, count=100, dtype = np.int8):
    # 고유값 빈도 계산
    uniqs, freqs = np.unique(s, return_counts=True)
    # 빈도 top 100 추출
    top = sorted(zip(uniqs, freqs), key = lambda vk : vk[1], reverse= True)[:count]

    # 기존데이터 : 순위 dict 생성
    top_map = {uf[0] : l + 1 for uf, l in zip(top, range(len(top)))}

    # 고빈도 100개의 데이터는 순위로 대체, 그외는 0으로 대체
    return s.map(lambda x: top_map.get(x,0)).astype(dtype)

#날짜데이터를 월 단위 숫자로 변환 utils.py

def date_to_float(str_date):
    if str_date.__class__ is float and math.isnan(str_date) or str_date =="":
        return np.nan

    Y, M, D = [int(a) for a in str_date.strip().split("-")]
    float_date = float(Y) * 12 + float(M)
    return float_date


#날짜데이터를 월단위로 변환하여 1~18사이로 제한
def date_to_int(str_date):
    Y,M,D = [int(a) for a in str_date.strip().split("-")]
    int_date = (int(Y) - 2015) * 12 + int(M)
    assert 1 <= int_date <= 12+6
    return int_date

# metric
def apk(actual, predicted, k=10, default=1.0):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return default

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10, default=1.0):
    return np.mean([apk(a,p,k,default) for a,p in zip(actual, predicted)])


## Model
#XGBoost 모델 학습
def xgboost(XY_train, XY_validate, test_df, features, XY_all = None, restore = False):
    # 최적 파라메터
    param = {
        'objective' : 'multi:softprob',
        'eta' : 0.1,
        'min_child_weight' : 10,
        'max_depth' : 8,
        'silent' : 1,
        'eval_metric' : 'mlogloss',
        'colsample_bytree' : 0.8,
        'colsample_bylevel' : 0.9,
        'num_class' : len(products),
        'tree_method' : 'gpu_hist',
    }

    if not restore:
        # 훈련 데이터에서 X,Y, weight 매트릭스 추출
        X_train = XY_train.as_matrix(columns=features)
        Y_train = XY_train.as_matrix(columns=["y"])
        W_train = XY_train.as_matrix(columns=["weight"])

        # xgboost 데이터로 변환
        train = xgb.DMatrix(X_train, label=Y_train, feature_names=features, weight=W_train)

        # 검증 데이터에 대해서 동일한 작업진행
        X_validate = XY_validate.as_matrix(columns=features)
        Y_validate = XY_validate.as_matrix(columns=["y"])
        W_validate = XY_validate.as_matrix(columns=["weight"])

        # xgboost 데이터로 변환
        validate = xgb.DMatrix(X_validate, label=Y_validate, feature_names=features, weight=W_validate)

        # XGBoost 모델 학습 - 얼리 스탑 : 20, 트리 갯수 : 1000
        evallist = [(train,'train'), (validate,'eval')]
        model = xgb.train(param, train, 1000, evals=evallist, early_stopping_rounds=20)

        # 학습 모델 저장
        pickle.dump(model, open("next_multi.pickle", "wb"))

    else:
        # 2016-06-28 테스트 데이터를 사용할 시 사전에 학습된 모델 불러옴
        model = pickle.load(open("next_multi.pickle", "rb"))

    # 교차 검증으로 최적의 트리 갯수를 정함
    best_ntree_limit = model.best_ntree_limit

    if XY_all is not None:
        # 전체 훈련 데이터에 대해 X,Y,weight를 추출하고 XGBoost 전용 데이터로 변환
        X_all = XY_all.as_matrix(columns = features)
        Y_all = XY_all.as_matrix(columns = ["y"])
        W_all = XY_all.as_matrix(columns = ["weight"])
        all_data = xgb.DMatrix(X_all, label=Y_all, feature_names=features, weight=W_all)

        evallist = [(all_data, 'all_data')]

        # 학습할 트리 갯수를 전체 훈련 데이터가 늘어난 만큼 조정??????
        best_ntree_limit = int(best_ntree_limit * (len(XY_train) + len(XY_validate))/ len(XY_train))

        # 모델 학습!!!
        model = xgb.train(param, all_data,best_ntree_limit, evals=evallist)


    # 변수 중요도 출력 .get_fscore()통해서
    print("Feature importance : ")
    for kv in sorted([(k,v) for k,v in model.get_fscore().items()], key = lambda kv : kv[1], reverse=True) :
        print(kv)

    # 예측에 사용할 테스트 데이터를 XGBoost 데이터로 변환, 이때의 웨이트는 전부 1이기에 별도 작업 없음
    X_test = test_df.as_matrix(columns = features)
    test = xgb.DMatrix(X_test, feature_names=features)

    # 학습된 모델, best_ntree_limit 기반 예측
    return model.predict(test, ntree_limit = best_ntree_limit)


def make_submission(f, Y_test, C):
    Y_ret = []
    # 파일 첫 줄에 header
    f.write("ncodpers,added_products\n".encode('utf-8'))
    # 고객 식별번호(C), 예측결과물(Y_test) for loop

    for c, y_test in zip(C,Y_test):
        # 확률, 금융변수, 금융변수 id 튜플
        y_prods = [(y, p , ip) for y, p, ip in zip(y_test, products, range(len(products)))]

        # 확률 기준 상위 7개 결과 추출
        y_prods = sorted(y_prods, key = lambda a : a[0], reverse = True)[:7]

        # 금융 id를 Y_ret에 저장
        Y_ret.append([ip for y,p,ip in y_prods])
        y_prods = [p for y,p, ip in y_prods]

        # 고객식별, 7개 변수 파일에 기재
        f.write(("%s,%s\n" % (int(c), " ".join(y_prods))).encode('utf-8'))

    return Y_ret

def train_predict(all_df, features, prod_features, str_date, cv):

    # str_date로 예측 결과물을 산출하는 날짜 지정
    test_date = date_to_int(str_date)

    # 훈련 데이터는 test_date 이전의 모든데이터
    train_df = all_df[all_df.int_date < test_date]
    # 테스트 데이터를 분리
    test_df = pd.DataFrame(all_df[all_df.int_date == test_date])


    # 신규 구매 고객만을 훈련 데이터로 추출
    X =[]
    Y = []

    for i, prod in enumerate(products):
        prev = prod + "_prev1"
        # 신규 구매 고객을 prX에 저장
        prX = train_df[(train_df[prod] == 1) & (train_df[prev] == 0)]
        # 신규 구매에 대한 label값을 prY에 저장
        prY = np.zeros(prX.shape[0], dtype = np.int8) + i
        X.append(prX)
        Y.append(prY)


    XY = pd.concat(X)
    Y = np.hstack(Y)

    # XY에는 신규 구매 데이터만 포함함
    XY["y"] = Y

    # 메모리에서 변수 삭제
    del train_df
    del all_df

    # 데이터별 가중치 계산위한 변수 (ncodpers  + fecha_dato) 생성
    XY["ncodpers_fecha_dato"] = XY["ncodpers"].astype(str) + XY["fecha_dato"]
    uniqs, counts = np.unique(XY["ncodpers_fecha_dato"], return_counts= True)

    # 자연상수를 통해서 count가 높은 데이터에 낮은 가중치
    weights = np.exp(1/counts - 1)

    # 가중치를 XY데이터에 추가
    wdf = pd.DataFrame()
    wdf["ncodpers_fecha_dato"] = uniqs
    wdf["counts"] = counts
    wdf["weight"] = weights
    XY = XY.merge(wdf, on="ncodpers_fecha_dato")

    # 교차검증을 위해 8:2로 분리
    mask = np.random.rand(len(XY)) < 0.8
    XY_train = XY[mask]
    XY_validate = XY[~mask]


    # 테스트 데이터 가중치 전부 1
    test_df["weight"] = np.ones(len(test_df), dtype=np.int8)

    # 테스트 데이터에서 신규 구매 정답값을 추출
    test_df["y"] = test_df["ncodpers"]
    Y_prev = test_df.as_matrix(columns = prod_features)
    for prod in products:
        prev = prod + "_prev1"
        padd = prod + "_add"
        # 신규구매여부
        test_df[padd] = test_df[prod] - test_df[prev]
    test_add_mat = test_df.as_matrix(columns = [prod + "_add" for prod in products])

    C = test_df.as_matrix(columns=["ncodpers"])
    test_add_list = [list() for i in range(len(C))]
    # MAP@7 계산을 위해 고객별 신규 구매 정답값을 test_add_list에 기록
    count = 0
    for c in range(len(C)):
        for p in range(len(products)):
            if test_add_mat[c, p] >0: # 즉 신규
                test_add_list[c].append(p)
                count +=1

    # 교차 검증에서, 테스트 데이터로 분리된 데이터가 얻을 수 있는 최대 MAP@7 값을 계산한다.
    if cv:
        max_map7 = mapk(test_add_list, test_add_list, 7, 0.0)
        map7coef = float(len(test_add_list)) / float(sum([int(bool(a)) for a in test_add_list]))
        print("Max MAP@7", str_date, max_map7, max_map7*map7coef)



    #XGBoost 모델 학습 후 예측 결과물 저장
    Y_test_xgb = xgboost(XY_train, XY_validate, test_df, features, XY_all = XY, restore=(str_date == "2016-06-28"))
    test_add_list_xgboost = make_submission(io.BytesIO() if cv else gzip.open("%s.xgboost.csv.gz" % str_date, "wb"), Y_test_xgb - Y_prev, C) #Y_prev를 빼면 신규 구매가 아닌것의 확률을 확 낮춤

    # 교차 검증일 시 XGBoost 모델의 테스트 데이터 MAP@7 척도 출력

    if cv:
        map7xgboost = mapk(test_add_list, test_add_list_xgboost, 7, 0.0)
        print("XGBoost MAP@7", str_date, map7xgboost, map7xgboost* map7coef)



if __name__ == '__main__':
    all_df, features, prod_features = make_data()

    all_df.to_pickle("/home/jeongchanwoo/workspace/git/study/Kaggle_data/santander-product-recommendation/input/8th.feature_engineer.all.pkl")
    pickle.dump((features, prod_features), open("/home/jeongchanwoo/workspace/git/study/Kaggle_data/santander-product-recommendation/input/8th.feature_engineer.cv_meta.pkl", "wb"))

    train_predict(all_df, features, prod_features, "2016-05-28", cv=True)
    train_predict(all_df, features, prod_features, "2016-06-28", cv=False)
