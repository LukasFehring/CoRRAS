import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from Corras.Model import log_linear
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_table(sep="\t",filepath_or_buffer="LabelRankingData/iris_dense.txt")
df = df.iloc[1:]
feature_columns = [x for x in df.columns if x[0]=="A"]
ranking_columns = [x for x in df.columns if x[0]=="L"]
features = df[feature_columns]
rankings = df[ranking_columns]
print(features)
print(rankings)

kf = KFold(n_splits=2 , shuffle=True, random_state=5)
split = next(kf.split(df), None)

# training_portions = np.linspace(start=0, stop=1, num=5)
training_portions = [1.0]
result_data = []

# for split_num, split in enumerate(kf.split(df)):

for portion in training_portions:
        
        
        
    train_features = features.iloc[split[0]].astype('float64')
    train_rankings = rankings.iloc[split[0]].astype('int32')
    test_features = features.iloc[split[1]].astype('float64')
    test_rankings = rankings.iloc[split[1]].astype('int32')

    train_features = train_features[:int(portion*len(train_features))] 
    train_rankings = train_rankings[:int(portion*len(train_rankings))]

    train_features_np = train_features[:int(portion*len(train_features))].values
    train_rankings_np = train_rankings[:int(portion*len(train_rankings))].values

    train_features_np = train_features[:4].values
    train_rankings_np = train_rankings[:4].values

    # train_rankings_np = train_rankings_np
    # # train_features_np = train_features_np

    # if(len(train_rankings) == 0):
    #     continue

    print("len", len(train_features), len(train_rankings))

    model = log_linear.LogLinearModel()
    # model.fit(train_rankings[:6],None,train_features[:6],None,lambda_value=1,regression_loss="Squared", maxiter=100)
    # model.fit_np(train_rankings_np,None,train_features_np,None,lambda_value=1,regression_loss="Squared", maxiter=100)

    test_weights = np.random.rand(train_rankings.shape[1], train_features.shape[1]+1)
    print("test weights shape",test_weights.shape)
    nll = model.negative_log_likelihood(train_rankings[:4],train_features[:4],test_weights)
    nll = model.vectorized_nll(train_rankings_np,train_features_np,test_weights)
    # print("nll", nll)
    current_taus = []

#         for index, row in test_features.iterrows():
#             predicted_ranking = model.predict_ranking(row)
#             true_ranking = test_rankings.loc[index].values
#             tau = kendalltau(predicted_ranking,true_ranking).correlation
#             result_data.append([split_num, portion, tau]) 

# results = pd.DataFrame(data=result_data,columns=["split", "train_portion", "tau"])
# print("avg kendalls tau:", results["tau"].mean())
# # sb.lineplot(x="train_portion", y="tau", data=results)
# plt.show()

