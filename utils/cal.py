import pandas as pd

# CSV 데이터를 pandas DataFrame으로 읽어오기
df = pd.read_csv("resultsForLOSO.csv")

# 'Dataset' 열을 제외하고 나머지 열들에 대해서만 평균 계산
numeric_cols = df.columns.difference(['Dataset', 'Architecture', 'Fold'])
#grouped_df = df.groupby('Architecture')[numeric_cols].mean().reset_index()
grouped_df = df.groupby('Architecture')[numeric_cols].std().reset_index()

# 'Majority class'와 'Random guess' 제외
filtered_df = grouped_df[~grouped_df['Architecture'].isin(['Majority class', 'Random guess'])]

# 'Accuracy', 'F1-score', 'ROC AUC' 열만 선택
selected_columns = ['Architecture', 'Accuracy', 'F1-score', 'ROC AUC']
result_df = filtered_df[selected_columns]

# 결과를 표로 출력
print(result_df.to_string(index=False))