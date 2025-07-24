import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('trades.csv')
df['label'] = (df['pnl'] > 0).astype(int)
features = ['score', 'atr', 'iv', 'delta_oi', 'pcr']
X = df[features]; y = df['label']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

with open('model.pkl','wb') as f:
    pickle.dump(model, f)
print("✅ model.pkl created!")
