"""
SuperDuper Broker Pre-Move Analysis
===================================
A comprehensive analysis of daily broker trading activity before known price moves.
Includes EDA, correlation, ML classification/regression, and clustering of broker behavior.
All visualizations are saved as PNG files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ==================================================
# 1. LOAD AND PREPARE DATA
# ==================================================
file_path = "/home/shanvi/nepse-engine/stat_method/output/broker_daily_pre_move_2026-05-30.csv"
df = pd.read_csv(file_path)

# Convert date columns to datetime
df['date'] = pd.to_datetime(df['date'])
df['move_start'] = pd.to_datetime(df['move_start'])
df['move_end'] = pd.to_datetime(df['move_end'])

# Sort and create a unique move identifier
df.sort_values(['symbol', 'move_start', 'date'], inplace=True)
df['move_id'] = df['symbol'] + '_' + df['window_days'].astype(str) + '_' + df['move_start'].dt.strftime('%Y%m%d')

print("Data shape:", df.shape)
print(df.head())

# ==================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==================================================
print("\n=== EDA ===")
print(df[['gain_pct', 'close', 'buy_units', 'sell_units', 'net_units', 'cumulative_net']].describe())

# Distribution of gain percentages
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
sns.histplot(df['gain_pct'], bins=30, kde=True)
plt.title('Distribution of Future Gain %')
plt.subplot(1,2,2)
sns.boxplot(y=df['gain_pct'])
plt.title('Gain % Boxplot')
plt.tight_layout()
plt.savefig('eda_gain_dist.png')
plt.close()

# Most active brokers
active_counts = df[df['active'] == True].groupby('broker_name').size().sort_values(ascending=False)
print("\nMost active brokers (number of trading days):")
print(active_counts.head(10))

# Average net units per broker (only active days)
avg_net = df[df['active']].groupby('broker_name')['net_units'].mean().sort_values(ascending=False)
print("\nBrokers with highest average net buy (units per active day):")
print(avg_net.head(10))

# ==================================================
# 3. CORRELATION: Broker cumulative net vs future gain
# ==================================================
print("\n=== Correlation Analysis ===")
corr_results = []
for days in [1, 5, 10, 20, 30]:
    subset = df[df['days_to_move'] == days].copy()
    if len(subset) > 0:
        corr = subset['cumulative_net'].corr(subset['gain_pct'])
        corr_results.append({'days_before': days, 'correlation': corr})
corr_df = pd.DataFrame(corr_results)
print("Correlation between cumulative_net and gain_pct at fixed days before move:")
print(corr_df)

plt.figure(figsize=(8,4))
plt.plot(corr_df['days_before'], corr_df['correlation'], marker='o')
plt.xlabel('Days before move start')
plt.ylabel('Pearson correlation')
plt.title('Predictive power of cumulative net by days before move')
plt.gca().invert_xaxis()
plt.grid(True)
plt.savefig('correlation_vs_days.png')
plt.close()

# ==================================================
# 4. FEATURE ENGINEERING FOR ML MODELS
# ==================================================
print("\n=== Feature Engineering ===")
move_features = []
for move_id, group in df.groupby('move_id'):
    move_info = group.iloc[0][['symbol', 'window_days', 'move_start', 'move_end', 'gain_pct']]
    agg = {
        'move_id': move_id,
        'gain_pct': move_info['gain_pct'],
        'symbol': move_info['symbol'],
        'window_days': move_info['window_days'],
         'move_start': move_info['move_start'],
        'total_days': len(group),
        'avg_close': group['close'].mean(),
        'close_volatility': group['close'].std(),
        'total_buy_units': group['buy_units'].sum(),
        'total_sell_units': group['sell_units'].sum(),
        'total_net_units': group['net_units'].sum(),
        'final_cumulative_net': group['cumulative_net'].iloc[-1] if len(group) else 0,
        'num_active_days': group['active'].sum(),
        'num_brokers': group['broker_id'].nunique(),
        'max_cumulative_net': group['cumulative_net'].max(),
        'min_cumulative_net': group['cumulative_net'].min(),
        'trend_cumulative': group['cumulative_net'].iloc[-1] - group['cumulative_net'].iloc[0] if len(group) > 1 else 0,
    }
    # Top 5 brokers by final cumulative net
    broker_last = group.groupby('broker_id').last().reset_index()
    broker_last_sorted = broker_last.sort_values('cumulative_net', ascending=False).head(5)
    for i, row in broker_last_sorted.iterrows():
        agg[f'broker_{row["broker_id"]}_final_cum'] = row['cumulative_net']
    move_features.append(agg)

move_df = pd.DataFrame(move_features)
print(f"Created {len(move_df)} move samples with features.")
print(move_df.head())

# ==================================================
# 5. MACHINE LEARNING: CLASSIFICATION (direction)
# ==================================================
print("\n=== Classification: Up vs Down (binary) ===")
move_df['direction'] = (move_df['gain_pct'] > 0).astype(int)

feature_cols = [c for c in move_df.columns if c not in ['move_id', 'symbol', 'move_start', 'move_end', 'gain_pct', 'direction', 'window_days']]
X = move_df[feature_cols].fillna(0)
y = move_df['direction']

# Time series split (respect chronological order)
move_df = move_df.sort_values('move_start')
X = X.loc[move_df.index]
y = y.loc[move_df.index]

tscv = TimeSeriesSplit(n_splits=5)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(rf_clf, X, y, cv=tscv, scoring='accuracy')
print(f"Cross-validated accuracy (time series): {scores.mean():.3f} +/- {scores.std():.3f}")

# Train on all data for feature importance
rf_clf.fit(X, y)
importances = pd.Series(rf_clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 important features for direction prediction:")
print(importances.head(10))

# ==================================================
# 6. MACHINE LEARNING: REGRESSION (gain percentage)
# ==================================================
print("\n=== Regression: Predict gain_pct ===")
y_reg = move_df['gain_pct']
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
scores_reg = cross_val_score(rf_reg, X, y_reg, cv=tscv, scoring='r2')
mae_scores = -cross_val_score(rf_reg, X, y_reg, cv=tscv, scoring='neg_mean_absolute_error')
print(f"Cross-validated R2: {scores_reg.mean():.3f} +/- {scores_reg.std():.3f}")
print(f"Mean Absolute Error: {mae_scores.mean():.2f}%")

rf_reg.fit(X, y_reg)
imp_reg = pd.Series(rf_reg.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 important features for gain regression:")
print(imp_reg.head(10))

# ==================================================
# 7. TIME-SERIES CLUSTERING OF BROKER BEHAVIOR
# ==================================================
print("\n=== Clustering Broker Cumulative Net Trajectories ===")
broker_trajectories = []
for (symbol, broker_id), group in df.groupby(['symbol', 'broker_id']):
    for move_id, move_group in group.groupby('move_id'):
        move_group = move_group.sort_values('days_to_move', ascending=False)
        cum_net = move_group['cumulative_net'].values
        broker_trajectories.append({
            'symbol': symbol,
            'broker_id': broker_id,
            'move_id': move_id,
            'gain_pct': move_group['gain_pct'].iloc[0],
            'cum_net': cum_net.tolist()
        })

# Pad to 30 days (or maximum available)
max_days = 30
X_cluster = []
for traj in broker_trajectories:
    cum = traj['cum_net']
    if len(cum) >= max_days:
        vec = cum[:max_days]
    else:
        vec = cum + [cum[-1]] * (max_days - len(cum))
    X_cluster.append(vec)
X_cluster = np.array(X_cluster)

# Normalize each trajectory
X_cluster_norm = (X_cluster - X_cluster.mean(axis=1, keepdims=True)) / (X_cluster.std(axis=1, keepdims=True) + 1e-8)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster_norm)
broker_trajectories_df = pd.DataFrame(broker_trajectories)
broker_trajectories_df['cluster'] = clusters

# Plot centroids
centroids = kmeans.cluster_centers_
plt.figure(figsize=(10,6))
for i, cent in enumerate(centroids):
    plt.plot(range(max_days), cent, label=f'Cluster {i}')
plt.xlabel('Days before move (0 = move start)')
plt.ylabel('Normalized cumulative net')
plt.title('Broker cumulative net trajectory clusters')
plt.legend()
plt.savefig('broker_clusters.png')
plt.close()

print("Cluster sizes:")
print(broker_trajectories_df['cluster'].value_counts())
cluster_gain = broker_trajectories_df.groupby('cluster')['gain_pct'].mean().sort_values()
print("\nAverage gain_pct per cluster:")
print(cluster_gain)

# ==================================================
# 8. ADDITIONAL VISUALIZATIONS
# ==================================================
print("\n=== Aggregate Broker Cumulative Net Over Time ===")
avg_cum_by_day = df.groupby('days_to_move')['cumulative_net'].mean().reset_index()
plt.figure(figsize=(10,5))
plt.plot(avg_cum_by_day['days_to_move'], avg_cum_by_day['cumulative_net'], marker='o')
plt.xlabel('Days before move start')
plt.ylabel('Average cumulative net (all brokers)')
plt.title('Average Broker Position Before Price Moves')
plt.gca().invert_xaxis()
plt.grid(True)
plt.savefig('avg_cumulative_net.png')
plt.close()

# Heatmap of net_units by broker and day before move (top brokers)
top_brokers = df.groupby('broker_name')['net_units'].sum().abs().sort_values(ascending=False).head(10).index
pivot = df[df['broker_name'].isin(top_brokers)].pivot_table(index='days_to_move', columns='broker_name', values='net_units', aggfunc='mean')
plt.figure(figsize=(14,8))
sns.heatmap(pivot, cmap='RdBu', center=0)
plt.title('Average net_units by days before move (top 10 brokers by absolute activity)')
plt.xlabel('Broker')
plt.ylabel('Days before move')
plt.savefig('broker_heatmap.png')
plt.close()

# ==================================================
# 9. FINAL SUMMARY
# ==================================================
print("\n=== ANALYSIS COMPLETE ===")
print("Generated PNG files:")
print(" - eda_gain_dist.png : distribution and boxplot of future gains")
print(" - correlation_vs_days.png : correlation between cumulative net and future gain over time")
print(" - broker_clusters.png : KMeans clusters of broker cumulative net trajectories (30 days)")
print(" - avg_cumulative_net.png : average cumulative net across all brokers vs days before move")
print(" - broker_heatmap.png : net_units heatmap for top 10 most active brokers")
print("\nMachine Learning Summary:")
print(f" - Direction classification accuracy (time series CV): {scores.mean():.3f}")
print(f" - Gain regression R2: {scores_reg.mean():.3f}, MAE: {mae_scores.mean():.2f}%")
print("\nKey predictive features (direction):")
print(importances.head(5).to_string())