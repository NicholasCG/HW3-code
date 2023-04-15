 # import probablistic matrix factorization, user-based collaborative filtering, and item-based collaborative filtering
from surprise import SVD, Dataset, Reader
from surprise.prediction_algorithms import KNNBasic
from surprise import accuracy
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

# clear the old images 
for f in glob('recsys_*.png'):
    os.remove(f)

# set the seed for reproducibility
np.random.seed(42)

# load ratings_small.csv
ratings = pd.read_csv('ratings_small.csv')

# get range of ratings
min_rating = ratings['rating'].min()
max_rating = ratings['rating'].max()

# print('min rating: ', min_rating)
# print('max rating: ', max_rating)
reader = Reader(rating_scale=(min_rating, max_rating))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# PMF tuned parameters
pmf_params = {'n_factors': 20, 'n_epochs': 50, 'lr_all': 0.01, 'reg_all': 0.1, 'biased': False}

print('PMF')
pmf = SVD(biased=False)
cross_validate(pmf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

print('\nPMF tuned')
pmf_tuned = SVD(n_factors=20, n_epochs=50, lr_all=0.01, reg_all=0.1, biased=False)
cross_validate(pmf_tuned, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

print('\nuser-user')
user_user = KNNBasic(user_based=True, name = 'msd', verbose=False)
cross_validate(user_user, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

print('\nitem-item')
item_item = KNNBasic( user_based=False, name = 'msd', verbose=False)
cross_validate(item_item, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

# Test the different similarity measures
user_user_cosine = KNNBasic(user_based=True, name='cosine', verbose=False)
user_user_msd = KNNBasic(user_based=True, name = 'msd', verbose=False)
user_user_pearson = KNNBasic(user_based=True, name = 'pearson', verbose=False)

item_item_cosine = KNNBasic(user_based=False, name = 'cosine', verbose=False)
item_item_msd = KNNBasic(user_based=False, name = 'msd', verbose=False)
item_item_pearson = KNNBasic(user_based=False, name = 'pearson', verbose=False)

print('\nuser-user cosine')
uu_cosine = cross_validate(user_user_cosine, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

print('\nuser-user msd')
uu_msd = cross_validate(user_user_msd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

print('\nuser-user pearson')
uu_pear = cross_validate(user_user_pearson, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

print('\nitem-item cosine')
ii_cosine = cross_validate(item_item_cosine, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

print('\nitem-item msd')
ii_msd = cross_validate(item_item_msd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

print('\nitem-item pearson')
ii_pear = cross_validate(item_item_pearson, data, measures=['RMSE', 'MAE'], cv=5, verbose=True, n_jobs=-1)

uu_scores = [uu_cosine, uu_msd, uu_pear]
ii_scores = [ii_cosine, ii_msd, ii_pear]

labels = ['cosine', 'msd', 'pearson']

# Plot the RMSE and MAE of the different similarity measures using boxplots. Have it be a 2x2 grid, where each row is a different scoring metric and each column is either user-user or item-item.
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].boxplot([uu_scores[0]['test_rmse'], uu_scores[1]['test_rmse'], uu_scores[2]['test_rmse']], showmeans=True)
axes[0, 0].set_title('User-User RMSE')
axes[0, 0].set_xticklabels(labels)
axes[0, 1].boxplot([ii_scores[0]['test_rmse'], ii_scores[1]['test_rmse'], ii_scores[2]['test_rmse']], showmeans=True)
axes[0, 1].set_title('Item-Item RMSE')
axes[0, 1].set_xticklabels(labels)
axes[1, 0].boxplot([uu_scores[0]['test_mae'], uu_scores[1]['test_mae'], uu_scores[2]['test_mae']], showmeans=True)
axes[1, 0].set_title('User-User MAE')
axes[1, 0].set_xticklabels(labels)
axes[1, 1].boxplot([ii_scores[0]['test_mae'], ii_scores[1]['test_mae'], ii_scores[2]['test_mae']], showmeans=True)
axes[1, 1].set_title('Item-Item MAE')
axes[1, 1].set_xticklabels(labels)
# save the plot
plt.savefig('recsys_similarity_measures.png')

# figure out which similarity measure is best
best_uu = labels[np.argmin([uu_scores[0]['test_rmse'].mean(), uu_scores[1]['test_rmse'].mean(), uu_scores[2]['test_rmse'].mean()])]
best_ii = labels[np.argmin([ii_scores[0]['test_rmse'].mean(), ii_scores[1]['test_rmse'].mean(), ii_scores[2]['test_rmse'].mean()])]

print('\nBest user-user similarity measure: {}'.format(best_uu))
print('Best item-item similarity measure: {}'.format(best_ii))

# Test the different values of k
uu_k = []
ii_k = []

for k in range(1, 50):
    user_user_k = KNNBasic(user_based=True, k=k, verbose=False, name=best_uu)
    item_item_k = KNNBasic(user_based=False, k=k, verbose=False, name=best_ii)
    
    uu_res = cross_validate(user_user_k, data, measures=['RMSE'], cv=5, n_jobs=-1)
    ii_res = cross_validate(item_item_k, data, measures=['RMSE'], cv=5, n_jobs=-1)

    uu_k.append(uu_res['test_rmse'].mean())
    ii_k.append(ii_res['test_rmse'].mean())

# clear the plot
plt.clf()

# Plot the RMSE of the different values of k
plt.plot(range(1, 50), uu_k, label='user-user')
plt.plot(range(1, 50), ii_k, label='item-item')
plt.xlabel('k')
plt.ylabel('RMSE')

# put a line at the best k
plt.axvline(x=np.argmin(uu_k) + 1, color='r', linestyle='--')
plt.axvline(x=np.argmin(ii_k) + 1, color='g', linestyle='--')

# legend for the plot and lines
plt.legend()
# plt.text(23, 1.1, 'Best k for user-user: ' + str(np.argmin(uu_k) + 1) + '\nRMSE: ' + np.min(uu_k), color='r')
plt.text(23, 1.1, f'Best k for user-user: {np.argmin(uu_k) + 1} \nRMSE: {np.min(uu_k):0.4f}', color='r')
# plt.text(6, 1.2, 'Best k for item-item: ' + str(np.argmin(ii_k) + 1) + '\nRMSE: ' + np.min(ii_k), color='g')
plt.text(6, 1.2, f'Best k for item-item: {np.argmin(ii_k) + 1} \nRMSE: {np.min(ii_k):0.4f}', color='g')

plt.savefig('recsys_k.png')

# print the best k for each and the corresponding RMSE
print('Best k for user-user: ', np.argmin(uu_k) + 1, ' with RMSE: ', min(uu_k))
print('Best k for item-item: ', np.argmin(ii_k) + 1, ' with RMSE: ', min(ii_k))