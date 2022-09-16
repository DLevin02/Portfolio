## Written by: Drew Levin


import pandas as pd
import numpy as np
import ast

# modify the following parameters if yours are different
num_of_parameters = 5
target_num_clusters = 8
target_states = ['Wisconsin', 'Alabama']

df = pd.read_csv('./time_series_covid19_deaths_US.csv')
all_states = list(set(df.Province_State))
# sort in alphabet order
all_states.sort()
to_remove_states = ['Grand Princess',
                    'Diamond Princess',
                    'Guam',
                    'American Samoa',
                    'Virgin Islands',
                    'Northern Mariana Islands',
                    'Puerto Rico',
                    'District of Columbia',
                    ]
all_states = [x for x in all_states if x not in to_remove_states]
num_of_states = len(all_states)


def get_cumulative_timeseries(df, target_states):
    '''This function returns the cumulative timeseries death data for target states or all states
    Input:
        df: df
        target_states: e.g., ['Wisconsin', 'Alabama'], or all states
    '''
    cumulative_timeseries_data_list = []
    first_date_col = df.columns.get_loc("1/22/20")
    for state in target_states:
        state_df = df[df.Province_State == state]
        # death in a million
        state_population = state_df['Population'].sum() / 10 ** 6
        if state_population == 0:
            state_population = 1
        state_timeseries = state_df.iloc[:, first_date_col:]
        if target_states == all_states:
            state_cumulative_timeseries = (state_timeseries.sum(axis=0) / state_population).tolist()
        else:
            state_cumulative_timeseries = (state_timeseries.sum(axis=0)).tolist()
        cumulative_timeseries_data_list.append(state_cumulative_timeseries)
    return cumulative_timeseries_data_list


def get_time_diff(cumulative_timeseries_data_list):
    '''This function returns the timeseries differnece data
    Input:
        a list of cumulative timeseries data
    Return:
        a list of numpy arrays
    '''
    time_diff_list = []
    for state_cum_ts in cumulative_timeseries_data_list:
        state_time_diff = []
        for i in range(len(state_cum_ts) - 1):
            state_time_diff.append(state_cum_ts[i + 1] - state_cum_ts[i])
        time_diff_list.append(np.array(state_time_diff))
    return time_diff_list


cumulative_timeseries_data_list = get_cumulative_timeseries(df, target_states)
# wisconsin
wi_cum_ts = cumulative_timeseries_data_list[0]
print(wi_cum_ts)
# alabama
al_cum_ts = cumulative_timeseries_data_list[1]
print(al_cum_ts)
time_diff_list = get_time_diff(cumulative_timeseries_data_list)
# wisconsin
wi_time_diff = time_diff_list[0]
print(wi_time_diff)
# alabama
al_time_diff = time_diff_list[1]
print(al_time_diff)
def get_beta(state_time_diff):
    above_sum = 0
    below_sum = 0
    for t in range(1, len(state_time_diff) + 1):
        above_sum += (state_time_diff[t - 1] - mean) * (t - (len(state_time_diff) + 1) / 2)
        below_sum += np.square(t - (len(state_time_diff) + 1) / 2)
    beta = above_sum / below_sum
    return beta


def get_pho(state_time_diff):
    above_sum = 0
    below_sum = 0
    for t in range(2, len(state_time_diff) + 1):
        above_sum += (state_time_diff[t - 1] - mean) * (state_time_diff[t - 2] - mean)
    for t in range(1, len(state_time_diff) + 1):
        below_sum += np.square(state_time_diff[t - 1] - mean)
    if below_sum != 0:
        pho = above_sum / below_sum
    else:
        pho = 1
    return pho


all_cum_ts = get_cumulative_timeseries(df, all_states)
all_time_diff = get_time_diff(all_cum_ts)

means = np.zeros(num_of_states)
stds = np.zeros(num_of_states)
medians = np.zeros(num_of_states)
betas = np.zeros(num_of_states)
phos = np.zeros(num_of_states)

for idx, state_time_diff in enumerate(all_time_diff):
    mean = np.mean(state_time_diff)
    std = np.std(state_time_diff)
    median = np.median(state_time_diff)
    beta = get_beta(state_time_diff)
    pho = get_pho(state_time_diff)
    means[idx] = mean
    stds[idx] = std
    medians[idx] = median
    betas[idx] = beta
    phos[idx] = pho


# https://www.stackvidhya.com/how-to-normalize-data-between-0-and-1-range/
def rescale(array):
    diff = array - np.min(array)
    max_diff = np.max(array) - np.min(array)
    new_array = diff / max_diff
    return new_array


means = rescale(means)
stds = rescale(stds)
medians = rescale(medians)
betas = rescale(betas)
phos = rescale(phos)

params = [means, stds, medians, betas, phos]
param_matrix = np.stack(params, axis=1)

#### HIERARCHICAL CLUSTERING
M = param_matrix
print(M)

def eu_distance(x, y):
    p = np.sum((x - y) ** 2)
    d = np.sqrt(p)
    return d


dist_matrix = np.zeros((num_of_states, num_of_states))
for i in range(num_of_states):
    for j in range(num_of_states):
        if i >= j:
            dist_matrix[i, j] = 10 ** 10
        else:
            dist_matrix[i, j] = eu_distance(M[i], M[j])


def single_linkage_dist(cluster1, cluster2, dist_matrix):
    dist_list = []
    for i in cluster1:
        for j in cluster2:
            if i < j:
                dist_list.append(dist_matrix[i, j])
            else:
                dist_list.append(dist_matrix[j, i])
    return min(dist_list)


def complete_linkage_dist(cluster1, cluster2, dist_matrix):
    dist_list = []
    for i in cluster1:
        for j in cluster2:
            if i < j:
                dist_list.append(dist_matrix[i, j])
            else:
                dist_list.append(dist_matrix[j, i])
    return max(dist_list)


def cluster_hierarchy(parameter_matrix, target_num_clusters, dist_matrix, method="single"):
    '''method should be either "single" or "complete"
    '''
    clusters = [[i] for i in range(len(parameter_matrix))]
    while len(clusters) > target_num_clusters:
        dmax = np.max(dist_matrix) + 1
        dmin = dmax
        dist_dic = {}
        # clusters with minimal distances
        min_cluster1 = None
        min_cluster2 = None
        for cluster1 in clusters:
            for cluster2 in clusters:
                if cluster1 != cluster2:
                    if method == "single":
                        dist = single_linkage_dist(cluster1, cluster2, dist_matrix)
                    elif method == "complete":
                        dist = complete_linkage_dist(cluster1, cluster2, dist_matrix)
                    else:
                        print("ERROR! METHOD should be either single or complete")
                    dist_dic[f'[{cluster1},{cluster2}]'] = dist
                    if dist < dmin:
                        dmin = dist
                        min_cluster1 = cluster1
                        min_cluster2 = cluster2
        distances = np.array(list(dist_dic.values()))
        dmin_idxs = np.where(distances == dmin)[0]
        cluster_pairs = list(dist_dic.keys())
        if len(dmin_idxs) > 1:
            # cluster pairs with the same dmin
            cluster_pairs_with_dmin = [ast.literal_eval(cluster_pairs[i]) for i in dmin_idxs]
            flat_list = [item for sublist in cluster_pairs_with_dmin for item in sublist]
            min_idx = min(flat_list)
            for cluster_pair in cluster_pairs_with_dmin:
                if min_idx in cluster_pair:
                    cluster_pair_with_min_idx = cluster_pair
            min_cluster1 = cluster_pair_with_min_idx[0]
            min_cluster2 = cluster_pair_with_min_idx[1]
        clusters.remove(min_cluster1)
        clusters.remove(min_cluster2)
        clusters.append(min_cluster1 + min_cluster2)

    clustering_result = []
    for i in range(len(parameter_matrix)):
        for c in clusters:
            c_index = clusters.index(c)
            if i in c:
                clustering_result.append(c_index)

    return clustering_result


#### SINGLE LINKAGE
single_linkage_clustering = cluster_hierarchy(
    M, target_num_clusters, dist_matrix, "single")
print(single_linkage_clustering)

### COMPLETE LINKAGE
complete_linkage_clustering = cluster_hierarchy(
    M, target_num_clusters, dist_matrix, "complete")
print(complete_linkage_clustering)

### K-MEANS CLUSTERING

k = target_num_clusters
n, m = M.shape
np.random.seed(2022)
a = np.arange(n)
np.random.shuffle(a)
centers = M[a[:k]]


def d_centers2nodes(M, centers):
    n, m = M.shape
    c, w = centers.shape
    d = M.reshape([n, 1, m]) - centers.reshape([1, c, w])
    d = d ** 2
    d = np.sum(d, axis=2)
    return d


for i in range(100):
    d = d_centers2nodes(M, centers)
    index = np.argmin(d, axis=1)
    for j in range(k):
        centers[j] = np.mean(M[index == j].reshape([-1, m]), axis=0)
    # print(index)

print(index)

centers = centers.round(decimals=4)
print(centers)
d = d_centers2nodes(M, centers)
index = np.argmin(d, axis=1)
distortion = 0
for j in range(k):
    distortion += np.sum(d[index == j, j])

print(distortion)