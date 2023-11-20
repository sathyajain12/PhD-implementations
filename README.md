# K-Means Clustering algorithm
## Overview
This Python repository implements the K-means clustering algorithm, a powerful unsupervised machine learning technique for grouping data points into distinct clusters. The code includes functions for centroid initialization, iterative assignment of data points to clusters, and centroid updates until convergence. Visualization tools are provided to facilitate understanding and interpretation of the clustering results.

##Algorithm Description


K-means(D, k, ε)
    
    1. t = 0
    
    2. Randomly initialize k centroids: μ₁ᵗ, μ₂ᵗ, ..., μₖᵗ ∈ ℝᵈ
    3. repeat
    4.     t ← t + 1
    5.     Cj ← ∅ for all j = 1, ..., k  // Cluster Assignment Step
    6.     foreach xj ∈ D do
    7.         i∗ ← argminₙₖ ||xj - μᵗᵢ||₂  // Assign xj to closest centroid
    8.         Ci∗ ← Ci∗ ∪ {xj}
    9.     foreach i = 1 to k do  // Centroid Update Step
    10.        μᵗᵢ ← (1/|Ci|) ∑ₓⱼ∈Ci xj
    11. until ∑ₖ ||μᵗᵢ - μᵗ₋₁ᵢ||₂² ≤ ε





## Usage
The primary functionality lies in the kmeans function:

def kmeans(dset, k=2, tol=1e-4):
    
    #K-means implementationd for a 
    #`dset`:  DataFrame with observations
    #`k`: number of clusters, default k=2
    #`tol`: tolerance=1E-4
    
    # Let us work in a copy, so we don't mess the original
    working_dset = dset.copy()
    # We define some variables to hold the error, the 
    # stopping signal and a counter for the iterations
    err = []
    goahead = True
    j = 0
    
    # Step 2: Initiate clusters by defining centroids 
    centroids = initiate_centroids(k, dset)

    while(goahead):
        # Step 3 and 4 - Assign centroids and calculate error
        working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids) 
        err.append(sum(j_err))
        
        # Step 5 - Update centroid position
        centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)

        # Step 6 - Restart the iteration
        if j>0:
            # Is the error less than a tolerance (1E-4)
            if err[j-1]-err[j]<=tol:
                goahead = False
        j+=1

    working_dset['centroid'], j_err = centroid_assignation(working_dset, centroids)
    centroids = working_dset.groupby('centroid').agg('mean').reset_index(drop = True)
    return working_dset['centroid'], j_err, centroids

