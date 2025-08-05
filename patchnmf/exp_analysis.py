## Imports
# system
import os

# dsci
import numpy as np
import matplotlib.pyplot as plt

# custom library
from devenw.plot import plot_emb_scatter_3d, plot_emb_pcscatter, plot_acts_sorted, plot_dimred, plot_acts, plot_corr, plot_eigenspec, plot_rois_colorcoded, plot_dist_corr_expfit, plot_somasxy_grid, plot_v_correlations
from devenw.compute import load_redcell_iscell, fit_umap_3d, load_data, fit_pca, fit_tsne, fit_tsne_1d, fit_kmeans, prepare_data, fit_rmap_1d, load_ops, load_stat, load_stat_iscell
from devenw.spatial import get_soma_feats, get_dists_corrs, compute_dist_corr_expfit, get_dist_binned_corr, compute_v_corr_coeffs

def exp_analysis(args):
    # setting some magic numbers
    n_clusters = 10 # for k means clustering on cells
    n_rmap_components = 10 #
    t_select = np.arange(0,800,1)
    n_bins=10 # for constructing mesh for vertical binning (in calculating the spatial correlations vertically)
    n_pix_res = 512 # pixel resolution of square

    ## loading data and preparing saving folder
    ds = args.identifier

    data = load_data(ds, acts='spks')
    ops = load_ops(args)
    stat_iscell = load_stat_iscell(args)
    if ops['nchannels']>1:
        redcell_iscell = load_redcell_iscell(args)

    t = np.arange(0,data.shape[1])

    bin_w = n_pix_res/n_bins # bins for analysing correlations vertically
    bins = np.linspace(0,n_pix_res,n_bins+1) #bins for analysing correlations vertically

    # make directory to save figures and preprocessed data
    if not os.path.exists(f'../data/{ds}/exp_analysis/'):
        os.mkdir(f'../data/{ds}/exp_analysis/')
    if not os.path.exists(f'../data/{ds}/exp_analysis/figures'):
        os.mkdir(f'../data/{ds}/exp_analysis/figures')
    
    # initialise outputs dictionary
    out = dict()

    os.chdir(f'../data/{ds}/')
    

    ### RUNNING ANALYSIS 

    # running embeddings
    (pca_emb, var_exp) = fit_pca(data)
    (pca_emb_cell, var_exp) = fit_pca(data.T)
    tsne_emb_cell = fit_tsne_1d(data.T)
    rmap_emb_cell = fit_rmap_1d(data.T, n_X=n_rmap_components)
    umap_emb = fit_umap_3d(data[:,t_select])

    # run kmeans clusterring on cells
    labels = fit_kmeans(data, n_clusters=n_clusters)

    # run distance-correlation analysis
    somas_y, somas_x, somas_plane, areas = get_soma_feats(stat_iscell)
    unique_planes = np.unique(somas_plane)

    for i_plane in unique_planes:
        dist_corr = get_dists_corrs(data, somas_x, somas_y, areas, somas_plane, i_plane)
        dist_corr_params = compute_dist_corr_expfit(dist_corr)
        out[f'dist_corr_plane{i_plane}'] = dist_corr
        out[f'dist_corr_params_plane{i_plane}'] = dist_corr_params

    v_correlations = compute_v_corr_coeffs(data, somas_x, somas_y, n_bins, bin_w, n_pix_res, somas_plane)

    ## preparing output struct and saving to dataset directory
    out_var_names = ['data', 'pca_emb', 'var_exp', 'tsne_emb_cell', 'rmap_emb_cell'] # names of variables to import
    for var_name in out_var_names:
        out[var_name] = locals()[var_name] # saving variables under keys of the same name
    
    print(f'Saving dictionary of output with keys: {out.keys()}')
    np.save('exp_analysis/out.npy', out, allow_pickle=True)
    
    
    ### PLOTTING RESULTS

    # plot activations (by themselves and sorted by embeddings)
    plot_acts(data, title='unsorted') # unsorted
    plot_corr(data, title='acts_corr_unsorted')

    for pc_sort in range(3): # pca sorted
        plot_acts_sorted(data, pca_emb_cell, add_plot=pca_emb, component=pc_sort, title=f'pc_{pc_sort+1}')

    plot_acts_sorted(data, tsne_emb_cell, component=0, title=f'tsne')    # tsne sorted
    plot_acts_sorted(data, rmap_emb_cell, component=0, title=f'rmap')    # rmap sorted
    plot_acts_sorted(data, labels[:,np.newaxis], component=0, title=f'kmeans')    # clustering sorted

    # plotting embeddings
    plot_dimred(pca_emb, t, title='pca_time') # on time
    plot_eigenspec(var_exp, trunc_fit_pcs=(1,100)) # on time
    plot_dimred(pca_emb_cell, labels, title='pca_cell') # on cells

    plot_emb_pcscatter(pca_emb, t, t_select)    #pca scatter
    plot_emb_scatter_3d(pca_emb[t_select], title='pca_3d')    # pca 3d
    plot_emb_scatter_3d(umap_emb, title='umap_3d')    # umap 3d

    # plotting colorcoded rois
    plot_rois_colorcoded(stat_iscell, image=ops['meanImgE'], c_cells=labels, title=f'k={n_clusters}_means')
    plot_rois_colorcoded(stat_iscell, image=ops['meanImgE'], c_cells=rmap_emb_cell, title='rmap_cell')
    plot_rois_colorcoded(stat_iscell, image=ops['meanImgE'], c_cells=tsne_emb_cell, title='tsne_cell')
    if ops['nchannels']>1:
        plot_rois_colorcoded(stat_iscell, image=ops['meanImgE'], c_cells=redcell_iscell, title='redcell_iscell')
        
    # plotting distance-correlation
    for i_plane in unique_planes:
        plot_dist_corr_expfit(out[f'dist_corr_plane{i_plane}'], out[f'dist_corr_params_plane{i_plane}'], title=f'dist-corr_expfit_plane{i_plane}')

    plot_somasxy_grid(somas_plane, somas_x, somas_y, bins, n_pix_res)
    plot_v_correlations(v_correlations)

    plt.close('all') # clear memory