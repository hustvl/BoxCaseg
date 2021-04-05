# demo
import numpy as np
from skimage import io
import glob
from core.DUT_eval.measures import compute_ave_MAE_of_methods


def dut_eval(gt_dir, rs_dirs):
    ## 0. =======set the data path=======
    print("------0. set the data path------")

    # # >>>>>>> Follows have to be manually configured <<<<<<< #
    data_name = 'TEST-DATA' # this will be drawn on the bottom center of the figures
    # data_dir = '../test_data/' # set the data directory,
    #                           # ground truth and results to-be-evaluated should be in this directory
    #                           # the figures of PR and F-measure curves will be saved in this directory as well
    # gt_dir = 'DUT-OMRON/pixelwiseGT-new-PNG'# 'gt' # set the ground truth folder name
    # rs_dirs = ['u2net_results']#['rs1','rs2'] # set the folder names of different methods
    #                         # 'rs1' contains the result of method1
    #                         # 'rs2' contains the result of method 2
    #                         # we suggest to name the folder as the method names because they will be shown in the figures' legend
    lineSylClr = ['r-', 'b-']  # curve style, same size with rs_dirs
    linewidth = [2, 1]  # line width, same size with rs_dirs
    # >>>>>>> Above have to be manually configured <<<<<<< #

    gt_name_list = glob.glob(gt_dir + '/' + '*.png')  # get the ground truth file name list

    ## get directory list of predicted maps
    rs_dir_lists = []
    for i in range(len(rs_dirs)):
        rs_dir_lists.append(rs_dirs[i] + '/')
    print('\n')

    ## 1. =======compute the average MAE of methods=========
    print("------1. Compute the average MAE of Methods------")
    aveMAE, gt2rs_mae = compute_ave_MAE_of_methods(gt_name_list, rs_dir_lists)
    print('\n')
    for i in range(0, len(rs_dirs)):
        print('>>%s: num_rs/num_gt-> %d/%d, aveMAE-> %.3f' % (rs_dirs[i], gt2rs_mae[i], len(gt_name_list), aveMAE[i]))

    ## 2. =======compute the Precision, Recall and F-measure of methods=========
    from core.DUT_eval.measures import compute_PRE_REC_FM_of_methods, plot_save_pr_curves, plot_save_fm_curves

    print('\n')
    print("------2. Compute the Precision, Recall and F-measure of Methods------")
    PRE, REC, FM, gt2rs_fm = compute_PRE_REC_FM_of_methods(gt_name_list, rs_dir_lists, beta=0.3)
    for i in range(0, FM.shape[0]):
        print(">>", rs_dirs[i], ":", "num_rs/num_gt-> %d/%d," % (int(gt2rs_fm[i][0]), len(gt_name_list)),
              "maxF->%.3f, " % (np.max(FM, 1)[i]), "meanF->%.3f, " % (np.mean(FM, 1)[i]))
    print('\n')
    
    ## end
    print('Done!!!')
    return aveMAE[0], np.max(FM, 1)[0]
