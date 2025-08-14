"""
This script warps the whole CLARE_SW logic, and generate the figures CLARE manuscript RTSS submission
(1) performance concerns:
    - due to the large amount of design points, conduct full simulations will be very time-consuming
    - e.g. to reproduce fig 11.a(w/t 16800 points in total) will take XX hours on a 64-core server
    - To conduct a small-scale reproduce of certain points, please refer to XX

(2) Figures to reproduce
    (1) Fig. 11(a)
    (2) Fig. 11(b)
    (3) Fig. 11(c)
    (4) Fig. 13

(3) Additional experiments: We plan to add one more figure in the camera-ready version, 
    comparing the simulation and schedulability analysis results,
    the code to produce this exp is also included in this file
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'CLARE_SW'))
import argparse
from datetime import datetime
import math
import pandas as pd

from CLARE_SW.parse_workload import AccConfig
from CLARE_SW.sim_util import ScheConfig
from CLARE_SW.search import Searcher
from CLARE_SW.utils import gen_bert_mi,gen_bert_t,gen_deit_t,gen_mlp_mixer,gen_pointnet



util_list_large = [0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975,1]
util_list_small = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,0.85, 0.9, 0.95,1]
num_util_large = 100
num_util_small = 40

def reproduce_fig11a(size, workspace='./temp/fig11a'):
    assert size in ['small','large']
    acc_config_path = './CLARE_SW/configs/acc_config.json'
    sche_config_path = './CLARE_SW/configs/sche_config.json'
    full_workspace=workspace+'_'+size
    if size == 'small':
        u_list = util_list_small
        num_util = num_util_small
    else:
        u_list = util_list_large
        num_util = num_util_large
    DNN = [
        [[1024,8192,1024],[1024,8192,1024]],
        [[1024,8192,1024],[1024,8192,1024]]
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = full_workspace,
                    )
    result = searcher.run()
    result_df = searcher.dump_sche_rate('sche_success')
    result_df = searcher.dump_sche_rate('ppp_success')
    result_df = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df)

def reproduce_fig11b(size, workspace='./temp/fig11b'):
    assert size in ['small','large']
    acc_config_path = './CLARE_SW/configs/acc_config.json'
    sche_config_path = './CLARE_SW/configs/sche_config.json'
    full_workspace=workspace+'_'+size
    if size == 'small':
        u_list = util_list_small
        num_util = num_util_small
    else:
        u_list = util_list_large
        num_util = num_util_large
    DNN = [
        [[2048,128,2048],[2048,128,2048]],
        [[2048,128,2048],[2048,128,2048]]
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = full_workspace,
                    )
    result = searcher.run()
    result_df = searcher.dump_sche_rate('sche_success')
    result_df = searcher.dump_sche_rate('ppp_success')
    result_df = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df)

def reproduce_fig11c(size, workspace='./temp/fig11c'):
    """add different apps sequentially"""
    assert size in ['small','large']
    acc_config_path = './CLARE_SW/configs/acc_config.json'
    sche_config_path = './CLARE_SW/configs/sche_config.json'
    if not os.path.exists(f"./temp/fig11c_{size}"):
        os.makedirs(f"./temp/fig11c_{size}")
    full_workspace=workspace+'_'+size
    if size == 'small':
        u_list = util_list_small
        num_util = math.ceil(num_util_small/5)
    else:
        u_list = util_list_large
        num_util = math.ceil(num_util_large/5)

    # app_workspace = os.path.join(full_workspace,'mlp1')
    # DNN = [
    #     [[2048,128,2048],[2048,128,2048]],
    #     [[2048,128,2048],[2048,128,2048]]
    # ]
    # searcher = Searcher(acc_config_path=acc_config_path,
    #                     sche_config_path=sche_config_path,
    #                     DNN_shapes=DNN,
    #                     utils=u_list,
    #                     num_util=num_util,
    #                     workspace = app_workspace,
    #                 )
    # result = searcher.run()
    # result_df = searcher.dump_sche_rate('sche_success')
    # result_df = searcher.dump_sche_rate('ppp_success')
    # result_df = searcher.dump_sche_rate('sim_success')
    # print(f"search finished, design point number: {num_util},simulation success num:")
    # print(result_df)

    # app_workspace = os.path.join(full_workspace,'mlp2')
    # DNN = [
    #     [[2048,128,2048],[2048,128,2048]],
    #     [[2048,128,2048],[2048,128,2048]]
    # ]
    # searcher = Searcher(acc_config_path=acc_config_path,
    #                     sche_config_path=sche_config_path,
    #                     DNN_shapes=DNN,
    #                     utils=u_list,
    #                     num_util=num_util,
    #                     workspace = app_workspace,
    #                 )
    # result = searcher.run()
    # result_df = searcher.dump_sche_rate('sche_success')
    # result_df = searcher.dump_sche_rate('ppp_success')
    # result_df = searcher.dump_sche_rate('sim_success')
    # print(f"search finished, design point number: {num_util},simulation success num:")
    # print(result_df)

    app_workspace = os.path.join(full_workspace,'deit-t')
    DNN = [
        gen_deit_t(),
        gen_deit_t(),
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = app_workspace,
                    )
    result = searcher.run()
    result_df1 = searcher.dump_sche_rate('sche_success')
    result_df1 = searcher.dump_sche_rate('ppp_success')
    result_df1 = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df1)

    app_workspace = os.path.join(full_workspace,'bert-t')
    DNN = [
        gen_bert_t(),
        gen_bert_t(),
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = app_workspace,
                    )
    result = searcher.run()
    result_df2 = searcher.dump_sche_rate('sche_success')
    result_df2 = searcher.dump_sche_rate('ppp_success')
    result_df2 = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df2)

    app_workspace = os.path.join(full_workspace,'bert-mi')
    DNN = [
        gen_mlp_mixer(),
        gen_mlp_mixer(),
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = app_workspace,
                    )
    result = searcher.run()
    result_df3 = searcher.dump_sche_rate('sche_success')
    result_df3 = searcher.dump_sche_rate('ppp_success')
    result_df3 = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df3)

    app_workspace = os.path.join(full_workspace,'mlp-mixer')
    DNN = [
        gen_bert_mi(),
        gen_bert_mi(),
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = app_workspace,
                    )
    result = searcher.run()
    result_df4 = searcher.dump_sche_rate('sche_success')
    result_df4 = searcher.dump_sche_rate('ppp_success')
    result_df4 = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df4)

    app_workspace = os.path.join(full_workspace,'pointNet')
    DNN = [
        gen_pointnet(),
        gen_pointnet(),
    ]
    searcher = Searcher(acc_config_path=acc_config_path,
                        sche_config_path=sche_config_path,
                        DNN_shapes=DNN,
                        utils=u_list,
                        num_util=num_util,
                        workspace = app_workspace,
                    )
    result = searcher.run()
    result_df5 = searcher.dump_sche_rate('sche_success')
    result_df5 = searcher.dump_sche_rate('ppp_success')
    result_df5 = searcher.dump_sche_rate('sim_success')
    print(f"search finished, design point number: {num_util},simulation success num:")
    print(result_df5)

    #rearrange the row idx
    row_order = ['np', 'lw', 'ir', 'ip', 'if', 'ir-ppp', 'ip-ppp', 'if-ppp']
    result_df1 = result_df1.reindex(row_order)
    result_df2 = result_df2.reindex(row_order)
    result_df3 = result_df3.reindex(row_order)
    result_df4 = result_df4.reindex(row_order)
    result_df5 = result_df5.reindex(row_order)
    final_df = pd.DataFrame(result_df1.values + result_df2.values + result_df3.values + result_df4.values + result_df5.values
                            , index=result_df1.index, columns=result_df2.columns)
    print(f"All search finished, total design point number: {num_util*5},simulation success num:")
    print(final_df)

parser = argparse.ArgumentParser(description="cmd tool for reproduce CLARE RTSS2025 Submission figures")
parser.add_argument(
    "--target",
    type=str,
    choices=["fig11a", "fig11b", "fig11c", "fig13", "sche_vs_sim"],
    help="The figure data to reproduce",
    required=True
)
parser.add_argument(
    "--size",
    type=str,
    choices=["small", "large"],
    help="size of the experiment",
    default="small"
)


if __name__ == '__main__':
    start = datetime.now()
    if not os.path.exists("temp"):
        os.makedirs("temp")
    args = parser.parse_args()
    target = args.target
    size = args.size
    if target == 'fig11a':
        reproduce_fig11a(size)
    if target == 'fig11b':
        reproduce_fig11b(size)
    if target == 'fig11c':
        reproduce_fig11c(size)

    end = datetime.now()
    print('start time:',start)
    print('end time: end',end)
    print('elapse:',end-start)