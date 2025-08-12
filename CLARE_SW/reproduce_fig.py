from typing import List, Type
import random #random seed are set in the Searcher class
import pandas as pd
import os
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

from utils import uunifast, lcm
from parse_workload import AccConfig,Workload
from apply_strategy import *
from schedulability_analysis import AccTaskset, schedulability_analyzer, PP_placer
from sim_util import ScheConfig, AccTasksetSim, SimManager



#####################
###random seed#######
random.seed(42)
#####################

# class TestResult:
#     """data class for handling the result"""
#     def __init__(self,sche_success:bool, PPP_success:bool, sim_success:bool):
#         self.sche_success = sche_success #pass schedulability analysis?
#         self.PPP_success = PPP_success #pass PPP?
#         self.sim_success = sim_success #pass simulation?

class TestDesignPt:
    """static class grouping series of funcs"""
    def test(DNN_shapes:List[List[List[int]]],utils:list,
                        acc_config:AccConfig,sche_config:ScheConfig,
                        strategies:List=['np','lw','ip','ir','if','ip-ppp','ir-ppp','if-ppp']):
        """test a design point, comparing different stategies
        input: 
            - DNN_shapes(list): 3-d list: task-layer-M,K,N shape of the layer
            - utilizations(list), 
            - acc_config(AccConfig), sche_config(ScheConfig)
            - strategies(list):
        Output:
            - pd.dataframe:
                - 3-rows for sche_analysis, PPP, and simulation success 
        Allowed strategies
            - np: Non-Preemptive
            - lw: LayerWise-preemptive
            - ip: Intra-layer-preemptive Persist
            - ir: Intra-layer-preemptive Recompute
            - if: Intra-layer-preemptive Flexible
            - ip-ppp,ir-ppp,if-ppp: ip,ir,if w/t PP placement optimization
        parse the shape --> workload --> strategies --> conduct sche analysis/PPP for each strategy --> simulation"""
        assert set(strategies).issubset(['np','lw','ip','ir','if','ip-ppp','ir-ppp','if-ppp']),\
            "[test_design_pt.test] invalid strategies"
        #dump shape into workload
        workloads:List[Workload] = []
        for idx, task_shape in enumerate(DNN_shapes):
            workloads.append(Workload(task_shape,acc_config,f"task{idx}"))
        #conduct test
        result_df_list = []
        for s in strategies:
            if s == 'np':
                result_df_list.append(TestDesignPt._test_sche_analysis(workloads,utils,sche_config,s,StrategyNonPreemptive))
            elif s== 'lw':
                result_df_list.append(TestDesignPt._test_sche_analysis(workloads,utils,sche_config,s,StrategyLayerwise))
            elif s== 'ip':
                result_df_list.append(TestDesignPt._test_sche_analysis(workloads,utils,sche_config,s,StrategyPersist))
            elif s== 'ir':
                result_df_list.append(TestDesignPt._test_sche_analysis(workloads,utils,sche_config,s,StrategyRecompute))
            elif s== 'if':
                result_df_list.append(TestDesignPt._test_sche_analysis(workloads,utils,sche_config,s,StrategyFlexible))
            elif s== 'ip-ppp':
                result_df_list.append(TestDesignPt._test_PPP(workloads,utils,sche_config,s,StrategyPersistPPP))
            elif s== 'ir-ppp':
                result_df_list.append(TestDesignPt._test_PPP(workloads,utils,sche_config,s,StrategyRecomputePPP))
            elif s== 'if-ppp':
                result_df_list.append(TestDesignPt._test_PPP(workloads,utils,sche_config,s,StrategyFlexiblePPP))
        return pd.concat(result_df_list,axis=1)

    def _test_sche_analysis(workloads:List[Workload], utils, sche_config:ScheConfig, strategy:str,strategy_cls:Type[PreemptionStrategy]):
        """test one of the strategy, suit for strategies w/o PPP
        Input: workload, utils, sche_config,strategy
        The strategy should be a class inherited from PreemptionStrategy, as shown in apply_strategy.py
        Output: 3x1 pd.dataframe"""
        #apply strategies
        strategies = []
        for workload in workloads:
            s = strategy_cls()
            s.from_workload(workload)
            strategies.append(s)
        #form taskset
        ts = AccTaskset(strategies,utils)
        #schedulability analysis
        ana = schedulability_analyzer(ts)
        ana.schedulability_test()
        if ana.sche_test_success:
            return pd.DataFrame(
                                {strategy: [True, False, True]},
                                index=['sche_success', 'ppp_success', 'sim_success']
                            ) #guarantee to pass sim, skip simulation for perf
        else:
            #dump taskset for simulation
            ts_sim = AccTasksetSim(ts)
            sim_time = lcm(ts_sim.periods)
            sim_manager = SimManager(sche_config,ts_sim,sim_time=sim_time)
            sim_success = sim_manager.run()
            return pd.DataFrame(
                                {strategy: [False, False, sim_success]},
                                index=['sche_success', 'ppp_success', 'sim_success']
                            )
            
    def _test_PPP(workloads:List[Workload], utils, sche_config:ScheConfig,strategy:str,strategy_cls:Type[PreemptionStrategy]):
        """test one of the strategy, suit for strategies w/t PPP
        Input: workload, utils, sche_config,strategy
        The strategy should be a class inherited from PreemptionStrategy, as shown in apply_strategy.py
        Output: TestResult obj"""
        #apply strategies
        strategies = []
        for workload in workloads:
            s = strategy_cls()
            s.from_workload(workload)
            strategies.append(s)
        #form taskset
        ts = AccTaskset(strategies,utils)
        #PP placement
        ppp = PP_placer(ts)
        ppp.PP_placement()
        if ppp.PPP_success:
            return pd.DataFrame(
                                {strategy: [True, True, True]},
                                index=['sche_success', 'ppp_success', 'sim_success']
                            ) #guarantee to pass sim, skip simulation for perf
        else:
            #dump taskset for simulation
            ts_sim = AccTasksetSim(ts)
            sim_time = lcm(ts_sim.periods)
            sim_manager = SimManager(sche_config,ts_sim,sim_time=sim_time)
            sim_success = sim_manager.run()
            return pd.DataFrame(
                                {strategy: [False, False, sim_success]},
                                index=['sche_success', 'ppp_success', 'sim_success']
                            )

class Searcher:
    """search (1)same DNN types, (2)different utils, (3)different strategies and static the success rate
    generate/load some util distributions us uunifast algorithm, then test the DNN shape on it
    Use a workspace folder to handle the input and output, 
        (1) the workspace is not a dir: create and save all configs 
        (2) the workspace is a valid dir: use existing configs"""
    def __init__(self,acc_config_path:str,sche_config_path:str,
                 DNN_shapes:List[List[List[int]]],
                 utils:list = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1],#total utils
                 num_util = 100,#points generate for each util
                 workspace:str = 'search_results'):
        self.workspace = workspace
        if not os.path.exists(self.workspace): #begin from scarch
            os.makedirs(self.workspace)
            # Copy config files into workspace with renamed files
            self.acc_config = AccConfig.from_json(acc_config_path)
            self.sche_config = ScheConfig.from_json(sche_config_path)
            self.DNN_shapes = DNN_shapes
            self.utils = utils
            self.num_util = num_util
            shutil.copy(acc_config_path, os.path.join(self.workspace, 'acc_config.json'))
            shutil.copy(sche_config_path, os.path.join(self.workspace, 'sche_config.json'))
            # dump configs to json files
            self._save_json(self.DNN_shapes,'DNN_shapes.json')
            self._save_json(self.utils,'utils.json')
            # generate 
            self.util_dict = self._gen_utils()
            self._save_json(self.util_dict,'util_dict.json')
        else: #load from existing workspace
            self.acc_config = AccConfig.from_json(os.path.join(self.workspace, 'acc_config.json'))
            self.sche_config = ScheConfig.from_json(os.path.join(self.workspace, 'sche_config.json'))
            self.DNN_shapes = self._load_json('DNN_shapes.json')
            self.utils = self._load_json('utils.json')
            self.util_dict = self._load_json('util_dict.json')     
    def run(self):
        #init run_work
        run_works = []
        for total_util, u_list in self.util_dict.items():
            for u in u_list:
                run_works.append((total_util,u))
        def worker(task):
            total_util, u = task
            result = TestDesignPt.test(self.DNN_shapes,u,self.acc_config,self.sche_config)
            return (total_util, u, result)
        
        raw_results = []
        with ProcessPoolExecutor(max_workers=None) as executor:
            futures = [executor.submit(worker, run_work) for run_work in run_works]
            for future in as_completed(futures):
                raw_result = future.result()
                raw_results.append(raw_result)
        # Group results by total_util
        accum_results = {}
        for total_util, u, df in raw_results:
            if total_util not in accum_results:
                accum_results[total_util] = df.astype(int).copy()
            else:
                # Example accumulation: sum boolean DataFrames as int (True=1, False=0)
                accum_results[total_util] += df.astype(int)
        self.results_dict = accum_results
        return accum_results

    def _save_json(self, data, filename):
        with open(os.path.join(self.workspace, filename), 'w') as f:
            json.dump(data, f)  
    def _load_json(self, filename):
        with open(os.path.join(self.workspace, filename), 'r') as f:
            return json.load(f)
    def _gen_utils(self):
        num_task = len (self.DNN_shapes)
        util_dict = {}
        for util in self.utils:
            util_dict[util]=[]
            for i in range(self.num_util):
                util_dict[util].append(uunifast(num_task,util))
        return util_dict




if __name__ == "__main__":
    acc_config = AccConfig.from_json('/home/shixin/RTSS2025_AE/CLARE/CLARE_SW/configs/acc_config.json')
    sche_config = ScheConfig.from_json('/home/shixin/RTSS2025_AE/CLARE/CLARE_SW/configs/sche_config.json')
    DNN = [
        [[1024,8192,1024],[1024,8192,1024]],
        [[1024,8192,1024],[1024,8192,1024]]
    ]
    result = TestDesignPt.test(DNN,[0.4,0.4],acc_config,sche_config,strategies=['np','lw','if-ppp'])
    print(result)
    result1 = TestDesignPt.test(DNN,[0.3,0.3],acc_config,sche_config,strategies=['np','lw','if-ppp'])
    print(result1)
    print(result1.astype(int)+result.astype(int))

    searcher = Searcher('/home/shixin/RTSS2025_AE/CLARE/CLARE_SW/configs/acc_config.json','/home/shixin/RTSS2025_AE/CLARE/CLARE_SW/configs/sche_config.json',
                        DNN,[0.5,0.9],5)
    result = searcher.run()
    for u, df in result.items():
        print(df)