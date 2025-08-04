"""
This script intakes the workloads that is parsed and applied strategy, and
- combine strategy classes into a whole taskset
    - each strategy class reflects a task
    - period of each tasks are required to input
    - the ddl of each task d == period, i.e. a job(instance of task) is required to be finished before its successor releases
- form the swap-in and swap-out operation latency to be compatible with the real-time theory
- conduct schedulability analysis to the task set --> the task set can meet ddl or not
- for the intra-layer preemptive model, also conduct preemption point placement algorithm to remove the redundant PPs
- generate:
    - (1) the metadata for accelerator, used in simulation and accelerator execution
    - (2) the schedulability analysis results
"""
"""
brief description for CLARE task modeling:
- A **task set**: several tasks
- A **task**: releases infinite instances of this task periodically
    - attributes: period(p), worst-case execution time(WCET)(e), 
- A **job**: one instance of task, every job is required to be finished before its deadline
- **Non-preemptive Region**: each task/job consists several Non-preemptive region where preemption can only happen between regions
    - attributes: WCET(b), preemption overhead(xi), xi is the overhead **before** each region
"""
from utils import print_iters
from typing import List
from copy import deepcopy
from parse_workload import AccIter, Workload, AccConfig
from apply_strategy import *
from utils import print_iters


class AccRegion:
    """
    One non-preemptive region in schedulability analysis
    """
    def __init__(self, WCET:int=0, ovhd:int=0, iters:List[AccIter]=None):
        self.WCET = WCET
        self.ovhd = ovhd
        self.iters = iters if iters is not None else [] #the acc iteration within this region, used to generate execution schedule
    def print_iters(self):
        print_iters(self,['layer','idx','is_preemptive','strategy'])

class AccTask:
    """
    One Task in the schedulability analysis
    """
    def __init__(self, strategy=None):
        self.regions = []
        self._from_strategy(strategy)
        self._comp_resume_ovhd()
        self._comp_wcet()

    def _from_strategy(self, strategy):
        """decouple the workload into NPRs"""
        assert isinstance(strategy, PreemptionStrategy), "[Acctask.from_strategy]:the input strategy must be subclass of PreemptionStrategy"
        s = deepcopy(strategy)
        iters_in_region = []
        for idx,iter in enumerate(s.iters):
            iter:AccIter
            iters_in_region.append(iter)
            if iter.is_preemptive or idx==len(s.iters)-1: #if is preemptive **after this iter**
                #create a NPR based on iters_in_region
                self.regions.append(AccRegion(iters=iters_in_region))
                #clear the buffer for next NPR
                iters_in_region = []
    
    def _comp_resume_ovhd(self):
        """compute the resume ovhd, i.e., the ovhd of 2nd+ regions, 
        the ovhd of the first region is the preemption ovhd(when this task preempt others)
        and the preemption ovhd will be affected by other tasks"""
        for idx, region in enumerate(self.regions):
            #skip the first region
            region:AccRegion
            if idx==0:
                continue
            #use the first iter to determine the ovhd 
            first_iter = region.iters[0]
            first_iter:AccIter
            if first_iter.strategy == PPStrategy.NA:
                raise ValueError("[AccTask._comp_resume_ovhd]:the first iter uses a NA PPStrategy, this may suggest an invalid parsing of workload or NPR")
            elif first_iter.strategy == PPStrategy.layer:
                region.ovhd = 0
            elif first_iter.strategy == PPStrategy.recomp:
                region.ovhd = first_iter.si_r
            elif first_iter.strategy == PPStrategy.persist:
                region.ovhd = first_iter.si_p
    
    def _comp_wcet(self):
        for region in self.regions:
            region:AccRegion
            exec = 0
            for iter in region.iters:
                exec += iter.exec
            region.WCET = exec
    
    def print_iters(self):
        for npr in self.regions:
            npr:AccRegion
            npr.print_iters()
            print('--------------------------')

    def __repr__(self):
        reprs = [f"({r.ovhd}|{r.WCET})" for r in self.regions]
        return f"{' -> '.join(reprs)}"


if __name__ == '__main__':
    config = AccConfig.from_json("./configs/acc_config.json")
    # print(config)
    iter = AccIter()
    # print(iter)
    w1=Workload()
    w1.decompose_NN([[256,4096,256],[256,256,256]],config)
    # w1.comp_ovhd(config)
    # w1.print_iters(['layer','idx','load','comp','store','o_start','last_o_start','so_r','so_p','si_r','si_p'])
    s1 = StrategyLayerwise()
    s1 = StrategyFlexible()
    s1.from_workload(w1)
    s1.print_iters(['layer','idx','is_preemptive','si_r','si_p','strategy'])

    t1 = AccTask(s1)
    print(t1)

