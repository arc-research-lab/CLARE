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
- NPR, AccIter, and DNN layer:
    - one layer has >=3 iterations in the Acc(load/comp/store)
    - one NPR has >=1 iterations
    - based on the strategies and taskset, a NPR can have less/euqal/more than a layer
    - still, the basic unit of CLARE scheduling is an Acc iteration
"""
from utils import print_iters
from typing import List
from copy import deepcopy
from parse_workload import AccIter, Workload, AccConfig
from apply_strategy import *
from utils import print_iters
import math
from functools import cached_property


class AccRegion:
    """
    One non-preemptive region in schedulability analysis
    """
    def __init__(self, exec_time:int=0, ovhd:int=0, iters:List[AccIter]=None):
        self.ovhd = ovhd #the preemption ovhd before this region, used in schedulability analysis
        self.so = 0 #swap-out ovhd after this region, used in simulation
        self.si = 0 #swap-in ovhd before this region, used in simulation
        self.iters = iters if iters is not None else [] #the acc iteration within this region, used to generate execution schedule
    def print_iters(self):
        print_iters(self,['layer','idx','is_preemptive','strategy'])

    @property
    def exec_time(self):
        return sum(iter.exec for iter in self.iters)
    @property
    def wcet(self):
        return self.exec_time + self.ovhd

class AccTask:
    """
    One Task in the schedulability analysis
    """
    def __init__(self, strategy=None):
        self.regions = []
        self.ID = None # used for future updates for generating metadata
        self.period = None
        if strategy is not None:
            self._from_strategy(strategy)
            self._comp_resume_ovhd()
            self._comp_swap_op_latency()

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
    
    def _comp_swap_op_latency(self):
        #swap-in
        for idx, region in enumerate(self.regions):
            region:AccRegion
            #first:region: no swap-in
            if idx==0:
                region.si = None
            else:
                #The swap-in operation affects the first  
                first_iter = region.iters[0]
                first_iter:AccIter
                if first_iter.strategy == PPStrategy.NA:
                    raise ValueError("[AccTask._comp_resume_ovhd]:the first iter of an NPR uses a NA PPStrategy, this may suggest an invalid parsing of workload or NPR")
                elif first_iter.strategy == PPStrategy.layer:
                    region.si = 0
                elif first_iter.strategy == PPStrategy.recomp:
                    region.si = first_iter.si_r
                elif first_iter.strategy == PPStrategy.persist:
                    region.si = first_iter.si_p
        #swap-out:
        for idx,region in enumerate(self.regions):
            region:AccRegion
            #last region: no swap out
            if idx == len(self.regions)-1:
                region.so = None
            else:
                last_iter = region.iters[-1]#swap out affects the last op
                next_iter = self.regions[idx+1].iters[0]#the strategy is stored next NPR, first region
                if next_iter.strategy == PPStrategy.NA:
                    raise ValueError("[AccTask._comp_resume_ovhd]:the first iter of an NPR uses a NA PPStrategy, this may suggest an invalid parsing of workload or NPR")
                elif next_iter.strategy == PPStrategy.layer:
                    region.so = 0
                elif next_iter.strategy == PPStrategy.recomp:
                    region.so = last_iter.so_r#the overhead is kept in this NPR
                elif next_iter.strategy == PPStrategy.persist:
                    region.so = last_iter.so_p

    @property
    def exec_time(self):
        return sum(region.exec_time for region in self.regions)

    @property
    def wcet(self):
        return sum(region.wcet for region in self.regions)

    def print_iters(self):
        for npr in self.regions:
            npr:AccRegion
            npr.print_iters()
            print('--------------------------')

    def __repr__(self):
        reprs = [f"({r.ovhd}|{r.exec_time})" for r in self.regions]
        return f"{' -> '.join(reprs)}"

class AccTaskset:
    """
    A taskset is composed by several tasks, additionaly, the period info of each task is required
    Input: (1) List of Strategies (2)List of coresponding utilization
    Processing pipeline:
    - convert strategy to Task
    - compute WCET of each region
    - compute resume ovhd (first 3 steps done by AccTask)
    - compute preemption ovhd
    """
    def __init__(self, strategies:list=[], utils:list=[]):
        assert len(strategies)==len(utils), "[AccTaskset.__init__]: #strategies must == #utils"
        self.tasks:List[AccTask] = []
        self.utils = []
        self.periods = []
        self.sche_test_result = None #schedulability test result
        if len(strategies)!=0:
            self.utils = deepcopy(utils)
            self._from_strategies(strategies)
            self._comp_period()
            self._comp_preemption_ovhd()
    def _from_strategies(self, strategies):
        assert len(strategies)!=0
        for strategy in strategies:
            self.tasks.append(AccTask(strategy))
    def _comp_period(self):
        self.periods = [0]*len(self.tasks)
        for idx,task in enumerate(self.tasks):
            period = math.ceil(task.exec_time/self.utils[idx])
            self.periods[idx]=period
    @property
    def sorted_tasks(self):
        assert len(self.periods)==len(self.tasks),'[AccTaskset.sorted_tasks]:unmatching #periods and #tasks'
        return sorted(zip(self.tasks, self.periods), key=lambda x: x[1])
    def _comp_preemption_ovhd(self):
        for idx, (task,_) in enumerate(self.sorted_tasks):
            #after sorting, the tasks can only preempt other tasks with higher idx
            #the preemption ovhd of one task, is the largest so ovhd of tasks with higher operations
            task:AccTask
            max_so = 0
            for preempted_task, _ in self.sorted_tasks[idx+1:]:
                preempted_task:AccTask
                for preempted_region in preempted_task.regions:
                    preempted_region:AccRegion
                    if preempted_region.so is not None:
                        if preempted_region.so>max_so:
                            max_so = preempted_region.so
            task.regions[0].ovhd=max_so
    def schedulability_analysis(self):
        """Based on the execution time(b), preemption overhead(xi), periods(p) 
        determine if the taskset can meet deadline even in the worst case"""

class schedulability_analyzer_():
    """a statistic class contains the functions used in schedulability analysis"""
    def _DBF(t,d_j,p_j,e_j):
        """t: time, d_j: LCM of task 1~(j-1)'s period, p_j: task j's period, e_j: task j's WCET(ovhd+exec)"""       
        return ( 1+math.floor((t-d_j)/p_j) )*e_j
    
    def _comp_q_max(taskset:AccTaskset):
        """q^max_i ranges from 1 to n+1, 
            i=1~n, q^max_i is the longest NPR in this task
            i=n+1, q^max_n+1 = 0"""
        q_max_list = []
        for idx, (task, period) in taskset.sorted_tasks:
            q_max = max(region.wcet for region in task.regions)
            q_max_list.append(q_max)
        q_max_list.append(0)
            
    def _comp_beta(taskset:AccTaskset):
        ...

    def _comp_p(taskset:AccTaskset):
        """return the period list in the taskset"""
        return [period for _,period in taskset.sorted_tasks]
    
    def _comp_d(taskset:AccTaskset):
        """d_i+1 = LCM(p1,p2,...pi), for d_k, k ranges from [2,n+1]"""
        SA = schedulability_analyzer
        p_list = SA._comp_p(taskset)
        d_list = []
        d_list.append(None)#d_1 is not defined
        for i in range(1,len(taskset.sorted_tasks)+1):
            d = math.lcm(p_list[1-1:(i-1)+1]) #e.g. d_list[2]=d_3=lcm(p1,p2)=lcm(p_list[0],p_list[1])
            d_list.append[d]
        return d_list
            
    def schedulability_test(taskset:AccTaskset):
        """Note: the therom indexes begining with 1, thus all list indexing should -1"""
        SA = schedulability_analyzer
        n = len(taskset.sorted_tasks) # #tasks
        q_max_list = SA._comp_q_max(taskset)
        beta_list = ...

class schedulability_analyzer():
    """
    a class contains the functions used in schedulability analysis
    In CLARE, the task/regions are indexed beginning with 1, here they're convert to 0-indexed
    """
    def __init__(self,taskset:AccTaskset):
        self.TS = deepcopy(taskset)

    @cached_property
    def _n(self):
        """#tasks, task ranging from [0, n-1]"""
        return len(self.TS.sorted_tasks)
    @property
    def _q_max(self):
        """q^max: max NPR WCET in a task, index ranging from (0, n]
        q^max[n] = 0"""
        q_max = [None]*(self._n+1) #index from [0,n], where q_max[0] never used
        for idx, (task, period) in self.TS.sorted_tasks:
            task:AccTask
            q_max[idx] = max(region.wcet for region in task.regions)
        q_max[self._n] = 0
        return q_max
    @cached_property 
    def _p(self):
        """periods of each task, ranging from [0,n-1]"""
        return [period for _,period in taskset.sorted_tasks]
    @cached_property
    def _d(self):
        """deadline, ranges from [0,n]
        to assist the analysis, define that d[n]=lcm(p[0],p[1],...p[n-1])
        where lcm stands for least common multiple
        in CLARE we set d[i]=p[i], i.e. a job must be finished before its successor releases"""
        d = deepcopy(self._p)
        d.append(math.lcm(self._p))
        return d  
    def _t(self,k):
        """the possible t points when computing beta:
        for computing each beta[k](ranging from [0,n-1]), the possible t satisfies:
            (1) d[k]<=t<d[k+1]
            (2) t = Gamma(t) = a*p[x]+d[x], where x ranging from [0,n-1]
        physical meaning, from this task k's ddl to next task k+1's ddl, all time instances of job release/ddl"""
        assert k>=0 and k<=self._n-1, '[sche_analyzer._t]:invalid k value'
        LB = self._d[k]
        UB = self._d[k+1]
        t = []
        for x in range(0,self._n):
            #release/ddl time instance for one job 
            #for: from dx(included) to UB(excluded), with p as step
            #if: ti is larger than LB
            tx = [ti for ti in range(self._d[x],UB,self._p[x])
                  if ti>=LB] 
            t+=tx
    @cached_property
    def _e(self):
        """
        WCET for each task, ranging from [0,n-1]
        e_i = sum(j)(b_ij + xi_ij), 
        where b_ij is the execution lengeth, xi_ij is the preemption ovhd of each task
        """
        return [task.wcet for (task,period) in taskset.sorted_tasks]
    def _DBF(self,j,t):
        """Demand Budget Func:
        j ranges from [0,n-1],t is the t's defined by Gamma(t) (in _t())"""
        assert 0<=j and j<=self._n-1, "[sche_analyzer._DBF]: j value out of range"
        DBF = 1 + math.floor((t-self._d[j])/self.p[j])
        DBF *= self._e[j]
        return DBF
    def _sum_DBF(self,t):
        """the sche analysis alway sum up the DBF func for all tasks at one time instance"""
        sum_DBF = sum(self._DBF(j,t) for j in range(0,self._n))
        return sum_DBF
    def _beta_k(self,k):
        """compute one beta point, k ranging from [0,n-1]"""
        assert 0<=k and k<=self._n-1, "[sche_analyzer._beta_k]:k value out of range"
        beta_k = min(
            t-self._sum_DBF(t) for t in self._t(k)
        )
        return beta_k
    @property
    def _beta(self):
        """beta: ???, index ranging from [0,n-1]"""
        beta = [None]*self._n
        for idx in range(0,self._n):
            beta[idx] = self._beta_k(idx)
        return beta

    def schedulability_test(self):
        """Note: the therom indexes begining with 1, thus all list indexing should -1
        for i ranges from [1,n], q^max[i] <= min(beta[k]) where k = [0,i-1]"""
        q_max = self._q_max
        beta = self._beta
        ineq_result = [q_max[i]<=min(beta[0:i]) for i in range(1,self._n+1)]
        return all(ineq_result)



if __name__ == '__main__':
    config = AccConfig.from_json("./configs/acc_config.json")
    # print(config)
    iter = AccIter()
    # print(iter)
    w1=Workload()
    w1.decompose_NN([[256,4096,256],[256,256,256]],config)
    # w1.comp_ovhd(config)
    w1.print_iters(['layer','idx','load','comp','store','o_start','last_o_start','so_r','so_p','si_r','si_p'])
    s1 = StrategyNonPreemptive()
    s1 = StrategyLayerwise()
    s1.from_workload(w1)
    # s1.print_iters(['layer','idx','is_preemptive','si_r','si_p','strategy'])

    t1 = AccTask(s1)
    # print(t1)

    taskset = AccTaskset([s1,s1],[0.1,0.2])
    print(taskset.periods)
    print(taskset.utils)
    print('!!!!!!!!!!!!!!!!!!!!!!!!')
    print(taskset.tasks)
    print('!!!!!!!!!!!!!!!!!!!!!!!!')

