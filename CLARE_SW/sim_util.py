"""
This script use simpy to run the simulation:
Key simulation structure:
|-------------------|
|   job_generator   |
|-------------------|
     |
     | job_release
     |
     V
|-------------------|
|   scheduler       |
|-------------------|
  |              ^
  |instr         |feedback
  |              |
  V              |
|-------------------|
|   accelerator     |
|-------------------|
Three modules run in parallel:
job_generator:
    - generate jobs for each task periodically(may add sporadic in the future)
scheduler:
    - handles 3 events in a fifo manner:
    - get feedback:
        - process the finished region:
            - change the status of the corresponding job
            - if the job is finished, (1) check deadline compliance (2)record
        - allows the scheduler to issue next region
        - pay latency
    - handle job release:
        - add the newly released jobs to the heap
        - pay latency of each added job
    - release task:
        - releases task for exec and pay latency
        - if the job is finished, (1) remove the job from the heap and (2) pay heap latency for selecting the new job
acc:
    - wait for the instr
    - once get instr:
        - when preemption happens: pay swap-out ovhd
        - when resume happens: pay swap-in ovhd
        - pay execution time
    - send feedback when finish
"""


from parse_workload import AccConfig, Workload
from apply_strategy import *
from schedulability_analysis import AccRegion, AccTask, AccTaskset, schedulability_analyzer, PP_placer
from utils import debug_print
from copy import deepcopy
import json
import simpy
import logging
from typing import List

class AccRegionSim:
    """static class, caches the info needed in a region to optimize simulation"""
    def __init__(self, region:AccRegion):
        self.exec_time = region.exec_time
        self.wcet = region.wcet
        self.ovhd = region.ovhd
        self.si = region.si
        self.so = region.so
    def __repr__(self):
        return (f"AccRegionSim(exec_time={self.exec_time}, "
                f"wcet={self.wcet}, ovhd={self.ovhd}, "
                f"si={self.si}, so={self.so})")

class AccTaskSim:
    """static class, caches the info needed in a region to optimize simulation"""
    def __init__(self, task:AccTask):
        self.ID = task.ID
        self.period = None
        self.regions = []
        self.exec_time = task.exec_time
        self.wcet = task.wcet
        for region in task.regions:
            self.regions.append(AccRegionSim(region))
        self.num_region=len(self.regions)
    def printNPR(self):
        header = f"AccTaskSim(ID={self.ID}, period={self.period}, exec_time={self.exec_time}, wcet={self.wcet})\n"
        region_header = f"{'Idx':<4} {'exec_time':<10} {'wcet':<10} {'ovhd':<10} {'si':<10} {'so':<10}\n"
        region_lines = []
        for i, region in enumerate(self.regions):
            line = f"{i:<4} {region.exec_time:<10} {region.wcet:<10} {region.ovhd:<10} {region.si:<10} {region.so:<10}"
            region_lines.append(line)
        return header + region_header + "\n".join(region_lines)
    def __repr__(self):
        header = f"AccTaskSim(ID={self.ID}, period={self.period}, exec_time={self.exec_time}, wcet={self.wcet})\n"
        region_chain = " -> ".join(
            f"({r.si}|{r.exec_time}|{r.so})" for r in self.regions
        )
        return header + region_chain

class AccTasksetSim:
    def __init__(self, taskset:AccTaskset):
        self.tasks = []
        for task in taskset.tasks:
            self.tasks.append(AccTaskSim(task))
        self.utils = taskset.utils
        self.periods = taskset.periods
        self.sche_test_success = taskset.sche_test_success
        self.PPP_success = taskset.PPP_success
        #add periods to each task
        for idx, task in enumerate(self.tasks):
            task:AccTaskSim
            task.period = self.periods[idx]

    def __repr__(self):
        header = (
            f"AccTasksetSim(utils={self.utils}, periods={self.periods}, "
            f"sche_test_success={self.sche_test_success}, PPP_success={self.PPP_success})\n"
        )
        task_reprs = "\n".join(repr(task) for task in self.tasks)
        return header + task_reprs

class ScheConfig:
    """Scheduler configuration parameters."""
    def __init__(self, feed_back_latency:int, task_release_latency:int,
                 issue_instr_latency:int, task_finish_latency:int,
                 heap_top_down_depth:int, heap_top_down_II:int,
                 heap_bottom_up_depth:int, heap_bottom_up_II:int):
        self.feed_back_latency = feed_back_latency
        self.task_release_latency = task_release_latency
        self.issue_instr_latency = issue_instr_latency
        self.task_finish_latency = task_finish_latency
        self.heap_top_down_depth = heap_top_down_depth
        self.heap_top_down_II = heap_top_down_II
        self.heap_bottom_up_depth = heap_bottom_up_depth
        self.heap_bottom_up_II = heap_bottom_up_II

    @classmethod
    def from_json(cls, filepath:str):
        """Load scheduler configuration from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        # filter out any comment fields
        filtered = {k: v for k, v in data.items() if not k.startswith('_')}
        return cls(**filtered)

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)

class JobSim:
    """data class representing a job in simulation"""
    def __init__(self,task,job_id:int,release:int,ddl:int):
        self.task = task #task
        self.job_id = job_id
        self.region = 0 #the **next** region of this job to issue, idx begin from 0
        self.release = release
        self.ddl = ddl #release and end cycle
        self.process = None #first time be processed
        self.issue = [] #cycles of issue
        self.feedback = [] #cycles of feedbacks
    
class JobGenerator:
    def __init__(self,env:simpy.Environment,taskset:AccTasksetSim,
                 task_release_fifo:simpy.Store,
                 logger:logging.Logger):
        self.env:simpy.Environment = env
        self.taskset:AccTasksetSim = taskset
        self.logger:logging.Logger = logger
        self.task_release_fifo:simpy.Store = task_release_fifo
        self.job_id:int = 0
        self._run()
    def _generate_job(self,env:simpy.Environment,
                      task:AccTaskSim,fifo:simpy.Store,
                      logger:logging.Logger):
        task_id = task.ID
        period = task.period
        while True:
            job_id = self.job_id
            self.job_id +=1
            release = env.now
            ddl = release + period
            new_job = JobSim(
                task=task_id,
                job_id=job_id,
                release = release,
                ddl= ddl 
            )
            logger.debug("[{}][JobGen] task released: task={}, Job={}, ddl={}".format(
                release, task_id, job_id, ddl
            ))
            yield fifo.put(new_job)
            yield self.env.timeout(period) #after a period, run again for periodically release
    def _run(self):
        for task in self.taskset.tasks:
            self.env.process(self._generate_job(self.env,task,self.task_release_fifo,self.logger))

class Instr:
    """data class for the instruction channel"""
    def __init__(self,job:JobSim,region:int,preempt:bool,resume:bool,
                 last_task:int,last_region:int):
        self.job:JobSim = None #pass the job to track the whole lifecycle
        #the task,job,release,ddl info is in the job obj
        self.region:int = None #region to execute
        self.preempt:bool = None
        self.resume:bool = None #instr to conduct preemption
        self.last_task:int = None
        self.last_region:int = None #used for conduct swap-in

class Feedback:
    def __init__(self):
        pass

class scheduler:
    """simulate the heap behavior:
    - for performance, use a list to represent the heap, and assume the heap operation always take the worst
    - In HW implementation, seperate FIFOs are used to handle the job release, here a unified fifo are used
    - still, the priority of event handling(feedback>task_release>issue instr) is the same, thus the latency is bounded
    """
    def __init__(self,sche_config:ScheConfig,
                 env:simpy.Environment,taskset:AccTasksetSim,
                 task_release_fifo:simpy.Store,
                 instr_fifo:simpy.Store,
                 feedback_fifo:simpy.Store,
                 logger:logging.Logger):
        self.sche_config:ScheConfig = sche_config
        self.env:simpy.Environment = env
        self.taskset:AccTasksetSim = task
        self.task_release_fifo:simpy.Store = task_release_fifo
        self.instr_fifo:simpy.Store = instr_fifo
        self.feedback_fifo:simpy.Store = feedback_fifo
        self.logger:logging.Logger = logger
        self.heap:List[JobSim] = []#use a list to represent the heap

    def _schedule(self):  
        #compute heap op latency
        num_task = len(self.taskset.tasks)
        max_heap_top_down_latency = self.sche_config.heap_top_down_depth \
            + num_task * self.sche_config.heap_top_down_II
        max_heap_bottom_up_latency = self.sche_config.heap_bottom_up_depth \
            + num_task * self.sche_config.heap_bottom_up_II
        #record current task and region
        #For init, since at t=0, it's always the task with the smallest period issued
        #point to that task
        min_task:AccTaskSim = min(self.taskset.tasks, key=lambda task:task.period)
        cur_task = min_task.ID
        cur_region =0
        
        #initialize
        issue_flag = True
        #register the first two events for listening
        task_release_evt = self.task_release_fifo.get()
        feedback_evt = self.feedback_fifo.get()
        while True:
            result = yield simpy.events.AnyOf(self.env, [task_release_evt, feedback_evt])
            """process feedback
            """
            if feedback_evt in result.events:
                ...#TODO: implement after implement acc
            """process job release
            all job within the task release fifo will be processed at once
            after each process, the latency will be paid, 
            this is compatible with the HW behavior"""
            if task_release_evt in result.events:
                job:JobSim = task_release_evt.value #get the job
                self.heap.append(job)
                #in HW, here should conduct sort, in Sim we sort only when release job, but pay sort/heap latency here
                #pay operation latency
                self.logger.debug("[{}][Sche] Job added to heap: Task={}, Job={}".format(
                    self.env.now, job.task, job.job_id))
                yield self.env.timeout(max_heap_bottom_up_latency)
                #check and process all other events
                while self.task_release_fifo.items:
                    job=self.task_release_fifo.items.pop(0)
                    self.heap.append(job)
                    self.logger.debug("[{}][Sche] Job added to heap: Task={}, Job={}".format(
                    self.env.now, job.task, job.job_id))
                    yield self.env.timeout(max_heap_bottom_up_latency)
                #register the event for listening to the feedback next time
                task_release_evt = self.task_release_fifo.get()
            """issue instruction
            it's ok if the flag is true but heap is empty, 
            since to join something to the heap, a job must be released,
            thus next time a job is released and add to the heap, 
            the code will pass the yield and issue the instr can run correctly
            """
            if issue_flag and self.heap:
                self.heap.sort(key=lambda job:job.ddl)#EDF
                new_job=self.heap[0]
                instr = Instr(new_job,
                              new_job.region,#seperate this since obj will be updated
                                
                              )






            



if __name__ == '__main__':
    config = AccConfig.from_json("/home/shixin/RTSS2025_AE/CLARE/CLARE_SW/configs/acc_config.json")
    # w1=Workload()
    debug_print('decompose_NN')
    # w1.decompose_NN([[1024,8192,1024],[1024,8192,1024]],config)
    w1=Workload([[1024,8192,1024],[1024,8192,1024]],config,'Task1')
    debug_print('apply strategy')
    s1 = StrategyFlexible()
    # s1 = StrategyNonPreemptive()
    # s1 = StrategyLayerwise()
    s1.from_workload(w1)
    print(s1.ID)

    w2 = Workload([[1024,8192,1024],[1024,8192,1024]],config,'Task2')
    s2 = StrategyFlexible()
    s2.from_workload(w2)

    debug_print('form taskset')
    taskset = AccTaskset([s1,s2],[0.2,0.7])
    for task in taskset.tasks: print(task.ID)
    # exit()

    # debug_print('begin sche analysis')
    # ana = schedulability_analyzer(taskset)
    # ana.schedulability_test()
    # debug_print('sche analysis:',ana.sche_test_success)

    print('begin PPP')
    PPP = PP_placer(taskset)
    TS = PPP.PP_placement()
    print("PPP success:",PPP.PPP_success)
    print(TS.tasks)

    ts_sim = AccTasksetSim(TS)
    print(ts_sim)