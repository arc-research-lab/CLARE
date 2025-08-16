CLARE software stack conducts the workload parsing & segmentation, quantifying execution, scheduling & preemption operation latencies, schedulability analysis and preemption point placement.
A simulation script comparing the scheduling success rate between CLARE and the baseline scheduling methods are also provided.


# brief description for CLARE task modeling:
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

# Components explaination
For more detailed explanation of the source codes, please refer to the comments within the code.
## parse_workload.py
Input workload format: A list, each element is a list containing the shape(MNK) of an Martix Multiplication(MM)
- e.g. [[256,256,256],[256,512,256]] describe a 2-layer MLP of shape 256x256x256 and 256x512x256
This scripts will parse the inputed workload shape, then:
1. compute the tile iterations (#tiles in M,K,N dim)
2. tag the attribute of each tile
    - start & end of a piece of output
    - start & end of a layer
    - start & end of a model
3. compute the cycles used in each tile (load/comp/store/load/persist)
    - i.e. the time for every possible preemption points(PPs)
    - Which PP will be enabled will be decided later by using the PPP algorithm(ours) or heuristics(baseline)
- `AccConfig`
    - contains the attribute of the accelerator
    - use ... = AccConfig.from_json() to init
- `AccIter`
    - Stores the info of each AccIteration when computing the DNN.
    - A iteration is the smallest segements of execution in CLARE, preemption can happen only betwwen two iterations
    - The load/comp/store pipeline are represented in the iteration, #Iter = #tile(x*y*z)+2, with different load/comp/store latency
    - The tiling are represented in the iterations, different iterations has different preemption/resume latency
- `Workload`
    - stores all the execution latency information of a DNN model
    - use `__init__()` or `decompose_NN()` to load from a list of workload, converting the shape into a sequence of `AccIter()`

## apply_strategy.py
This script add different preemption strategy evaluated in CLARE:
- Input: parsed Workload class
- Based on selected strategies, change the 'is_preemptive' and 'strategy' attribute of each iteration within one class
    - preemption strategy: how to do the PP **before** this segment (persist or recompute)
    - is_preemptive: if the workload is preemptive **after** this segment
- Merge the non-preemptive iterations into metadata to be executed
- Output: Task_metadata class
- implemented preemption strategy:
    - non-preemptive (np)
    - layerwise-preemptive (lw)
    - intra-layer preemptive, recompute only, w/o preemption point placement(PPP) (ir)
    - intra-layer preemptive, persist only, w/o PPP (ip)
    - intra-layer preemptive, flexible, w/o PPP (if)
    - intra-layer preemptive, recompute only, w/t PPP (ir-PPP)
    - intra-layer preemptive, persist only, w/t PPP (ip-PPP)
    - intra-layer preemptive, flexible, w/t PPP (if-PPP)
Using the proposed heuristic, the strategy of each PP can be defined before schedulability PP placement

- `PPStrategy`
    - a enum class for different PP implementation
    - Note that different import methods will affect the comparison between two enum values.

- `PreemptionStrategy`
    - base class for all preemption strategies
    - use `__init__()` or `from_workload()` to apply strategies to a `Workload()`
    - Do not use this class directly, instead, use the child classes
    - child class including:
        - `StrategyNonPreemptive`
        - `StrategyLayerwise`
        - `StrategyRecompute`
        - `StrategyPersist`
        - `StrategyFlexible`
        - `StrategyRecomputePPP`
        - `StrategyPersistPPP`
        - `StrategyFlexiblePPP`

## schedulability_analysis.py
This script intakes the workloads that is parsed and applied strategy, and
- combine strategy classes into a whole taskset
    - each strategy class reflects a task
    - period of each tasks are required to input
    - the ddl of each task d == period, i.e. a job(instance of task) is required to be finished before its successor releases
- form the swap-in and swap-out operation latency to be compatible with the real-time theory
- conduct schedulability analysis to the task set --> the task set can meet ddl or not
- for the intra-layer preemptive model, also conduct preemption point placement algorithm to remove the redundant PPs, a new taskset after PPP will be generated if successful
- generate:
    - (1) the metadata for accelerator, used in simulation and accelerator execution
    - (2) the schedulability analysis results

- `AccRegion`, `AccTask`, `AccTaskset`
    - classes to represent a nun-preemptive region, a task and a whole taskset in the schedulability analysis
    - the attributes are compatible to the task modeling as described in clare paper
    - use the `AccTaskset.__init__()` to construct a task from a list of `PreemptionStrategy()` objs
-  `schedulability_analyzer`
    - a class contains the functions used in schedulability analysis
    - In CLARE, the task/regions are indexed beginning with 1, here they're convert to 0-indexed
    - use `__init__()` to load from a `AccTaskset`.
    - use `schedulability_test()` to conduct the schedulability test
    - is success, `self.sche_test_success` will be set to True
- `PP_placer`
    - inherit from schedulability analyzer, group functions for PP placement
    - use `__init__()` to load from a `AccTaskset`
    - tries to remove reduant PP by merging the NPRs, the WCET is thus saved.
    - if PPP success `self.PPP_success` will be set to True. `self.sche_test_success` will also be set to True
    - when PPP success. `self.TS` will contain the taskset after PP placement and mergeing NPRs
    - when PPP fails, `self.TS` will be meaningless and shouldn't be used in later steps

## sim_utils.py
This script use simpy to run the simulation:
Key simulation structure:
```
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
```
- `SimManager`
    - should use this class for running the simulation
    - init and handles the simpy env
    - also handles a logger for print results
- `JobGenerator`, `Scheduler`, and `Accelerator` will run simutanously
- `JobGenerator`:
    - generate jobs for each task periodically(may add sporadic in the future)
- `Scheduler`:
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
- `Accelerator`:
    - wait for the instr
    - once get instr:
        - when preemption happens: pay swap-out ovhd
        - when resume happens: pay swap-in ovhd
        - pay execution time
    - send feedback when finish
- `JobSim`
    - data class representing a job in simulation
    - this obj will be generated by `JobGenerator` and passed to `Scheduler`. Each `Instr` of this job will keep a referce to the object to track the lifecycle of this job
- `Instr`
    - data class for the instruction channel
- `Feedback`
    - small data class notifying the execution is finished 
- `ScheConfig`
    - Scheduler configuration parameters.
    - use `ScheConfig.from_json()` to load data


## search.py
This script is used for conduct schedulability analysis, PPP, and simulation in larger scale
The input DNN shape is fixed, and the searcher sweeps through different total utils and strategies

- `TestDesignPt`
    - static class grouping series of funcs for test the design point at 1 DNN shape, 1 acc and sche config, 1 utilization distribution and 1 strategy
    - use `test()` to analyze one design point
    - input: 
            - DNN_shapes(list): 3-d list: task-layer-M,K,N shape of the layer
            - utilizations(list), 
            - acc_config(AccConfig), sche_config(ScheConfig)
            - strategies(str):
    - Output:
            - pd.dataframe:
                - 3-rows for sche_analysis, PPP, and simulation success 
                - one col for the strategy
    - Note the result data processing will be handled by the warpper of this func
- `Searcher`
    - search (1)same DNN types, (2)different utils, (3)different strategies and static the success rate
    - generate/load some util distributions us uunifast algorithm, then test the DNN shape on it
    - Use a workspace folder to handle the input and output, 
        1. If the workspace is not a dir: create and save all configs 
        2. If the workspace is a valid dir: use existing configs
    - use `__init__()` for a group of design
    - use `run()` to conduct analysis, multi-processing is used via `ProcessPoolExecutor`
    - saves the raw result and accumulated results in the `workspace` dir using `pickle`. Saves the input configs and final outputs
- `comp_WCET()`
    - For the same DNN shapes, sweep through different utils and strategies, compare the WCET generated by different stategies
    - for the strategy without PPP, use the taskset's wcet directly
    - for the strategy with PPP:
        - if PPP succeed, return the wcet after placement
        - else, return the wcet of the original taskset
    - return the wcet of the task with the largest period 
## utils.py
stores the util functions
- `print_iters()`
    - Print a list of AccIter object in pandas dataframe manner
    - used in `Workload`,`PreemptionStrategy`,`AccTaskset`
- `lcm_pair()`, `lcm()`
    - compute the least common multiple of 2/a list of numbers
- `debug_print()`
    - print the calling frame and script line number when printing
- `init_logger()`
    - init a logging.logger with level, output and enabling options
- `uunifast()`
    - based on (1) number of tasks (2) total utilization of the tasks, distribute the utilization for each task
    - Return a List of utilizations
    - For ease of reading, sort the utilizations from large to small, this will not affect the randomness
- `gen_transformer()`
    - generate DNN shapes for transformer base on give parameters
- `gen_deit_t()`, `gen_bert_t()`, `gen_bert_mi()`, `gen_mlp_mixer()`, `gen_pointnet()`
    - generate a list representing DNN workloads for a realistic workloads