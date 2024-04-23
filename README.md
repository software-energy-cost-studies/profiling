# profiling
This repository contains the list of instructions and results for profiling of baler (data compression library):

### Dependencies installation:
```
pip3 install -r requirements.txt
```

### CPU profilers:

#### Profiling using cProfile
```
poetry run python3 -m cProfile -o cpu_profile/cProfile_train.prof  -m  baler --project CMS_workspace CMS_project_v1  --mode train
poetry run python3 -m cProfile -o cpu_profile/cProfile_compress.prof  -m  baler --project CMS_workspace CMS_project_v1  --mode compress
poetry run python3 -m cProfile -o cpu_profile/cProfile_decompress.prof  -m  baler --project CMS_workspace CMS_project_v1  --mode decompress
```
Generate the report using Shakeviz
```
cd ./cpu_profile
snakeviz cProfile_train.prof
snakeviz cProfile_compress.prof 
snakeviz cProfile_compress.prof 
```



#### Scalene provides the easy to read report about the CPU, GPU, memory, and counts (number of memory-expensive operations that appears during copying objects 
```
poetry run python3 -m scalene --profile-all baler/baler.py  --project CMS_workspace CMS_project_v1  --mode train
poetry run python3 -m scalene --profile-all baler/baler.py  --project CMS_workspace CMS_project_v1  --mode decompress
poetry run python3 -m scalene --profile-all baler/baler.py  --project CMS_workspace CMS_project_v1  --mode compress
```

#### py-spy
```
py-spy top -- python3 baler.py --mode train --project profile cpu
py-spy top -- python3 baler.py --mode compress --project profile cpu
py-spy top -- python3 baler.py --mode decompress --project profile cpu
```

### Run the torch profiler:
Installation:
```
pip install torch_tb_profiler
```

In code:
```
prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/baler'),
        record_shapes=True,
        with_stack=True)
prof.start()
#The training code here
#.....................
prof.stop()
```

In order to see the traced results of the profiler:
```
tensorboard --logdir=./log
```

### Energy cost estimation using energy meters:

#### Zeus:
Installation:
```
pip install zeus-ml
```
Usage:
```
from zeus.monitor import ZeusMontior
# Time/Energy measurements for four GPUs will begin and end at the same time.
gpu_indices = [0]
monitor = ZeusMonitor(gpu_indices)

# Mark the beginning of a measurement window. You can use any string
# as the window name, but make sure it's unique.
monitor.begin_window("entire_training")

# Actual work
#Call the Training function here:

# Mark the end of a measurement window and retrieve the measurement result.
result = monitor.end_window("entire_training")

# Print the measurement result.
time_consumed, energy_consumed = prof_result.time, prof_result.energy
print(f"Training took {time_consumed} seconds.")
for gpu_idx, gpu_energy in zip(gpu_indices, energy_consumed):
    print(f"GPU {gpu_idx} consumed {gpu_energy} Joules.")

```
![alt text](https://github.com/software-energy-cost-studies/profiling/blob/30994ba2132905c428a60807ddd894d36e37819e/results/lxplus/gpu/zeus/gpu_energy_zeus.png)
![alt text](https://github.com/software-energy-cost-studies/profiling/blob/30994ba2132905c428a60807ddd894d36e37819e/results/lxplus/gpu/zeus/duration_zeus.png)


#### CodeCarbon
Installation:
```
pip install codecarbon
```

```
from codecarbon import track_emissions
@track_emissions()
def your_function_to_track():
```
![alt text](https://github.com/software-energy-cost-studies/profiling/blob/30994ba2132905c428a60807ddd894d36e37819e/results/lxplus/gpu/codecarbon/cpu_code_carbon.png)
![alt text](https://github.com/software-energy-cost-studies/profiling/blob/30994ba2132905c428a60807ddd894d36e37819e/results/lxplus/gpu/codecarbon/gpu_energy_code_carbon.png)

#### Eco2AI
Installation:
```
pip install eco2ai
```
```
tracker = eco2ai.Tracker(
    project_name="baler", 
    experiment_description="baler_training",
    file_name="baler_training_emmision.csv"
    )

tracker.start()
#Call the function here
tracker.stop()
```

![alt text](https://github.com/software-energy-cost-studies/profiling/blob/5c661ced16d7136ca6d466e02cb6bbd9166c4da2/results/lxplus/gpu/eco2ai/co2_emission.png)
![alt text](https://github.com/software-energy-cost-studies/profiling/blob/5c661ced16d7136ca6d466e02cb6bbd9166c4da2/results/lxplus/gpu/eco2ai/duration.png)

### GPU profilers:
1. [scalene](https://github.com/plasma-umass/scalene)
2. [PyProf](https://github.com/adityaiitb/PyProf)
3. [Nvidia-DLProf](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/)
#### lxplus:
```
 /opt/nvidia/nsight-systems/2023.2.3/bin/nsys  profile  -f true -c cudaProfilerApi  --export sqlite python <code2profile.py>
```

### List of the framework for energy cost estimation:
1. [boagent](https://github.com/Boavizta/boagent)
2. [powermeter](https://github.com/autoai-incubator/powermeter)
3. [powerjoular](https://gitlab.com/joular/powerjoular)
4. [scaphandre](https://github.com/hubblo-org/scaphandre)
   Scaphandre installation (https://hubblo-org.github.io/scaphandre-documentation/index.html)
6. [AIPowerMeter](https://github.com/GreenAI-Uppa/AIPowerMeter)


### List of the framework for C02 emissions estimation:

1. [green-ai](https://github.com/daviddao/green-ai)
2. [Codecarbon](https://github.com/mlco2/codecarbon)
3. [Eco2AI](https://github.com/sb-ai-lab/Eco2AI)
4. [CarbonAI](https://github.com/Capgemini-Invent-France/CarbonAI)
5. [carbontracker](https://github.com/lfwa/carbontracker)
6. [tracarbon](https://github.com/fvaleye/tracarbon)


#### Metrics that used for Energy Cost Analysis
| Metrics | #Definition   | #Link    |
| :---:   | :---: | :---: |
| Greeness | Accuracy/(Average number of inferences in each lifecycle * Inference energy cost + Train Energy Cost)   | TODO   |
| CO2 emission [kg] |  (the power consumption of the hardware (𝑃)) * (the training time (𝑇 )) * (the carbon intensity of the energy grid (𝐼) )  | TODO   |
| Energy [kWh] | Total Energy Consumed by Program   | TODO   |
| FLOPS| Number of floating points operation | TODO   |
| Number of Model Parameters and Data Points| Number of the trainable variables    | TODO   |
| Run Time [s] | Total Time Spent by Program   | TODO   |
| PUE (Power Usage Effectiveness) | (Total Facility Power) / (IT Equipment Power) | TODO   |


#### Benchmarks and Dataset 
[1] https://mlcommons.org/


