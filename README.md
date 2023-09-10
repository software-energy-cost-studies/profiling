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
scalene --cpu --gpu --memory baler.py --mode train --project profile cpu
scalene --cpu --gpu --memory baler.py --mode compress --project profile cpu
scalene --cpu --gpu --memory baler.py --mode decompress --project profile cpu
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

### GPU profilers:
1. [experiment-impact-tracker](https://github.com/Breakend/experiment-impact-tracker)
2. [scalene](https://github.com/plasma-umass/scalene)
3. [PyProf](https://github.com/adityaiitb/PyProf)
4. [Nvidia Nsight]()
5. [Nvidia-DLProf](https://docs.nvidia.com/deeplearning/frameworks/dlprof-user-guide/)

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
