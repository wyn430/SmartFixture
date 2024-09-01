# SmartFixture
Implementation of our recent paper, [SmartFixture: Physics-guided Reinforcement Learning for Automatic Fixture Layout Design in Manufacturing Systems](https://scholar.google.com/citations?user=-vfMhrEAAAAJ&hl=en).

## Abstract
Fixture layout design critically impacts the shape deformation of large-scale sheet parts and the quality of the final product in the assembly process. The existing works focus on developing mathematical-optimization (MO)-based methods to generate the optimal fixture layout via interaction finite element analysis (FEA)-based simulations or its surrogate models. Their limitations can be summarized as memorylessness and lack of scalability. Memorylessness indicates that the experience in designing the fixture layout for one part is usually not transferable to others. Scalability becomes an issue for MO-based methods when the design space of fixtures is large. Furthermore, the surrogate models might have limited representation capacity when modeling high-fidelity simulations. To address these limitations, we propose a learning-based framework, SmartFixture, to design the fixture layout by training a Reinforcement learning (RL) agent through direct interaction with the FEA-based simulations. The advantages of the proposed framework include: (1) it is generalizable to design fixture layouts for unseen scenarios after offline training; (2) it is capable of finding the optimal fixture layout over a massive search space. Experiments demonstrate that the proposed framework consistently generates the best fixture layouts that receive the smallest shape deformations on the sheet parts with different initial shape variations.

## Citation

If you find our work useful in your research, please consider citing:

## Installation

The experiment is conducted to directly have the reinforcement learning agent (on the computational server) interact with the ansys simulation (on the ansys server). The communication is enabled by the Python package [Ansys](https://pypi.org/project/pyansys/). 

### On the ANSYS server

1. The ANSYS with a valid license should be first installed on the ANSYS server.
2. The Python script [monitor.py](https://github.com/wyn430/SmartFixture/blob/master/monitor.py) should be put in the ANSYS server to launch and monitor the execution of ANSYS.
3. Modify the [monitor.py](https://github.com/wyn430/SmartFixture/blob/master/monitor.py) to specify the location of ANSYSxxx.exe and the port number that the ANSYS process will be executed on.
4. Record the IP address of the ANSYS server.

### On the computational server

The code has been tested in the following environment:

```
Ubuntu 20.04.5 LTS
python 3.6.8
CUDA 11.0
torch 1.10.2
```

## Dataset
The experiments are conducted on large-scale sheet metal or composite parts that are built and simulated in ANSYS. A detailed description of the dataset generation can be found in the paper. The generated dataset for deformed sheet parts is included [Deformed_inputs_2mm](https://github.com/wyn430/SmartFixture/tree/master/Deformed_inputs_2mm) 

## Usage

### Launch the ANSYS process.

The ANSYS instance is launched on the ANSYS server, which can be accessed remotely through the IP address and port number.
```
python monitor.py
```

### Training of non-deformed sheet part

The Reinforcement Learning agent is trained and tested on the computational server. The training and testing scripts for the fixture layout optimization of the non-deformed sheet part are included in [folder](https://github.com/wyn430/SmartFixture/tree/master/code_nondeformed_inputs_nofix).

```
python train.py [existence of the pre-trained model] [directory of pre-trained model] [pre-trained checkpoint] [IP of ANSYS Server]

e.g., continue the training from an existing model: python train.py 1 nondeformed_inputs_nofix_metal nondeformed_inputs_nofix_metal_pretrained.pth xx.3.127.xxx
      restart the training from scratch: python train.py 0 None None xx.3.127.xxx
```

### Validation and visualization of the training results on non-deformed sheet part
The validation of the trained agent is included in the [Jupyter Notebook](https://github.com/wyn430/SmartFixture/blob/master/code_nondeformed_inputs_nofix/test.ipynb). The pre-trained model is saved in [folder](https://github.com/wyn430/SmartFixture/tree/master/code_nondeformed_inputs_nofix/PPO_preTrained/Ansys_assembly/nondeformed_inputs_nofix_metal), which will be used in the validation by default.

The visualization of the validation results is included in the [Jupyter Notebook] (https://github.com/wyn430/SmartFixture/blob/master/code_nondeformed_inputs_nofix/plot.ipynb)


### Training of deformed sheet part

The Reinforcement Learning agent is trained and tested on the computational server. The training and testing scripts for the fixture layout optimization of the deformed sheet part are included in [folder](https://github.com/wyn430/SmartFixture/tree/master/code_deformed_inputs_nofix).

```
python train.py [existence of the pre-trained model] [directory of pre-trained model] [pre-trained checkpoint] [IP of ANSYS Server]

e.g., continue the training from an existing model: python train.py 1 deformed_inputs_nofix_metal deformed_inputs_nofix_metal_pretrained.pth xx.3.127.xxx
      restart the training from scratch: python train.py 0 None None xx.3.127.xxx
```

### Validation and visualization of the training results on non-deformed sheet part
The validation of the trained agent is included in the [Jupyter Notebook](https://github.com/wyn430/SmartFixture/blob/master/code_deformed_inputs_nofix/test.ipynb). The pre-trained model is saved in [folder](https://github.com/wyn430/SmartFixture/tree/master/code_deformed_inputs_nofix/PPO_preTrained/Ansys_assembly/deformed_inputs_nofix_metal), which will be used in the validation by default.


The visualization of the validation results is included in the [Jupyter Notebook](https://github.com/wyn430/SmartFixture/blob/master/code_deformed_inputs_nofix/plot.ipynb)

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/wyn430/SmartFixture/blob/master/LICENSE) file for details.


