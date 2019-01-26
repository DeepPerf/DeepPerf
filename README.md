# DeepPerf

DeepPerf is an end-to-end deep learning based solution that can train a software performance prediction model from a limited number of samples and predict the performance value of a new configuration. 

## Prerequisites

- Python 3.6.x
- Tensorflow (no more than tensorflow 1.10.0)

## Installation

DeepPerf can be directly executed through source code

1. Download and install Python 3.6.x [here](https://www.python.org/downloads/).

2. Install Tensorflow

    ```$ pip3 install tensorflow==1.10.0```

3. Clone DeepPerf

    ``` $ clone https://github.com/DeepPerf/DeepPerf.git```


## Data

DeepPerf has been evaluated on 11 real-world configurable software systems: six of these systems have only binary configuration options, the other five systems have both binary and numeric configuration options. The data is store in .csv files in the DeepPerf\Data directory. These software systems were measured and published online by the SPLConqueror team. More information of these systems and how they were measured can be found in [here](http://www.fosd.de/SPLConqueror/).

## Usage

To run DeepPerf, users need to specify the name of the software system they wish to evaluate and then run the script AutoDeepPerf.py. The script will then evaluate DeepPerf on the chosen software system with 5 different sample sizes and 30 experiments for each sample size. For example, if users want to evaluate the system LLVM, the command line to run DeepPerf will be:

    ```$ python AutoDeepPerf.py LLVM```

When finishing each sample size, the script will output a .csv file that shows the mean prediction error and the margin (95% confidence interval) of that sample size over the 30 experiments. There are 11 software systems that users can evaluate: Apache, LLVM, x264, BDBC, BDBJ, SQL, Dune, hipacc, hsmgp, javagc, sac. 

Alternatively, users can customize the sample size and the number of experiments for each sample size by using the optional arguments ```-ss``` and ```-ne```. For example, to set the sample size to be 20 and the number of experiments to be 10, the corresponding command lines is:

    ```$ python AutoDeepPerf.py LLVM -ss 20 --ne 10```

   
    


    



