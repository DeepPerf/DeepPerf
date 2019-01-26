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

DeepPerf has been evaluated on 11 real-world configurable software systems: six of these systems have only binary configuration options, the
other five systems have both binary and numeric configuration options. The data is store in .csv files. These software systems were measured and published
online by the SPLConqueror team. More information of these systems and how they were measured can be found in [here](http://www.fosd.de/SPLConqueror/).

## Usage
