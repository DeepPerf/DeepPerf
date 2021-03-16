# DeepPerf

Many software systems provide users with a set of configuration options and different configurations may lead to different runtime performance of the system. It is necessary to understand the performance of a system under a certain configuration, before the system is actually configured and deployed. This helps users make rational decisions in configurations and reduce performance testing cost. As the combination of configurations could be exponential, it is difficult to exhaustively deploy and measure system performance under all possible configurations. Recently, several learning methods have been proposed to build a performance prediction model based on performance data collected from a small sample of configurations, and then use the model to predict system performance with a new configuration. DeepPerf is an end-to-end deep learning based solution that can train a software performance prediction model from a limited number of samples and predict the performance value of software system under a new configuration. DeepPerf consists of two main stages:
- Stage 1: Tune the hyperparameters of the neural network
- Stage 2: Utilize the hyperparameters in Stage 1 to train the neural network with the samples and predict the performance value of software system under a new configuration.

## Citing DeepPerf

If you find our code useful, please cite our paper:
```
@inproceedings{Ha2019DeepPerf,
  author    = {Huong Ha and
               Hongyu Zhang},
  title     = {DeepPerf: performance prediction for configurable software with deep
               sparse neural network},
  booktitle = {Proceedings of the 41st International Conference on Software Engineering,
               {ICSE} 2019, Montreal, QC, Canada, May 25-31, 2019},
  pages     = {1095--1106},
  publisher = {{IEEE} / {ACM}},
  year      = {2019}
}
```

## Prerequisites

- Python 3.6.x
- Tensorflow (tested with tensorflow 1.10.0, 1.8.0)

## Installation

DeepPerf can be directly executed through source code

1. Download and install Python 3.6.x [here](https://www.python.org/downloads/).

2. Install Tensorflow

    ```$ pip install tensorflow==1.10.0```

3. Clone DeepPerf

    ``` $ clone https://github.com/DeepPerf/DeepPerf.git```


## Data

DeepPerf has been evaluated on 11 real-world configurable software systems: 
- Apache
- LLVM
- x264
- BDBC
- BDBJ
- SQL
- Dune
- hipacc
- hsmgp
- javagc
- sac

Six of these systems have only binary configuration options, the other five systems have both binary and numeric configuration options. The data is store in the DeepPerf\Data directory. These software systems were measured and published online by the SPLConqueror team. More information of these systems and how they were measured can be found in [here](http://www.fosd.de/SPLConqueror/).

## Usage

To run DeepPerf, users need to specify the name of the software system they wish to evaluate and then run the script `AutoDeepPerf.py`. There are 11 software systems that users can evaluate: Apache, LLVM, x264, BDBC, BDBJ, SQL, Dune, hipacc, hsmgp, javagc, sac. The script will then evaluate DeepPerf on the chosen software system with the same experiment setup presented in our paper. Specifically, for binary software systems, DeepPerf will run with five different sample sizes: n, 2n, 3n, 4n, 5n with n being the number of options, and 30 experiments for each sample size. For binary-numeric software systems, DeepPerf will run with the sample sizes specified in Table IV of our paper, and 30 experiments for each sample size. For example, if users want to evaluate DeepPerf with the system LLVM, the command line to run DeepPerf will be: 

```$ python AutoDeepPerf.py LLVM```

When finishing each sample size, the script will output a .csv file that shows the mean prediction error and the margin (95% confidence interval) of that sample size over the 30 experiments. These results will be same/similar as the results we report in Table III and IV of our paper.

Alternatively, users can customize the sample size and/or the number of experiments for each sample size by using the optional arguments ```-ss``` and ```-ne```. For example, to set the sample size = 20 and the number of experiments = 10, the corresponding command line is:

```$ python AutoDeepPerf.py LLVM -ss 20 -ne 10```

Setting none or one option will result in the other option(s) running with the default setting. The default setting of the number of experiments is 30. The default setting of the sample size is: (a) the five different sample sizes: n, 2n, 3n, 4n, 5n, with n being the number of configuration options, when the evaluated system is a binary system OR (b) the four sample sizes specified in Table IV of our paper when the evaluated system is a binary-numeric system.

**NOTE**: The time cost of tuning hyperparameters and training the final neural network for each experiment ranges from 2-20 minutes depends on the software system, the sample size and the user's CPU. Typically, the time cost will be smaller when the software systems has smaller number of configurations or when the sample size is small. Therefore, please be aware that for each sample size, the time cost of evaluating 30 experiments ranges from 1 hour to 10 hours. 

## Experimental Results

To evaluate the prediction accuracy, we use the mean relative error (MRE), which is computed as,

<a href="https://www.codecogs.com/eqnedit.php?latex=MRE&space;=&space;\dfrac{1}{\vert&space;C&space;\vert}&space;\sum_{c&space;\in&space;V}&space;\dfrac{\vert&space;predicted_c&space;-&space;actual_c&space;\vert}{actual_c}&space;\times&space;100," target="_blank"><img src="https://latex.codecogs.com/gif.latex?MRE&space;=&space;\dfrac{1}{\vert&space;C&space;\vert}&space;\sum_{c&space;\in&space;V}&space;\dfrac{\vert&space;predicted_c&space;-&space;actual_c&space;\vert}{actual_c}&space;\times&space;100," title="MRE = \dfrac{1}{\vert C \vert} \sum_{c \in V} \dfrac{\vert predicted_c - actual_c \vert}{actual_c} \times 100," /></a>

where V is the testing dataset, predicted_c is the predicted performance value of configuration c generated using the model, actual_c is the actual performance value of configuration c. In the two tables below, Mean is the mean of the MREs seen in 30 experiments and Margin is the margin of the 95% confidence interval of the MREs in the 30 experiments. The results are obtained when evaluating DeepPerf on a Windows 7 computer with Intel Xeon CPU E5-1650 3.2GHz 16GB RAM.

### Prediction accuracy for software systems with binary options

<table>
    <thead>
        <tr>
            <th rowspan="2" >Subject System</th>
            <th rowspan="2" >Sample Size</th>
            <th colspan="2" >DECART</th>
            <th colspan="2" >DeepPerf</th>
        </tr>
        <tr>
            <th scope="col">Mean</th>
            <th scope="col">Margin</th>
            <th scope="col">Mean</th>
            <th scope="col">Margin</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=5>Apache</td>
            <td>n</td>
            <td>NA</td>
            <td>NA</td>
            <td>17.87</td>
            <td>1.85</td>
        </tr>
        <tr>
            <td>2n</td>
            <td>15.83</td>
            <td>2.89</td>
            <td> <b> 10.24 </b> </td>
            <td>1.15</td>
        </tr>
        <tr>
            <td>3n</td>
            <td>11.03</td>
            <td>1.46</td>
            <td> <b>8.25</b> </td>
            <td>0.75</td>
        </tr>
        <tr>
            <td>4n</td>
            <td>9.49</td>
            <td>1.00</td>
            <td> <b>6.97</b> </td>
            <td>0.39</td>
        </tr>
        <tr>
            <td>5n</td>
            <td>7.84</td>
            <td>0.28</td>
            <td> <b>6.29</b> </td>
            <td>0.44</td>
        </tr>
        <tr>
            <td rowspan=5>x264</td>
            <td>n</td>
            <td>17.71</td>
            <td>3.87</td>
            <td> <b>10.43</b> </td>
            <td>2.28</td>
        </tr>
        <tr>
            <td>2n</td>
            <td>9.31</td>
            <td>1.30</td>
            <td> <b>3.61</b> </td>
            <td>0.54</td>
        </tr>
        <tr>
            <td>3n</td>
            <td>6.37</td>
            <td>0.83</td>
            <td> <b>2.13</b> </td>
            <td> 0.31</td>
        </tr>
        <tr>
            <td>4n</td>
            <td>4.26</td>
            <td>0.47</td>
            <td> <b>1.49</b> </td>
            <td>0.38</td>
        </tr>
        <tr>
            <td>5n</td>
            <td>2.94</td>
            <td>0.52</td>
            <td> <b>0.87</b> </td>
            <td>0.11</td>
        </tr>
        <tr>
            <td rowspan=5>BDBJ</td>
            <td>n</td>
            <td>10.04</td>
            <td>4.67</td>
            <td> <b>7.25</b> </td>
            <td>4.21</td>
        </tr>
        <tr>
            <td>2n</td>
            <td>2.23</td>
            <td>0.16</td>
            <td> <b>2.07</b> </td>
            <td>0.32</td>
        </tr>
        <tr>
            <td>3n</td>
            <td>2.03</td>
            <td>0.16</td>
            <td> <b>1.73</b> </td>
            <td>0.12</td>
        </tr>
        <tr>
            <td>4n</td>
            <td>1.72</td>
            <td>0.09</td>
            <td> <b>1.67</b> </td>
            <td>0.12</td>
        </tr>
        <tr>
            <td>5n</td>
            <td>1.67</td>
            <td>0.09</td>
            <td> <b>1.61</b> </td>
            <td>0.09</td>
        </tr>
        <tr>
            <td rowspan=5>LLVM</td>
            <td>n</td>
            <td>6.00</td>
            <td>0.34</td>
            <td> <b>5.09</b> </td>
            <td>0.80</td>
        </tr>
        <tr>
            <td>2n</td>
            <td>4.66</td>
            <td>0.47</td>
            <td> <b>3.87</b> </td>
            <td>0.48</td>
        </tr>
        <tr>
            <td>3n</td>
            <td>3.96</td>
            <td>0.39</td>
            <td> <b>2.54</b> </td>
            <td>0.15</td>
        </tr>
        <tr>
            <td>4n</td>
            <td>3.54</td>
            <td>0.42</td>
            <td> <b>2.27</b> </td>
            <td>0.16</td>
        </tr>
        <tr>
            <td>5n</td>
            <td>2.84</td>
            <td>0.33</td>
            <td> <b>1.99</b> </td>
            <td>0.15</td>
        </tr>
        <tr>
            <td rowspan=5>BDBC</td>
            <td>n</td>
            <td>151.0</td>
            <td>90.70</td>
            <td> <b>133.6</b> </td>
            <td>54.33</td>
        </tr>
        <tr>
            <td>2n</td>
            <td>43.8</td>
            <td>26.72</td>
            <td> <b>16.77</b> </td>
            <td>2.25</td>
        </tr>
        <tr>
            <td>3n</td>
            <td>31.9</td>
            <td>22.73</td>
            <td> <b>13.1</b> </td>
            <td>3.39</td>
        </tr>
        <tr>
            <td>4n</td>
            <td> <b>6.93</b> </td>
            <td>1.39</td>
            <td>6.95</td>
            <td>1.11</td>
        </tr>
        <tr>
            <td>5n</td>
            <td> <b>5.02</b> </td>
            <td>1.69</td>
            <td>5.82</td>
            <td>1.33</td>
        </tr>
        <tr>
            <td rowspan=5>SQL</td>
            <td>n</td>
            <td> <b>4.87</b> </td>
            <td>0.22</td>
            <td>5.04</td>
            <td>0.32</td>
        </tr>
        <tr>
            <td>2n</td>
            <td>4.67</td>
            <td>0.17</td>
            <td> <b>4.63</b> </td>
            <td>0.13</td>
        </tr>
        <tr>
            <td>3n</td>
            <td> <b>4.36</b> </td>
            <td>0.09</td>
            <td>4.48</td>
            <td>0.08</td>
        </tr>
        <tr>
            <td>4n</td>
            <td> <b>4.21</b> </td>
            <td>0.1</td>
            <td>4.40</td>
            <td>0.14</td>
        </tr>
        <tr>
            <td>5n</td>
            <td> <b>4.11</b> </td>
            <td>0.08</td>
            <td>4.27</td>
            <td>0.13</td>
        </tr>
    </tbody>
</table>
   
### Prediction accuracy for software systems with binary-numeric options

<table>
    <thead>
        <tr>
            <th rowspan="2" >Subject System</th>
            <th rowspan="2" >Sample Size</th>
            <th colspan="2" >SPLConqueror</th>
            <th colspan="3" >DeepPerf</th>
        </tr>
        <tr>
            <th scope="col">Sampling Heuristic</th>
            <th scope="col">Mean</th>
            <th scope="col">Sampling Heuristic</th>
            <th scope="col">Mean</th>
            <th scope="col">Margin</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>Dune</td>
            <td>49</td>
            <td>OW RD</td>
            <td>20.1</td>
            <td>RD</td>
            <td> <b>15.73</b> </td>
            <td>0.90</td>
        </tr>
        <tr>
            <td>78</td>
            <td>PW RD</td>
            <td>22.1</td>
            <td>RD</td>
            <td> <b>13.67</b> </td>
            <td>0.82</td>
        </tr>
        <tr>
            <td>240</td>
            <td>OW PBD(49, 7)</td>
            <td>10.6</td>
            <td>RD</td>
            <td> <b>8.19</b> </td>
            <td>0.34</td>
        </tr>
        <tr>
            <td>375</td>
            <td>OW PBD(125, 5)</td>
            <td>18.8</td>
            <td>RD</td>
            <td> <b>7.20</b> </td>
            <td>0.17</td>
        </tr>
        <tr>
            <td rowspan=4>hipacc</td>
            <td>261</td>
            <td>OW RD</td>
            <td>14.2</td>
            <td>RD</td>
            <td> <b>9.39</b> </td>
            <td>0.37</td>
        </tr>
        <tr>
            <td>528</td>
            <td>OW PBD(125, 5)</td>
            <td>13.8</td>
            <td>RD</td>
            <td> <b>6.38</b> </td>
            <td>0.44</td>
        </tr>
        <tr>
            <td>736</td>
            <td>OW PBD(49, 7)</td>
            <td>13.9</td>
            <td>RD</td>
            <td> <b>5.06</b> </td>
            <td>0.35</td>
        </tr>
        <tr>
            <td>1281</td>
            <td>PW RD</td>
            <td>13.9</td>
            <td>RD</td>
            <td> <b>3.75</b> </td>
            <td>0.26</td>
        </tr>
        <tr>
            <td rowspan=4>hsmgp</td>
            <td>77</td>
            <td>OW RD</td>
            <td> <b>4.5</b> </td>
            <td>RD</td>
            <td>6.76</td>
            <td>0.87</td>
        </tr>
        <tr>
            <td>173</td>
            <td>PW RD</td>
            <td> <b>2.8</b> </td>
            <td>RD</td>
            <td>3.60</td>
            <td>0.2</td>
        </tr>
        <tr>
            <td>384</td>
            <td>OW PBD(49, 7)</td>
            <td> <b>2.2</b> </td>
            <td>RD</td>
            <td>2.53</td>
            <td>0.13</td>
        </tr>
        <tr>
            <td>480</td>
            <td>OW PBD(125, 5)</td>
            <td> <b>1.7</b> </td>
            <td>RD</td>
            <td>2.24</td>
            <td>0.11</td>
        </tr>
        <tr>
            <td rowspan=4>javagc</td>
            <td>423</td>
            <td>OW PBD(49, 7)</td>
            <td>37.4</td>
            <td>RD</td>
            <td> <b>24.76</b> </td>
            <td>2.42</td>
        </tr>
        <tr>
            <td>534</td>
            <td>OW RD</td>
            <td>31.3</td>
            <td>RD</td>
            <td> <b>23.27</b> </td>
            <td>4.00</td>
        </tr>
        <tr>
            <td>855</td>
            <td>OW PBD(125, 5)</td>
            <td>21.9</td>
            <td>RD</td>
            <td> <b>21.83</b> </td>
            <td>7.07</td>
        </tr>
        <tr>
            <td>2571</td>
            <td>OW PBD(49, 7)</td>
            <td>28.2</td>
            <td>RD</td>
            <td> <b>17.32</b> </td>
            <td>7.89</td>
        </tr>
        <tr>
            <td rowspan=4>sac</td>
            <td>2060</td>
            <td>OW RD</td>
            <td>21.1</td>
            <td>RD</td>
            <td> <b>15.83</b> </td>
            <td>1.25</td>
        </tr>
        <tr>
            <td>2295</td>
            <td>OW PBD(125, 5)</td>
            <td>20.3</td>
            <td>RD</td>
            <td> <b>17.95</b> </td>
            <td>5.63</td>
        </tr>
        <tr>
            <td>2499</td>
            <td>OW PBD(49, 7)</td>
            <td> <b>16</b> </td>
            <td>RD</td>
            <td>17.13</td>
            <td>2.22</td>
        </tr>
        <tr>
            <td>3261</td>
            <td>PW RD</td>
            <td>30.7</td>
            <td>RD</td>
            <td> <b>15.40</b> </td>
            <td>2.05</td>
        </tr>
    </tbody>
</table>
    



