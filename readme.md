
# HSPARC: A Hierarchical Sparse Communication Framework for Scalable Coordination in Multi-Agent Systems


## Overview

Coordination in multi-agent systems under the centralized training and decentralized execution paradigm becomes increasingly challenging as agent scale grows and communication resources are limited. Global interaction mechanisms often incur excessive computational overhead, limiting scalability in large-scale settings. 

To address this issue, this paper proposes HSPARC, a hierarchical coordination framework that integrates dynamic grouping, sparse communication, and priority-based decision ordering. By decomposing global coordination into structured local interactions, HSPARC aims to achieve a better performance--efficiency trade-off under a fixed communication budget. Theoretical analysis shows that, with a bounded group size, the proposed design reduces the growth of coordination cost compared with global interaction schemes. Experiments on SMACv2 and MAgent demonstrate that HSPARC maintains competitive task performance while exhibiting better scalability and lower computational overhead under limited communication resources. These results suggest that structured, sparse interactions are an effective approach for scalable multi-agent coordination.


## Installation

### Dependences
``` Bash
pip install -r requirements.txt
```

### StarCraft II & SMAC
Run the script
``` Bash
bash install_sc2.sh
```
Or you could install them manually to other path you like, just follow here: https://github.com/oxwhirl/smac.

### MAgent
Install IMP environment
``` Bash
pip install git+https://github.com/moratodpg/imp_marl.git
```
Install MAgent environment
``` Bash
pip install magent==0.1.14
pip install pettingzoo==1.12.0
cp battle_v3_view7.py PATH_TO_YOUR_PETTINGZOO_ENV/pettingzoo/magent/
cp adversarial_pursuit_view8_v3.py PATH_TO_YOUR_PETTINGZOO_ENV/pettingzoo/magent/
```


## Quick Start

When your environment is ready, you could run shells in the "scripts" folder with algo="HSPARC". For example:

``` Bash
bash ./train_smacv2.sh  # run HSPARC on SMACv2
bash ./train_smac.sh  # run HSPARC on SMAC
bash ./train_magent.sh  # run HSPARC on magent
```
If you would like to change the configs of experiments, you could modify sh files or look for config.py for more details.




