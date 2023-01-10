# flybys

A Python module to compute magnetopause and bowshock crossing events during BepiColombo flybys to Venus and Mercury

## Installation
To install this project's package run:

pip install flybys

To install the package in editable mode, use:

pip install –editable flybys

## Usage
```
from flybys.venus import * 
from flybys.mercury import *

metakernel = '/path/to/spice/dataset/mk/bc_plan.tm'
entry, exit = venus_bowshock_crossings(metakernel, 
                                       "2021-08-09T14:00:00",
                                       "2021-08-11T14:00:00",
                                       model="tricicle",
                                       aberration=True)
mercury_closest_approach(metakernel,
                         "2021-10-01T00:00:00",
                         "2021-10-02T23:59:00")                                       
                                       

```