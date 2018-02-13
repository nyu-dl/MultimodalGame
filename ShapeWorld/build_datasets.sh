#!/bin/bash

python generate.py -d /scratch/lhg256/comms/oneshape_textselect -f "(1,5,1,1)" -n 'oneshape_textselect' -i 100 -M -H
python generate.py -d /scratch/lhg256/comms/oneshape_simple_textselect -f "(1,5,1,1)" -n 'oneshape_simple_textselect' -i 100 -M -H
python generate.py -d /scratch/lhg256/comms/multishape_simple_textselect -f "(1,5,1,1)" -n 'multishape_simple_textselect' -i 100 -M -H
python generate.py -d /scratch/lhg256/comms/multishape_textselect -f "(1,5,1,1)" -n 'multishape_textselect' -i 100 -M -H
