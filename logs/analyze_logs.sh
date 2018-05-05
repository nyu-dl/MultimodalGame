#!/bin/bash
num=$[$2+1]
file="$1.log"
root_folder="$1_analysis"
if [ ! -d "$root_folder" ]; then
  mkdir $root_folder
fi

echo "Analyzing $file"
best_dev="$root_folder/bestdev.txt"
cat $file | grep "best" | sed 's/.*: //' > $best_dev
echo "Best dev results written to $best_dev"

if [ ! -d "$root_folder/self_com" ]; then
  mkdir $root_folder/self_com
fi
anum=1
while [ $anum -lt $num ]
do
  cat $file | grep "Agent $anum self communication: In Domain Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' > $root_folder/self_com/agent_$anum.txt
  anum=$[$anum+1]
done
echo "Individual agent self communication results written to $root_folder/self_com"

echo "Creating combined self communication csv..."
join -t , $root_folder/self_com/agent_1.txt $root_folder/self_com/agent_2.txt > $root_folder/self_com.csv
anum=3
while [ $anum -lt $num ]
do
  join -t , $root_folder/self_com.csv $root_folder/self_com/agent_$anum.txt > $root_folder/tmp && mv $root_folder/tmp $root_folder/self_com.csv
  anum=$[$anum+1]
done
echo "Combined self communication csv written to $root_folder/self_com.csv"

echo "Creating pairwise stats..."
if [ ! -d "$root_folder/pairwise_BR_AC" ]; then
  mkdir $root_folder/pairwise_BR_AC
fi
if [ ! -d "$root_folder/pairwise_BR_BC" ]; then
  mkdir $root_folder/pairwise_BR_BC
fi
if [ ! -d "$root_folder/pairwise_1R_AC" ]; then
  mkdir $root_folder/pairwise_1R_AC
fi
if [ ! -d "$root_folder/pairwise_1R_BC" ]; then
  mkdir $root_folder/pairwise_1R_BC
fi
anum=1
bnum=2
while [ $anum -lt $[$num-1] ]
do
  cat $file | grep "In Domain: Agents $anum,$bnum Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' > $root_folder/pairwise_BR_AC/$anum.txt
  cat $file | grep "In Domain: Agents $anum,$bnum Development Accuracy, both right, no comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' > $root_folder/pairwise_BR_BC/$anum.txt
  cat $file | grep "In Domain: Agents $anum,$bnum Development Accuracy, at least 1 right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' > $root_folder/pairwise_1R_AC/$anum.txt
  cat $file | grep "In Domain: Agents $anum,$bnum Development Accuracy, at least right, no comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' > $root_folder/pairwise_1R_BC/$anum.txt
  anum=$[$anum+1]
  bnum=$[$bnum+1]
done

join -t , $root_folder/pairwise_BR_AC/1.txt $root_folder/pairwise_BR_AC/2.txt > $root_folder/pairwise_BR_AC.csv
join -t , $root_folder/pairwise_BR_BC/1.txt $root_folder/pairwise_BR_BC/2.txt > $root_folder/pairwise_BR_BC.csv
join -t , $root_folder/pairwise_1R_AC/1.txt $root_folder/pairwise_1R_AC/2.txt > $root_folder/pairwise_1R_AC.csv
join -t , $root_folder/pairwise_1R_BC/1.txt $root_folder/pairwise_1R_BC/2.txt > $root_folder/pairwise_1R_BC.csv
anum=3
while [ $anum -lt $[$num-1] ]
do
  join -t , $root_folder/pairwise_BR_AC.csv $root_folder/pairwise_BR_AC/$anum.txt > $root_folder/tmp && mv $root_folder/tmp $root_folder/pairwise_BR_AC.csv
  join -t , $root_folder/pairwise_BR_BC.csv $root_folder/pairwise_BR_BC/$anum.txt > $root_folder/tmp && mv $root_folder/tmp $root_folder/pairwise_BR_BC.csv
  join -t , $root_folder/pairwise_1R_AC.csv $root_folder/pairwise_1R_AC/$anum.txt > $root_folder/tmp && mv $root_folder/tmp $root_folder/pairwise_1R_AC.csv
  join -t , $root_folder/pairwise_1R_BC.csv $root_folder/pairwise_1R_BC/$anum.txt > $root_folder/tmp && mv $root_folder/tmp $root_folder/pairwise_1R_BC.csv
  anum=$[$anum+1]
done
echo "Pairwise stats calculated"
