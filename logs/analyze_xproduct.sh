#!/bin/bash
file="$1.log"
output="$1_xproduct.csv"
num=$[$2+1]
echo "Analyzing $file"

cat $file | grep "In Domain Agents 1,[0-9][0-9]* Development Accuracy, both right, after comms:" | sed 's/.*Agents [0-9][0-9]*,//g' | sed 's/\([0-9][0-9]*\).*comms: /\1,/g' > $output

anum=2
while [ $anum -lt $num ]
do
  # cat $file | grep "In Domain Agents $anum,[0-9][0-9]* Development Accuracy, both right, after comms:"
  cat $file | grep "In Domain Agents $anum,[0-9][0-9]* Development Accuracy, both right, after comms:" | sed 's/.*Agents [0-9][0-9]*,//g' | sed 's/\([0-9][0-9]*\).*comms: /\1,/g' > temp.txt
  join -t , $output temp.txt > tmp && mv tmp $output
  anum=$[$anum+1]
done

cat $output
rm temp.txt
