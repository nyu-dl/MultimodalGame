#!/bin/bash
file="$1.log"
output="$1_analysis.csv"
echo "Analyzing $file"

# Remove file if it exists
if [ -f $output ]; then
  rm $output
fi

self_com_1pplus_1="0,0"
self_com_1pplus_2="5,5"
self_com_1pplus_3="14,14"
self_com_1pplus_4="18,18"
self_com_1pplus_5="19,19"
self_com_1p_1="na"
self_com_1p_2="na"
self_com_1p_3="na"
self_com_1p_4="na"
self_com_1p_5="na"
pool_com_tt_1="0,1"
pool_com_tt_2="3,4"
pool_com_tt_3="10,7"
pool_com_tt_4="18,16"
pool_com_tt_5="21,20"
pool_com_ntt_1="na"
pool_com_ntt_2="na"
pool_com_ntt_3="na"
pool_com_ntt_4="na"
pool_com_ntt_5="na"
xpool_com_tt_1="1,18"
xpool_com_tt_2="4,18"
xpool_com_tt_3="14,19"
xpool_com_tt_4="17,4"
xpool_com_tt_5="20,15"
xpool_com_ntt_1="1,13"
xpool_com_ntt_2="4,8"
xpool_com_ntt_3="15,0"
xpool_com_ntt_4="18,0"
xpool_com_ntt_5="20,13"

declare -a com_types=($self_com_1pplus_1 $self_com_1pplus_2 $self_com_1pplus_3 $self_com_1pplus_4 $self_com_1pplus_5 $self_com_1p_1 $self_com_1p_2 $self_com_1p_3 $self_com_1p_4 $self_com_1p_5 $pool_com_tt_1 $pool_com_tt_2 $pool_com_tt_3 $pool_com_tt_4 $pool_com_tt_5 $pool_com_ntt_1 $pool_com_ntt_2 $pool_com_ntt_3 $pool_com_ntt_4 $pool_com_ntt_5 $xpool_com_tt_1 $xpool_com_tt_2 $xpool_com_tt_3 $xpool_com_tt_4 $xpool_com_tt_5 $xpool_com_ntt_1 $xpool_com_ntt_2 $xpool_com_ntt_3 $xpool_com_ntt_4 $xpool_com_ntt_5)

echo $com_types

# Dummy list for missing fields, and first field to make iteration simpler
cat $file | grep "In Domain, Pool 1" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,na/' > temp_combined.txt

for i in "${com_types[@]}"
do
  # Check for non zero
  # echo $i
  if [[ $i != "na" ]]; then
    a1="$(echo $i | cut -d',' -f1)"
    a2="$(echo $i | cut -d',' -f2)"
    # echo $a1 $a2
    a1=$[$a1+1]
    a2=$[$a2+1]
    echo "Processing results for agents $a1 and $a2"
    cat $file | grep "In Domain Dev: Agent $a1 | Agent $a2, ids" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' > temp.txt
  else
    echo "No results in this category"
    cat $file | grep "In Domain, Pool 1" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,na/' > temp.txt
  fi
    join -t , temp_combined.txt temp.txt > tmp && mv tmp temp_combined.txt
done

cat $file | grep "In Domain, Pool 1" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' > temp.txt
join -t , temp_combined.txt temp.txt > tmp && mv tmp temp_combined.txt
cat $file | grep "In Domain, Pool 2" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' > temp.txt
join -t , temp_combined.txt temp.txt > tmp && mv tmp temp_combined.txt
cat $file | grep "In Domain, Pool 3" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' > temp.txt
join -t , temp_combined.txt temp.txt > tmp && mv tmp temp_combined.txt
cat $file | grep "In Domain, Pool 4" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' > temp.txt
join -t , temp_combined.txt temp.txt > tmp && mv tmp temp_combined.txt
cat $file | grep "In Domain, Pool 5" | grep "Development Accuracy, both right, after comms:" | sed 's/.*Step: \([0-9]*\).*comms: \(.*\)/\1,\2/' > temp.txt
join -t , temp_combined.txt temp.txt > tmp && mv tmp temp_combined.txt

# Build output file
echo "step,dummy,self_com_1p+_1,self_com_1p+_2,self_com_1p+_3,self_com_1p+_4,self_com_1p+_5,self_com_1p_1,self_com_1p_2,self_com_1p_3,self_com_1p_4,self_com_1p_5,pool_com_tt_1,pool_com_tt_2,pool_com_tt_3,pool_com_tt_4,pool_com_tt_5,pool_com_ntt_1,pool_com_ntt_2,pool_com_ntt_3,pool_com_ntt_4,pool_com_ntt_5,xpool_com_tt_1,xpool_com_tt_2,xpool_com_tt_3,xpool_com_tt_4,xpool_com_tt_5,xpool_com_ntt_1,xpool_com_ntt_2,xpool_com_ntt_3,xpool_com_ntt_4,xpool_com_ntt_5,frozen1,frozen2,frozen3,frozen4,frozen5" >> $output

cat temp_combined.txt >> $output
cat $output

# Cleanup temp files
rm temp_combined.txt
rm temp.txt
