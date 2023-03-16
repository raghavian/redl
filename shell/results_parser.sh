#!/bin/bash
set -x

LOC='LOG_DIR'
cd $LOC

for f in *
do
    tail -2 $f/log.txt >> $f.txt
    tail -6 $f/*_output*.log >> $f.txt
    sed -i 's/, Test Loss/\nTest Loss/g' $f.txt
    sed -i 's/, GPU/\nGPU/g' $f.txt
    sed -i 's/Acc: /Acc:/g' $f.txt
    sed -i 's/Param: /Param:/g' $f.txt
    sed -i 's/ M//g' $f.txt
    sed -i 's/ G//g' $f.txt
    sed -i 's/ kWh//g' $f.txt
    sed -i 's/ g//g' $f.txt
    sed -i 's/ ,//g' $f.txt
    sed -i 's/ km//g' $f.txt
    sed -i 's/\t//g' $f.txt
    sed -i 's/This is equivalent to://g' $f.txt
    sed -i 's/travelled by car//g' $f.txt
    sed -i '/CarbonTracker/d' $f.txt
    txt=$(tr '\n' ',' < $f.txt)
    echo $txt > $f.txt
    txt=$(awk '{sub(/,,/,",dist:")}1' $f.txt)
    echo $txt > $f.txt

done

