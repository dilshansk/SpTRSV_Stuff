#!/bin/bash

for i in {1..90}
do
   echo "===Execute iteration $i==="
   rm -rf run.log
   make clean
   make gcc_compile |& tee run.log

    if grep -q "Run completed" run.log; then
        echo "Test passed in iteration $i"
    else
        echo "Test falied iteration $i"
        exit 1;
    fi
done

echo "ALL TESTS PASSED"
exit 0