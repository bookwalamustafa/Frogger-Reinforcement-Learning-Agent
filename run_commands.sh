#!/bin/bash

while IFS= read -r command || [ -n "$command" ]; do
    if [ -z "$command" ] || [[ "$command" == \#* ]]; then
        continue
    fi
    
    echo "Running: $command"
    eval "$command"
done < "commands.txt"