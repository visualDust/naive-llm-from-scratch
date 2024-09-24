#!/bin/bash

# Enable nullglob so the loop doesn't execute if no files match
shopt -s nullglob

# Loop over all tar files in the current directory
for file in *.tar; do
    if [ -f "$file" ]; then
        echo "Extracting $file..."
        tar -xf "$file"
    fi
done
