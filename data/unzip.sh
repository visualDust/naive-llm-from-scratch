#!/bin/bash

# Enable nullglob so the loop doesn't execute if no files match
shopt -s nullglob

cd openwebtext/subsets
# Loop over all tar files in the current directory
for file in *.tar; do
    if [ -f "$file" ]; then
        echo "Extracting $file..."
        tar -xf "$file"
    fi
done

# move all files in openwebtext folder into .
mv openwebtext/* .

# remove all the .tar files
rm *.tar

# remove empty subsets folder inside
rm -r openwebtext