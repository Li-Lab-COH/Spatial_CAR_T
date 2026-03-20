#!/usr/bin/env bash

job=$1
del=$2
test=$3
if [[ "$job" == 'push' ]]; then
    if [[ "$del" == 'delete' ]]; then
        if [[ "$test" == "test" ]]; then
            rsync -azvn --itemize-changes --delete --partial --progress \
                /Users/janzules/Roselab/Spatial/Second_sequencing_run_2025/second_run_images \
                janzules@apollo-acc.coh.org:/home/janzules/spatial/
        else 
            rsync -azv --delete --partial --progress \
                /Users/janzules/Roselab/Spatial/Second_sequencing_run_2025/second_run_images \
                janzules@apollo-acc.coh.org:/home/janzules/spatial/
        fi
    else 
        rsync -azv --partial --progress \
            /Users/janzules/Roselab/Spatial/Second_sequencing_run_2025/second_run_images \
            janzules@apollo-acc.coh.org:/home/janzules/spatial/
    fi
elif [[ "$job" == 'pull' ]]; then
    if [[ "$del" == 'delete' ]]; then
        if [[ "$test" == "test" ]]; then
            rsync -azvn --itemize-changes --delete --partial --progress \
                janzules@apollo-acc.coh.org:/home/janzules/spatial/second_run_images \
                /Users/janzules/Roselab/Spatial/Second_sequencing_run_2025/
        else
            rsync -azv --delete --partial --progress \
                janzules@apollo-acc.coh.org:/home/janzules/spatial/second_run_images \
                /Users/janzules/Roselab/Spatial/Second_sequencing_run_2025/

        fi
    else 
        rsync -azv --partial --progress \
            janzules@apollo-acc.coh.org:/home/janzules/spatial/second_run_images \
            /Users/janzules/Roselab/Spatial/Second_sequencing_run_2025/
    fi
else
    echo "Usage: $0 {push|pull}" >&2
    exit 1
fi