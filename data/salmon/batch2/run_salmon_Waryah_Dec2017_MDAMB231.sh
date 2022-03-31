#!/bin/bash

# shell script to deploy salmon for the alignment and quantification of RNA-seq data
# generated 2017-08-12_2006h

# specify sample sets
declare -a fileSets=("1_MDA231_Wildtype" "2_MDA231_Wildtype" "3_MDA231_Wildtype" "1_MDA231_empty_control" "2_MDA231_empty_control" "3_MDA231_empty_control" "1_MDA231_gRNA_" "2_MDA231_gRNA_" "3_MDA231_gRNA_" "1_MDA231_gRNA4_" "2_MDA231_gRNA4_" "3_MDA231_gRNA4_" "1_SUM159_Wildtype" "2_SUM159_Wildtype" "3_SUM159_Wildtype")

# specify path aliases
DIRPATH=/wehisan/general/academic/lab_davis/Joe/deploy
FASTQDIR=/wehisan/general/academic/lab_davis/data/Blancafort/AGRF_CAGRF16331_CC0U1ANXX
DIROUT=${DIRPATH}/salmon
DIRINDEX=/wehisan/general/academic/lab_davis/genomes/salmon_index_GRCh38

# ensure the Salmon directory is on the system path
[[ ":$PATH:" != *":~/home/salmon/bin:"* ]] && PATH="~/home/salmon/bin:${PATH}"

# loop over all of the specified file sets
for set in "${fileSets[@]}"
do
# if necessary, create the root output directory
[[ -d ${DIROUT} ]] || mkdir ${DIROUT}

	# extract the fastq file names
	FASTQ_files=`ls ${FASTQDIR}/*fastq.gz | grep "R1" | grep $set`

	filesAppend=""
	for File in ${FASTQ_files}
	do
		filesAppend=$filesAppend" "$File
	done
	# if necessary, create the output directory
	[[ -d ${DIROUT}/$set ]] || mkdir ${DIROUT}/$set

	salmon quant -i ${DIRINDEX} -l A -r $filesAppend -p 8 -o ${DIROUT}/$set

done
echo Salmon alignment complete
