#!/bin/bash

# shell script to deploy salmon for the alignment and quantification of RNA-seq data
# generated 2017-01-10_1442h

# specify sample sets
declare -a fileSets=("1_SUM159" "2_SUM159" "3_SUM159" "1_pLV_empty_control" "2_pLV_empty_control" "3_pLV_empty_control" "1_pLV_Guide_4" "2_pLV_Guide_4" "3_pLV_Guide_4" "1_pLV_All_Guides" "2_pLV_All_Guides" "3_pLV_All_Guides")

# specify path aliases
DIRPATH=/wehisan/general/academic/lab_davis/Joe/deploy
FASTQDIR=/wehisan/general/academic/lab_davis/data/Blancafort/AGRF_CAGRF15987_CB2YGANXX
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
	FASTQ_files=`ls ${FASTQDIR}/*fastq.gz| grep "R1" | grep $set`

	filesAppend=""
	for File in ${FASTQ_files}
	do
		filesAppend=$filesAppend" "$File
	done
	# if necessary, create the output directory
	[[ -d ${DIROUT}/$set ]] || mkdir ${DIROUT}/$set

	salmon quant -i ${DIRINDEX} -l A -r $filesAppend -p 7 -o ${DIROUT}/$set

done
echo Salmon alignment complete
