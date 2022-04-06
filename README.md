# 2022_Waryah_ZEB1_inhibition

A repository with computational code and outputs to accompany the manuscript Waryah *et al.* (submitted), Synthetic epigenetic reprograming of breast cancer hybrid epithelial to mesenchymal states using the CRISPR/dCas9 platform.

Analysis of the RNA-seq and DNAme data was performed by @jcursons and @MomenehForoutan. Analysis of the ChIP-seq and ATAC-seq data was performed by Dr Christian Pflueger. Please see the script folder for further details.

## Contact information

For information on the associated code please contact:
- Dr Joe Cursons (joseph.cursons (at) monash.edu)
- Dr Momeneh (Sepideh) Foroutan (momeneh.foroutan (at) monash.edu)
- Dr Christian Pflueger (christian.pflueger (at) uwa.edu.au)
- Dr Liam Fearnley (fearnley.l (at) wehi.edu.au)

For further information on the manuscript or project please contact:
- Dr Charlene Waryah (charlene.waryah (at) perkins.org.au)
- A/Prof. Pilar Blancafort (pilar.blancafort (at) uwa.edu.au)

## Associated manuscript

The scientific manuscript associated with this repository has been submitted for peer review.


## Data availability

Data generated for this project will be uploaded to GEO over the coming months.

### Public data

ENSEMBL reference for RNA-seq analysis:
- http://ftp.ensembl.org/pub/release-89/gtf/homo_sapiens/Homo_sapiens.GRCh38.89.gtf.gz  


## Project structure

### script

A folder containing scripts and functions used for data analysis and figure generation.


### preproc

A folder containing intermediate output files used in this study. 

- 20180518_dCas_pipe_out.txt: off-target gene predictions for gRNAs used in this study; determined with dsNickFury using the Azimuth and Elevation on-/off-target scoring algorithms. For further information please refer to the script <not yet uploaded> and the methods section of the associated manuscript.


#### preproc/rnaseq

A folder containing different expression results from the RNA-seq.


#### preproc/salmon

A folder containing the salmon pre-processed data for the SUM159 and MDA-MB-231 RNA-seq data (Figures NN & GSENNNN).


#### preproc/dname

A folder containing the pre-processed data for the SUM159 differential expression data (Figures NN & GSENNNNNN).


#### preproc/tcga

A folder containing the subset of TCGA-BRCA data used for this paper (Figures NN).

For further details please see:
- [Comprehensive Molecular Portraits of Invasive Lobular Breast Cancer](https://www.cell.com/cell/fulltext/S0092-8674(15)01195-2)
  - Giovanni Ciriello, Michael L. Gatza *et al.* (2015). *Cell*.
- [Comprehensive molecular portraits of human breast tumours](https://www.nature.com/articles/nature11412)
  - TCGA Network (2012). *Nature*
