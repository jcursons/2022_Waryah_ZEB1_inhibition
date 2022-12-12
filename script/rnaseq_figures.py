from adjustText import adjust_text
import copy
import csv
import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
import scipy.cluster.hierarchy as SciPyClus
import scipy.stats as scs
from singscore.singscore import *
import sys

class PathDir:

    dirCurrent = os.path.dirname(sys.argv[0])
    dirBaseGit = os.path.dirname(os.path.normpath(dirCurrent))

    pathOutFolder = dirBaseGit
    for strFolder in ['figures']:
        pathOutFolder = os.path.join(pathOutFolder, strFolder)

    pathProcRNAData = dirBaseGit
    for strFolder in ['preproc', 'rnaseq']:
        pathProcRNAData = os.path.join(pathProcRNAData, strFolder)

    pathRefData = dirBaseGit
    for strFolder in ['preproc', 'ref']:
        pathRefData = os.path.join(pathRefData, strFolder)

    pathPublicData = dirBaseGit
    for strFolder in ['preproc', 'public_data']:
        pathPublicData = os.path.join(pathPublicData, strFolder)

class Process:

    listLinesForDisp = ['MDA-MB-231',
                        'SUM159']
    listLines = [strLine.replace('-','') for strLine in listLinesForDisp]
    listDiffExprFiles = [
        f'voom-limma_{strLine}_GAll-EVC_diffExpr.csv' for strLine in listLines]

    listOfListsConds = [['NTC', 'NTC', 'NTC',
                         'EVC', 'EVC', 'EVC',
                         'g4', 'g4', 'g4',
                         'gAll', 'gAll', 'gAll'],
                        ['NTC', 'NTC', 'NTC',
                         'EVC', 'EVC', 'EVC',
                         'g4', 'g4', 'g4',
                         'gAll', 'gAll', 'gAll']]

    def quant_data(flagResult=False):

        strQuantFile = 'Waryah_ZEB1-epiCRISPR_QuantGeneLevel_lengthScaledTPM.csv'

        dfData = pd.read_table(os.path.join(PathDir.pathProcRNAData, strQuantFile),
                               sep=',', header=0, index_col=0)

        return dfData

    def diff_expr_data(flagResult=False):

        listDFToMerge = []
        for iFile in range(len(Process.listDiffExprFiles)):
            strFileName = Process.listDiffExprFiles[iFile]
            # strCond = strFileName.split('.csv')[0]
            strCellLine = strFileName.split('_GAll-EVC_diffExpr.csv')[0].split('voom-limma_')[1]
            dfIn = pd.read_csv(os.path.join(PathDir.pathProcRNAData, strFileName),
                                 sep=',', header=0, index_col=0)
            if iFile == 0:
                dfIn.drop(labels=['t', 'P.Value'],
                          axis=1,
                          inplace=True)
            else:
                dfIn.drop(labels=['AveExpr', 't', 'P.Value'],
                          axis=1,
                          inplace=True)

            arrayHasNullStats = dfIn['adj.P.Val'].isnull().astype(bool)
            arrayHasNullDiffExpr = dfIn['logFC'].isnull().astype(bool)

            arrayAdjPVals = dfIn['adj.P.Val'].values.astype(float)
            arrayLogFC = dfIn['logFC'].values.astype(float)

            arrayAdjPVals[np.where(arrayHasNullStats)[0]] = 1.0
            arrayLogFC[np.where(arrayHasNullDiffExpr)[0]] = 0.0

            dfIn['adj.P.Val'] = pd.Series(arrayAdjPVals, index=dfIn.index.tolist())
            dfIn['logFC'] = pd.Series(arrayLogFC, index=dfIn.index.tolist())

            listColumns = dfIn.columns.tolist()
            dictColToRename = {}
            for strCol in listColumns:
                if np.bitwise_or(strCol == 'external_gene_name', strCol == 'AveExpr'):
                    dictColToRename[strCol] = strCol
                else:
                    dictColToRename[strCol] = strCellLine + ':' + strCol

            dfIn.rename(columns=dictColToRename,
                        inplace=True)
            listDFToMerge.append(dfIn)

        dfMerged = pd.concat(listDFToMerge, axis=1, sort=True)

        return dfMerged

    def tcga_scores(flagResult=False,
                    dfIn=pd.DataFrame(),
                    flagPerformExtraction=False):

        strTempFileName = 'TCGA-BRCA-EpiMesScores.tsv'
        pathOut = os.path.join(PathDir.pathOutFolder, 'figure_5')

        if not os.path.exists(os.path.join(pathOut, strTempFileName)):
            flagPerformExtraction = True

        if flagPerformExtraction:

            listTCGAGenes = dfIn.index.tolist()
            listTCGASamples = dfIn.columns.tolist()
            numSamples = len(listTCGASamples)

            dictEpiMesCellLine = Process.tan2012_tissue_genes()
            listEpiTissueGenes = dictEpiMesCellLine['epi_genes']
            listMesTissueGenes = dictEpiMesCellLine['mes_genes']

            # create lists of the cell line/tissue epithelial/mesenchymal gene lists for scoring
            listOutputEpiTissueGenesMatched = [strGene for strGene in listTCGAGenes
                                               if strGene.split('|')[0] in listEpiTissueGenes]
            listOutputMesTissueGenesMatched = [strGene for strGene in listTCGAGenes
                                               if strGene.split('|')[0] in listMesTissueGenes]

            dfScoresOut = pd.DataFrame(
                {'Epithelial Score':np.zeros(numSamples, dtype=float),
                 'Mesenchymal Score':np.zeros(numSamples, dtype=float)},
                index=listTCGASamples)

            for iSample in range(numSamples):
                print('Patient ' + '{}'.format(iSample))
                strSample = listTCGASamples[iSample]
                dfScore = score(up_gene=listOutputEpiTissueGenesMatched,
                                sample=dfIn[[strSample]])
                dfScoresOut.loc[strSample,'Epithelial Score'] = \
                    dfScore['total_score'].values.astype(float)[0]

                dfScore = score(up_gene=listOutputMesTissueGenesMatched,
                                sample=dfIn[[strSample]])
                dfScoresOut.loc[strSample,'Mesenchymal Score'] = \
                    dfScore['total_score'].values.astype(float)[0]

            dfScoresOut.to_csv(os.path.join(pathOut, strTempFileName),
                               sep='\t')

        else:

            dfScoresOut = pd.read_table(os.path.join(pathOut, strTempFileName),
                                        sep='\t', index_col=0, header=0)

        return dfScoresOut

    def tcga_brca(flagResult=False,
                  flagPerformExtraction=False):

        strPanCanRNASeqFile = 'EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv'
        strTempFileName = 'TCGA_BrCa_PreProc_RNA.pickle'

        if not os.path.exists(os.path.join(PathDir.pathPublicData, strTempFileName)):
            flagPerformExtraction = True

        if flagPerformExtraction:

            # extract the TCGA pan-cancer patient metadata
            dfMeta = pd.read_excel(
                os.path.join(PathDir.pathPublicData, 'TCGA-CDR-SupplementalTableS1.xlsx'),
                header=0, index_col=0, sheet_name='TCGA-CDR')
            dfMeta.set_index('bcr_patient_barcode', inplace=True)

            # identify patients which are flagged as the breast cancer cohort
            listBRCAPatients = dfMeta[dfMeta['type']=='BRCA'].index.tolist()

            dfTCGAPanCanSamples = pd.read_table(
                os.path.join(PathDir.pathPublicData, strPanCanRNASeqFile),
                sep='\t', header=None, index_col=None, nrows=1)
            listTCGAPanCanColumns = dfTCGAPanCanSamples.iloc[0,:].tolist()
            listTCGAPanCanSamples = listTCGAPanCanColumns[1:]

            # extract primary tumour (index 01) samples from the full sample list
            listBRCASamples = [strSample for strSample in listTCGAPanCanSamples
                               if np.bitwise_and(strSample[0:len('TCGA-NN-NNNN')] in listBRCAPatients,
                                                 strSample[13:15]=='01')]

            #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
            # extract the TCGA pan-cancer RNA-seq data

            #take this subset
            dfTCGABrCa = pd.read_table(
                os.path.join(PathDir.pathPublicData, strPanCanRNASeqFile),
                sep='\t', header=0, index_col=0,
                usecols=[listTCGAPanCanColumns[0]]+listBRCASamples)
            dfTCGABrCa.to_pickle(os.path.join(PathDir.pathPublicData, strTempFileName))
        else:
            dfTCGABrCa = pd.read_pickle(os.path.join(PathDir.pathPublicData, strTempFileName))

        return dfTCGABrCa

    def ccle_brca(flagResult=False,
                  flagPerformExtraction=False):

        strTempFile = 'CCLE_BRCA_RNA_Abund.tsv'

        if not os.path.exists(os.path.join(PathDir.pathPublicData, strTempFile)):
            flagPerformExtraction = True

        if flagPerformExtraction:

            #https://ndownloader.figshare.com/files/35020903
            dfMetaData = pd.read_table(os.path.join(PathDir.pathPublicData, 'sample_info.csv'),
                                       sep=',', index_col=0, header=0)
            listBRCALinesACH = dfMetaData[dfMetaData['primary_disease'] == 'Breast Cancer'].index.tolist()
            dictACHToCCLE = dict(zip(listBRCALinesACH,
                                     dfMetaData['CCLE_Name'].reindex(listBRCALinesACH).values.tolist()))

            #https://ndownloader.figshare.com/files/34989919
            dfCCLE = pd.read_table(os.path.join(PathDir.pathPublicData, 'CCLE_expression.csv'),
                                   sep=',', index_col=0, header=0)

            dfBrCa = dfCCLE.reindex(listBRCALinesACH).copy(deep=True)
            dfBrCa.rename(
                index=dict(zip(listBRCALinesACH,[dictACHToCCLE[strLine] for strLine in listBRCALinesACH])),
                inplace=True)

            dfBrCa.to_csv(os.path.join(PathDir.pathPublicData, strTempFile),
                          sep='\t')

        else:
            dfBrCa = pd.read_table(os.path.join(PathDir.pathPublicData, strTempFile),
                                   sep='\t', index_col=0)

        return dfBrCa

    def ccle_scores(flagResult=False,
                    flagPerformExtraction=False,
                    dfIn=pd.DataFrame()):

        strTempFileName = 'CCLE-BRCA-EpiMesScores.tsv'
        pathOut = os.path.join(PathDir.pathOutFolder, 'figure_5')

        if not os.path.exists(os.path.join(pathOut, strTempFileName)):
            flagPerformExtraction = True

        if flagPerformExtraction:

            listCCLEGenes = dfIn.index.tolist()
            listCellLines = dfIn.columns.tolist()
            numSamples = len(listCellLines)

            dictEpiMesCellLine = Process.tan2012_cell_line_genes()
            listEpiCellLineGenes = dictEpiMesCellLine['epi_genes']
            listMesCellLineGenes = dictEpiMesCellLine['mes_genes']

            # create lists of the cell line/tissue epithelial/mesenchymal gene lists for scoring
            listOutputEpiCellLineGenesMatched = [strGene for strGene in listCCLEGenes
                                               if strGene.split(' (')[0] in listEpiCellLineGenes]
            listOutputMesCellLineGenesMatched = [strGene for strGene in listCCLEGenes
                                               if strGene.split(' (')[0] in listMesCellLineGenes]

            dfScoresOut = pd.DataFrame(
                {'Epithelial Score':np.zeros(numSamples, dtype=float),
                 'Mesenchymal Score':np.zeros(numSamples, dtype=float)},
                index=listCellLines)

            for iSample in range(numSamples):
                print('Cell line ' + '{}'.format(iSample))
                strSample = listCellLines[iSample]
                dfScore = score(up_gene=listOutputEpiCellLineGenesMatched,
                                sample=dfIn[[strSample]])
                dfScoresOut.loc[strSample,'Epithelial Score'] = \
                    dfScore['total_score'].values.astype(float)[0]

                dfScore = score(up_gene=listOutputMesCellLineGenesMatched,
                                sample=dfIn[[strSample]])
                dfScoresOut.loc[strSample,'Mesenchymal Score'] = \
                    dfScore['total_score'].values.astype(float)[0]

            dfScoresOut.to_csv(os.path.join(pathOut, strTempFileName),
                               sep='\t')

        else:

            dfScoresOut = pd.read_table(os.path.join(pathOut, strTempFileName),
                                        sep='\t', index_col=0, header=0)

        return dfScoresOut

    def local_scores(flagResult=False,
                    flagPerformExtraction=False,
                     dfIn=pd.DataFrame()):

        strTempFileName = 'LocalData-EpiMesScores.tsv'
        pathOut = os.path.join(PathDir.pathOutFolder, 'figure_5')

        if not os.path.exists(os.path.join(pathOut, strTempFileName)):
            flagPerformExtraction = True

        if flagPerformExtraction:

            listGenes = dfIn.index.tolist()
            listConditions = dfIn.columns.tolist()
            numSamples = len(listConditions)

            dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()
            dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(), dictENSGToHGNC.keys()))

            dictEpiMesCellLine = Process.tan2012_cell_line_genes()
            listEpiCellLineGenes = dictEpiMesCellLine['epi_genes']
            listEpiCellLineGenesENSG = [dictHGNCToENSG[strGene] for strGene in listEpiCellLineGenes]
            listMesCellLineGenes = dictEpiMesCellLine['mes_genes']
            listMesCellLineGenesENSG = [dictHGNCToENSG[strGene] for strGene in listMesCellLineGenes]

            # create lists of the cell line/tissue epithelial/mesenchymal gene lists for scoring
            listOutputEpiCellLineGenesMatched = list(set(listGenes).intersection(listEpiCellLineGenesENSG))
            listOutputMesCellLineGenesMatched = list(set(listGenes).intersection(listMesCellLineGenesENSG))

            dfScoresOut = pd.DataFrame(
                {'Epithelial Score':np.zeros(numSamples, dtype=float),
                 'Mesenchymal Score':np.zeros(numSamples, dtype=float)},
                index=listConditions)

            for iSample in range(numSamples):
                print('Cell line ' + '{}'.format(iSample))
                strSample = listConditions[iSample]
                dfScore = score(up_gene=listOutputEpiCellLineGenesMatched,
                                sample=dfIn[[strSample]])
                dfScoresOut.loc[strSample,'Epithelial Score'] = \
                    dfScore['total_score'].values.astype(float)[0]

                dfScore = score(up_gene=listOutputMesCellLineGenesMatched,
                                sample=dfIn[[strSample]])
                dfScoresOut.loc[strSample,'Mesenchymal Score'] = \
                    dfScore['total_score'].values.astype(float)[0]

            dfScoresOut.to_csv(os.path.join(pathOut, strTempFileName),
                               sep='\t')

        else:

            dfScoresOut = pd.read_table(os.path.join(pathOut, strTempFileName),
                                        sep='\t', index_col=0, header=0)

        return dfScoresOut

    def all_epi_mes_scores(flagResult=False,
                           flagPerformExtraction=False):

        dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()
        dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(), dictENSGToHGNC.keys()))

        pathOut = os.path.join(PathDir.pathOutFolder, 'figure_5')
        flagScoreTCGA = False
        if not os.path.exists(os.path.join(pathOut, 'TCGA-BRCA-EpiMesScores.tsv')):
            flagScoreTCGA = True
        flagScoreCCLE = False
        if not os.path.exists(os.path.join(pathOut, 'CCLE-BRCA-EpiMesScores.tsv')):
            flagScoreCCLE = True
        flagScoreLocalData = False
        if not os.path.exists(os.path.join(pathOut, 'LocalData-EpiMesScores.tsv')):
            flagScoreLocalData = True

        if np.any([flagScoreTCGA, flagScoreCCLE, flagScoreLocalData]):
            flagPerformExtraction=True

        if flagPerformExtraction:
            dfLocalData = Process.quant_data()
            listLocalGenesENSG = dfLocalData.index.tolist()
            setLocalGenesENSG = set(listLocalGenesENSG)

            dfTCGA = Process.tcga_brca()
            listTCGAGenes = dfTCGA.index.tolist()
            listTCGAGenesHGNC = [strGene.split('|')[0] for strGene in listTCGAGenes]
            for strGene in list(set(listTCGAGenesHGNC).difference(set(dictHGNCToENSG.keys()))):
                dictHGNCToENSG[strGene] = 'failed_map|'+strGene
            listTCGAGenesENSG = [dictHGNCToENSG[strGene] for strGene in listTCGAGenesHGNC]
            setTCGAGenesENSG = set(listTCGAGenesENSG)

            dfCCLE = Process.ccle_brca()
            listCCLEGenes = dfCCLE.columns.tolist()
            listCCLEGenesHGNC = [strGene.split(' (')[0] for strGene in listCCLEGenes]
            listCCLEGenesENSG = [dictHGNCToENSG[strGene] for strGene in listCCLEGenesHGNC]
            setCCLEGenesENSG = set(listCCLEGenesENSG)

            listCommonGenesENSG = list(setCCLEGenesENSG.intersection(setLocalGenesENSG.intersection(setTCGAGenesENSG)))
            listCommonGenesHGNC = [dictENSGToHGNC[strGene] for strGene in listCommonGenesENSG]

            listTCGAGenesOut = [strGene for strGene in listTCGAGenes if strGene.split('|')[0] in listCommonGenesHGNC]
            listCCLEGenesOut = [strGene for strGene in listCCLEGenes if strGene.split(' (')[0] in listCommonGenesHGNC]
            listLocalDataGenesOut = list(set(listLocalGenesENSG).intersection(listCommonGenesENSG))

            dfTCGAScores = Process.tcga_scores(dfIn=dfTCGA.reindex(listTCGAGenesOut))
            dfCCLEScores = Process.ccle_scores(dfIn=dfCCLE[listCCLEGenesOut].transpose())
            dfLocalScores = Process.local_scores(dfIn=dfLocalData.reindex(listLocalDataGenesOut))

        else:

            dfTCGAScores = Process.tcga_scores()
            dfCCLEScores = Process.ccle_scores()
            dfLocalScores = Process.local_scores()

        return {'TCGA':dfTCGAScores,
                'CCLE':dfCCLEScores,
                'LocalData':dfLocalScores}

    def ccle_brca_subtypes(flagResult=False):

        dfMeta = pd.read_table(os.path.join(PathDir.pathPublicData, 'sample_info.csv'),
                               sep=',', header=0, index_col=0)
        listBreastLinesACH = dfMeta[dfMeta['primary_disease']=='Breast Cancer'].index.tolist()
        listBreastLinesCCLE = dfMeta['CCLE_Name'].reindex(listBreastLinesACH).values.tolist()
        listSubtype = dfMeta['lineage_molecular_subtype'].reindex(listBreastLinesACH).values.tolist()

        for iLine in range(len(listBreastLinesACH)):
            if not listSubtype[iLine] == listSubtype[iLine]:
                listSubtype[iLine] = 'unknown'

        return dict(zip(listBreastLinesCCLE, listSubtype))

    def fig5_rnaseq_gene_lists(flagResult=False):
        # select a subset of genes from the RNA-seq DE analyses for display in Fig. 5b

        # define significance threshold for later use
        numAdjPValThresh = 0.05

        # we want to end up with a reasonable number for display
        # --> set a maximum but it may be slightly lower as we filter on prior gene annotations
        numMaxGenes = 40

        # load dictionaries for mapping between ENSG and HGNC
        dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()
        dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(), dictENSGToHGNC.keys()))

        # load the data
        dfMergedRNA = Process.diff_expr_data()
        # listDataGenes = dfMergedRNA.index.tolist()
        listDataColumns = dfMergedRNA.columns.tolist()
        listPValCols = [strCol for strCol in listDataColumns if 'adj.P.Val' in strCol]
        listCellLines = [strCol.split(':adj.P.Val')[0] for strCol in listPValCols]

        # only focus on genes that show significant DE in at least one condition
        arrayIsSigInEitherLine = np.any(
            dfMergedRNA[listPValCols].values.astype(float) < numAdjPValThresh,
            axis=1)
        listSigInEitherLine = [dfMergedRNA.index.tolist()[i] for i in np.where(arrayIsSigInEitherLine)[0]]

        # define a dataframe for gene ranks by product-of-rank
        dfRanks = pd.DataFrame(data=np.zeros((len(listSigInEitherLine), len(listCellLines)),
                                             dtype=float),
                               index=listSigInEitherLine,
                               columns=listCellLines)

        # step through the cell lines and rank genes
        for iCond in range(len(listCellLines)):
            strCellLine = listCellLines[iCond]
            # extract the logFC and adj-p value data
            arrayLogFC = np.nan_to_num(dfMergedRNA[f'{strCellLine}:logFC'].reindex(listSigInEitherLine).values.astype(float))
            arrayAdjPVal = dfMergedRNA[f'{strCellLine}:adj.P.Val'].reindex(listSigInEitherLine).values.astype(float)
            # fix any nan values to be non-significant
            arrayAdjPVal[np.isnan(arrayAdjPVal)] = 1.0
            # rank genes by the logFC*-log(p)
            listGenesRanked = [listSigInEitherLine[i] for i in
                               np.argsort(np.product((arrayLogFC, -np.log10(arrayAdjPVal)), axis=0))]
            dfRanks.loc[listGenesRanked, strCellLine] = np.arange(start=1, stop=len(listGenesRanked)+1)

        # take the product of ranks across conditions and extract the final ranks
        arrayProdRankAcrossCond = np.product(dfRanks.values.astype(float), axis=1)
        arraySortedByProdRank = np.argsort(arrayProdRankAcrossCond)
        listSortedByProdRank = [listSigInEitherLine[i] for i in arraySortedByProdRank]

        # ZEB1 inactivation tends to drive increases in gene expression (mainly epithelial genes)
        #  so weight this towards up-regulated genes
        numFromProdRank = numMaxGenes
        numUpGenes = int((2/3)*numFromProdRank)
        numDownGenes = int((1/3)*numFromProdRank)

        listOutputDownGenes = listSortedByProdRank[0:numDownGenes]
        listOutputUpGenes = listSortedByProdRank[-numUpGenes:]

        listOutputGeneOrder = listOutputDownGenes + \
                              listOutputUpGenes

        return {'HeatmapOrder':listOutputGeneOrder,
                'SigEitherLine':listSigInEitherLine}

    def guides(flagResult=False,
               strGuideFileName='hu_guides.txt'):

        dfGuides = pd.read_table(os.path.join(PathDir.pathProcResults, strGuideFileName),
                                 sep='\t', header=0, index_col=None)

        return dfGuides

    def off_targets(flagResult=False,
                    strOffTargetFileName='20180518_dCas_pipe_out.txt'):

        dfGuides = Process.guides()

        numGuides = np.shape(dfGuides)[0]
        listGuides = [dfGuides['Sequence'].iloc[i] + '_' + dfGuides['PAM'].iloc[i] for i in range(numGuides)]

        dictOffTargetGenes = dict()
        listIsGuideRow = []
        listIndicesForGuidesOfInt = []
        listFile = []
        with open(os.path.join(PathDir.pathProcResults, strOffTargetFileName)) as handFile:
            listFileContents = csv.reader(handFile)
            for listRow in listFileContents:
                strRow = listRow[0]
                listFile.append(strRow)
                if 'Mismatch Risk:' in strRow:
                    listIsGuideRow.append(True)
                    strGuide = strRow.split('\t')[0]
                    if strGuide in listGuides:
                        listIndicesForGuidesOfInt.append(len(listIsGuideRow))
                else:
                    listIsGuideRow.append(False)

        arrayGuideRNARowIndices = np.where(listIsGuideRow)[0]

        arrayGuideRNAOfIntRowIndices = np.array(listIndicesForGuidesOfInt, dtype=np.int)

        listGuideAndPAM = []
        listBindChr = []
        listBindStart = []
        listBindEnd = []
        listNumMisMatch = []
        listOffTargGene = []
        listOffTargSeq = []
        listScores = []
        listStrand = []
        for iGuide in range(len(listIndicesForGuidesOfInt)):
            numStartRow = arrayGuideRNAOfIntRowIndices[iGuide]
            numEndRow = arrayGuideRNARowIndices[arrayGuideRNARowIndices > numStartRow][0]
            listToProc = listFile[numStartRow-1:numEndRow]

            numMisMatches = -1
            for iRow in range(len(listToProc)):
                strFirstRow = listToProc[0]
                strGuideAndPAM = strFirstRow.split('\t')[0]
                strRow = listToProc[iRow]
                if 'Mismatches: ' in strRow:
                    strMisMatches = strRow.split('Mismatches: ')[1]
                    numMisMatches = np.int(strMisMatches)
                elif strRow[0:2] == '\t\t':
                    # extract the required information
                    arrayMisMatchInfo = strRow[2:].split('\t')

                    listGuideAndPAM.append(strGuideAndPAM)
                    listNumMisMatch.append(numMisMatches)
                    listBindChr.append(arrayMisMatchInfo[0])
                    listBindStart.append(arrayMisMatchInfo[1])
                    listBindEnd.append(arrayMisMatchInfo[2])

                    strOffTarg = arrayMisMatchInfo[3]
                    if '/' in strOffTarg:
                        listOffTargGene.append(strOffTarg.split('/')[0])
                        listOffTargSeq.append(strOffTarg.split('/')[1])
                    else:
                        listOffTargGene.append(strOffTarg)
                        listOffTargSeq.append('')

                    if len(arrayMisMatchInfo) >= 5:
                        listScores.append(arrayMisMatchInfo[4])
                    else:
                        listScores.append('')
                    if len(arrayMisMatchInfo) >= 6:
                        listStrand.append(arrayMisMatchInfo[5])
                    else:
                        listStrand.append('')

        dfOffTargets = pd.DataFrame({'Guide_PAM':listGuideAndPAM,
                                     'Binds_chr':listBindChr,
                                     'Binds_start':listBindStart,
                                     'Binds_end':listBindEnd,
                                     'Binds_numMisMatch':listNumMisMatch,
                                     'Binds_HGNC':listOffTargGene,
                                     'Binds_seq':listOffTargSeq,
                                     'Binds_score':listScores,
                                     'Binds_strand':listStrand},
                                    index=np.arange(len(listGuideAndPAM)))

        return dfOffTargets

    def dict_gtf_ensg_to_hgnc(flagResult=False,
                     numRelease=102,
                     strReference='h38',
                     flagPerformExtraction=False):

        strTempFilename = f'GRC{strReference}_{numRelease}_ENSGToHGNC.pickle'

        if not os.path.exists(os.path.join(PathDir.pathRefData, strTempFilename)):
            flagPerformExtraction=True

        if flagPerformExtraction:

            strDataFile = f'Homo_sapiens.GRC{strReference}.{numRelease}.gtf.gz'

            dfEnsDB = pd.read_csv(os.path.join(PathDir.pathRefData, strDataFile),
                                    sep='\t',
                                    compression='gzip',
                                    header=None,
                                    comment='#')

            if numRelease >= 75:
                arrayGeneRowIndices = np.where((dfEnsDB.iloc[:,2]=='gene').values.astype(np.bool))[0]
            else:
                arrayGeneRowIndices = np.where((dfEnsDB.iloc[:,2]=='exon').values.astype(np.bool))[0]
            numGenes = len(arrayGeneRowIndices)

            listGenes = [None]*numGenes
            listGeneENSG = [None]*numGenes

            strFirstGeneDetails = dfEnsDB.iloc[arrayGeneRowIndices[0],8]
            listFirstGeneDetails = strFirstGeneDetails.split(';')
            numGeneNameIndex = np.where(['gene_name "' in strDetails for strDetails in listFirstGeneDetails])[0][0]
            numGeneIDIndex = np.where(['gene_id "' in strDetails for strDetails in listFirstGeneDetails])[0][0]

            for iGene in range(numGenes):
                strGeneDetails = dfEnsDB.iloc[arrayGeneRowIndices[iGene],8]
                listGeneDetails = strGeneDetails.split(';')

                strGene = listGeneDetails[numGeneNameIndex].split('gene_name "')[1].strip('"')
                strENSG = listGeneDetails[numGeneIDIndex].split('gene_id "')[1].strip('"')

                listGenes[iGene] = strGene
                listGeneENSG[iGene] = strENSG

            if len(listGeneENSG) > len(set(listGeneENSG)):

                dfMapped = pd.DataFrame({'ENSG':listGeneENSG, 'HGNC':listGenes})
                dfMapped.drop_duplicates(subset='ENSG', inplace=True)

                dictEnsGeneToHGNC = dict(zip(dfMapped['ENSG'].values.tolist(),
                                             dfMapped['HGNC'].values.tolist()))
            else:
                dictEnsGeneToHGNC = dict(zip(listGeneENSG, listGenes))

            with open(os.path.join(PathDir.pathRefData, strTempFilename), 'wb') as handFile:
                pickle.dump(dictEnsGeneToHGNC, handFile, protocol=pickle.HIGHEST_PROTOCOL)
        else:

            with open(os.path.join(PathDir.pathRefData, strTempFilename), 'rb') as handFile:
                dictEnsGeneToHGNC = pickle.load(handFile)

        return dictEnsGeneToHGNC

    def tan2012_cell_line_genes(flagResult=False):

        # Cell line epithelial & mesenchymal gene lists from:
        #   TZ Tan et al. (2012) [JP Thiery]. Epithelial-mesenchymal transition spectrum quantification
        #    and its efficacy in deciphering survival and drug responses of cancer patients.
        #   DOI: 10.15252/emmm.201404208

        # load the gene lists
        dfGeneLists = pd.read_csv(
            os.path.join(PathDir.pathRefData, 'Thiery_generic_EMT_signatures.txt'),
            sep='\t', header=0, index_col=None)

        # extract as individual lists
        listEpiGenes = dfGeneLists['genes'][dfGeneLists['epiMes_cellLine'] == 'epi'].values.tolist()
        listMesGenes = dfGeneLists['genes'][dfGeneLists['epiMes_cellLine'] == 'mes'].values.tolist()

        # update some more recently defined gene names
        dictNewNames = {'C1orf106':'INAVA',
                        'GPR56':'ADGRG1',
                        'AIM1':'CRYBG1',
                        'C19orf21':'MISP',
                        'C10orf116':'ADIRF',
                        'C12orf24':'FAM216A',
                        'LEPRE1':'P3H1',
                        'LHFP':'LHFPL6',
                        'KDELC1':'POGLUT2',
                        'PTRF':'CAVIN1'}

        for iGene in range(len(listEpiGenes)):
            strGene = listEpiGenes[iGene]
            if strGene in dictNewNames.keys():
                listEpiGenes[iGene] = dictNewNames[strGene]

        for iGene in range(len(listMesGenes)):
            strGene = listMesGenes[iGene]
            if strGene in dictNewNames.keys():
                listMesGenes[iGene] = dictNewNames[strGene]

        # return as a dictionary of lists
        return {'epi_genes': listEpiGenes, 'mes_genes': listMesGenes}

    def tan2012_tissue_genes(flagResult=False):

        # Tissue epithelial & mesenchymal gene lists from:
        #   TZ Tan et al. (2012) [JP Thiery]. Epithelial-mesenchymal transition spectrum quantification
        #    and its efficacy in deciphering survival and drug responses of cancer patients.
        #   DOI: 10.15252/emmm.201404208

        # load the gene lists
        dfGeneLists = pd.read_csv(
            os.path.join(PathDir.pathRefData, 'Thiery_generic_EMT_signatures.txt'),
            sep='\t', header=0, index_col=None)

        # extract as individual lists
        listEpiGenes = dfGeneLists['genes'][dfGeneLists['epiMes_tumor'] == 'epi'].values.tolist()
        listMesGenes = dfGeneLists['genes'][dfGeneLists['epiMes_tumor'] == 'mes'].values.tolist()

        # update some more recently defined gene names
        dictNewNames = {'C14orf139':'SYNE3',
                        'GUCY1B3':'GUCY1B1',
                        'KIAA1462':'JCAD',
                        'LHFP':'LHFPL6',
                        'PTRF':'CAVIN1',
                        'SEPT6':'SEPTIN6',
                        'C19orf21':'MISP',
                        'C1orf106':'INAVA',
                        'GPR56':'ADGRG1',
                        'PPAP2C':'PLPP2'
                        }

        # dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()
        # dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(), dictENSGToHGNC.keys()))

        for iGene in range(len(listEpiGenes)):
            strGene = listEpiGenes[iGene]
            if strGene in dictNewNames.keys():
                listEpiGenes[iGene] = dictNewNames[strGene]

        for iGene in range(len(listMesGenes)):
            strGene = listMesGenes[iGene]
            if strGene in dictNewNames.keys():
                listMesGenes[iGene] = dictNewNames[strGene]

        # return as a dictionary of lists
        return {'epi_genes': listEpiGenes, 'mes_genes': listMesGenes}

    def go_all_gene_with_traversal(flagPerformExtraction=False,
                                   flagProcessGOToArray=False):

        strOutputSaveFile='GO_annot_traversed.pickle'

        if not os.path.exists(os.path.join(PathDir.pathRefData, strOutputSaveFile)):
            flagPerformExtraction=True

        if flagPerformExtraction:
            # extract the GO data
            dfGeneOntology = Extract.goa_human(flagPerformExtraction=False)
            dfGOMapping = Extract.go_obo(flagPerformExtraction=False)
            strTempMembMatrixFile = 'GO_memb_matrix_proc.pickle'
            strInitGraphFile = 'GO_annot-orig_graph.pickle'

            # determine the unique list of GO annotations
            listUniqueGONumsFromMapping = pd.unique(dfGOMapping['ID'].values.ravel()).tolist()
            listUniqueGONumsFromOntology = pd.unique(dfGeneOntology['GO_Num'].values.ravel()).tolist()
            listUniqueMessRNAs = sorted(pd.unique(dfGeneOntology['HGNC'].values.ravel()).tolist())

            listUniqueGONums = sorted(list(set(listUniqueGONumsFromMapping + listUniqueGONumsFromOntology)))

            numUniqueMessRNAs = np.int64(len(listUniqueMessRNAs))
            numUniqueGONums = np.int64(len(listUniqueGONums))

            if np.bitwise_or(not os.path.exists(os.path.join(PathDir.strDataPath, strTempMembMatrixFile)),
                             flagProcessGOToArray):

                print('creating a membership matrix for GO annotations against genes, this may take some time')

                arrayGOMembMatrix = np.zeros((numUniqueMessRNAs, numUniqueGONums), dtype=np.bool)

                dictMessRNAIndices = dict(zip(listUniqueMessRNAs, np.arange(start=0, stop=len(listUniqueMessRNAs))))
                dictGONumIndices = dict(zip(listUniqueGONums, np.arange(start=0, stop=len(listUniqueGONums))))

                arrayProgCounter = np.linspace(start=0, stop=np.shape(dfGeneOntology)[0], num=100)[1:]

                iProg = 0
                for iRow in range(np.shape(dfGeneOntology)[0]):
                    strGene = dfGeneOntology['HGNC'].iloc[iRow]
                    strGOCat = dfGeneOntology['GO_Num'].iloc[iRow]

                    arrayGOMembMatrix[dictMessRNAIndices[strGene], dictGONumIndices[strGOCat]] = True

                    if iRow > arrayProgCounter[iProg]:
                        print(f'\t{(iProg+1)}% complete..')
                        iProg += 1

                dfGOMembMatrix = pd.DataFrame(
                    data=arrayGOMembMatrix,
                    index=listUniqueMessRNAs,
                    columns=listUniqueGONums)
                dfGOMembMatrix.to_pickle(os.path.join(PathDir.strDataPath, strTempMembMatrixFile))

            else:
                print('Loading pre-processed boolean membership matrix for GO annotations..')
                dfGOMembMatrix = pd.read_pickle(os.path.join(PathDir.strDataPath, strTempMembMatrixFile))
                arrayGOMembMatrix = dfGOMembMatrix.values.astype(np.bool)

            if not os.path.exists(os.path.join(PathDir.strDataPath, strInitGraphFile)):
                print('creating a directed graph of GO annotation structure/hierarchy')
                graphAnnotRel = nx.DiGraph()
                graphAnnotRel.add_nodes_from(listUniqueGONums)

                arrayProgCounter = np.linspace(start=0, stop=len(listUniqueGONums), num=100)[1:]

                iProg = 0
                for iGONum in range(len(listUniqueGONums)):
                    strGONum = listUniqueGONums[iGONum]
                    if strGONum in dfGOMapping['ID'].values.tolist():
                        listParents = dfGOMapping['Parents'][dfGOMapping['ID'] == strGONum].tolist()[0]
                        if not (not listParents):
                            for strParent in listParents:
                                graphAnnotRel.add_edge(strGONum, strParent)


                    if iGONum > arrayProgCounter[iProg]:
                        print(f'\t{(iProg+1)}% complete..')
                        iProg += 1

                nx.write_gpickle(graphAnnotRel, os.path.join(PathDir.strDataPath, strInitGraphFile))
            else:
                graphAnnotRel = nx.read_gpickle(os.path.join(PathDir.strDataPath, strInitGraphFile))

            print('attempting to traverse GO graph structure')
            print('warning: this uses an iterative while loop; ensure progression beyond this')
            while Map.has_cycle(graphAnnotRel):
                # while len(nx.find_cycle(graphAnnotRel)) > 0:
                listCycles = nx.find_cycle(graphAnnotRel)
                print('.. attempting to resolve cycle:')
                for iCycleEdge in range(len(listCycles)):
                    print('\t\t.. ' + listCycles[iCycleEdge][0] + ' <- ' + listCycles[iCycleEdge][1])

                listCycleNodes = list()
                for iCycleEdge in range(len(listCycles)):
                    if listCycles[iCycleEdge][0] not in listCycleNodes:
                        listCycleNodes.append(listCycles[iCycleEdge][0])
                    if listCycles[iCycleEdge][1] not in listCycleNodes:
                        listCycleNodes.append(listCycles[iCycleEdge][1])
                arrayNodeGenes = np.sum(dfGOMembMatrix[listCycleNodes].values.astype(np.float), axis=0)
                strLeastPopulatedNode = listCycleNodes[np.argsort(arrayNodeGenes)[0]]

                for iCycleEdge in range(len(listCycles)):
                    if listCycles[iCycleEdge][1] == strLeastPopulatedNode:
                        graphAnnotRel.remove_edge(listCycles[iCycleEdge][0], listCycles[iCycleEdge][1])

            print('attempting topological sort prior to traversal to increase coverage')
            # listTopologicalSort = nx.topological_sort(graphAnnotRel, reverse=True)
            listTopologicalSort = list(reversed(list(nx.topological_sort(graphAnnotRel))))
            for strGO in listTopologicalSort:
                # I think in a digraph, neighbors lists only the parents/input nodes (whereas predecessors does the full
                #  traversal
                listParentNodes = graphAnnotRel.neighbors(strGO)
                if not (not listParentNodes):
                    numNodeMatrixIndex = listUniqueGONums.index(strGO)
                    arrayGeneIndices = np.where(arrayGOMembMatrix[:,numNodeMatrixIndex])[0]
                    for strParent in listParentNodes:
                        numParentNodeMatrixIndex = listUniqueGONums.index(strParent)
                        arrayGOMembMatrix[arrayGeneIndices, numParentNodeMatrixIndex] = True

            dfGOMemb = pd.DataFrame(data=arrayGOMembMatrix,    # values
                                    index=listUniqueMessRNAs,    # 1st column as index
                                    columns=listUniqueGONums)

            # save the full dataframe using pickle
            dfGOMemb.to_pickle(os.path.join(PathDir.strDataPath, strOutputSaveFile))

        else:
            if os.path.exists(os.path.join(PathDir.pathRefData, strOutputSaveFile)):
                # load the data from the specified files
                print('Loading the pre-processed GO annotation (w/ traversal) data frame from ' +
                      os.path.join(PathDir.pathRefData, strOutputSaveFile))

                dfGOMemb = pd.read_pickle(os.path.join(PathDir.pathRefData, strOutputSaveFile))

            else:
                print('Cannot load the pre-processed GO annotation (w/ traversal) data frame, ' +
                      os.path.join(PathDir.pathRefData, strOutputSaveFile) +
                      ' does not exist, change flagPerformExtraction')

        return dfGOMemb

    def go_rnaseq_diffexpr_genes(flagPerformExtraction=False):

        strOutputSaveFile = 'rnaseq_diff_expr_GO_annot.tsv'

        if not os.path.exists(os.path.join(PathDir.pathRefData, strOutputSaveFile)):
            flagPerformExtraction=True

        if flagPerformExtraction:
            dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()
            # dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(), dictENSGToHGNC.keys()))

            dfAllGenes = Process.go_all_gene_with_traversal()

            listRNASeqDEGenesENSG = Process.fig5_rnaseq_gene_lists()['SigEitherLine']
            listRNASeqDEGenes = [dictENSGToHGNC[strGene] for strGene in listRNASeqDEGenesENSG
                                 if strGene in dictENSGToHGNC.keys()]

            dfTargetGenes = dfAllGenes.reindex(listRNASeqDEGenes).copy(deep=True)
            dfTargetGenes[pd.isnull(dfTargetGenes)] = False
            arrayGOObs = np.sum(dfTargetGenes.values.astype(float), axis=0)
            listGOHasGene = [dfTargetGenes.columns.tolist()[i] for i in np.where(arrayGOObs > 0)[0]]

            dfGOMemb = dfTargetGenes[listGOHasGene]
            dfGOMemb.to_csv(os.path.join(PathDir.pathRefData, strOutputSaveFile),
                            sep='\t')

        else:
            dfGOMemb = pd.read_table(os.path.join(PathDir.pathRefData, strOutputSaveFile),
                                     sep='\t', index_col=0)

        return dfGOMemb

    def transcription_factors(flagResult=False):

        # 'Homo_sapiens_TF.txt' & 'Homo_sapiens_TF_cofactors.txt' from AnimalTFDB
        #   now seems to be hosted at http://bioinfo.life.hust.edu.cn/AnimalTFDB/#!/
        dfTFs = pd.read_csv(os.path.join(PathDir.pathRefData, 'Homo_sapiens_TF.txt'),
                            sep='\t', header=0, index_col=1)

        dfCoFactors = pd.read_csv(os.path.join(PathDir.pathRefData, 'Homo_sapiens_TF_cofactors.txt'),
                                  sep='\t', header=0, index_col=1)

        listAllENSG = dfTFs.index.tolist() + dfCoFactors.index.tolist()

        listAllHGNC = dfTFs['Ensembl'].values.tolist() + dfCoFactors['Ensembl'].values.tolist()
        listAllType = ['TF']*np.shape(dfTFs)[0] + ['CoFact']*np.shape(dfCoFactors)[0]

        dfTFs = pd.DataFrame({'ENSG':listAllENSG,
                              'HGNC':listAllHGNC,
                              'Type':listAllType})

        return dfTFs

class PlotFunc:

    def es_ms_landscape(
            flagResult=False,
            handAxIn='undefined',
            handFigIn='undefined'):

        listOfListsCellLineSubtypes = [['luminal'],
                                       ['HER2_amp', 'luminal_HER2_amp'],
                                       ['basal', 'basal_A'],
                                       ['basal_B'],
                                       ['unknown']]

        listSubtypePlotColors = ['#53a9eb', # dark blue
                                 '#aeb3f1', # light blue
                                 '#f5a2bc', # pink
                                 '#ec1c24', # red
                                 '#ababab'] # gray

        listCellLineSubtypesToDisp = ['Luminal',
                                      'HER2$^{++}$',
                                      'Basal',
                                      'Basal B',
                                      'Not classified']
        numCellLineSubtypes = len(listCellLineSubtypesToDisp)

        listSamplesToPlot = ['SUM159_EVC',
                             'SUM159_gAll',
                             'MDAMB231_EVC',
                             'MDAMB231_gAll']

        numMaxXTicks = 5
        numMaxYTicks = 5

        numCellLineMarkerSize = 35
        numCellLineMarkerLineWidth = 1.0

        numScatterZOrder = 11

        dictLineLabel = {'SUM159':'SUM159',
                         'MDAMB231':'MDA-MB-231'}
        dictCondLabel = {'EVC': 'No gRNA',
                         'gAll': 'All gRNAs'}

        dictOfDictOffsets = {'SUM159': {},
                             'MDAMB231': {}}
        dictOfDictOffsets['SUM159']['EVC'] = (-0.08, 0.10)
        dictOfDictOffsets['SUM159']['gAll'] = (0.01, -0.12)
        dictOfDictOffsets['MDAMB231']['EVC'] = (-0.03, 0.10)
        dictOfDictOffsets['MDAMB231']['gAll'] = (0.05, 0.125)

        dictAllScores = Process.all_epi_mes_scores()

        dfTCGAScores = dictAllScores['TCGA']
        dfCCLEScores = dictAllScores['CCLE']
        arrayCCLELineHasNoScore = np.sum(np.isnan(dfCCLEScores.values.astype(float)), axis=1) > 0
        listCCLELineNoScore = [dfCCLEScores.index.tolist()[i] for i in np.where(arrayCCLELineHasNoScore)[0]]
        dfCCLEScores.drop(index=listCCLELineNoScore, inplace=True)

        dfLocalScores = dictAllScores['LocalData']

        dictBrCaLineToType = Process.ccle_brca_subtypes()

        numMinTCGAEpiScore = np.min(dfTCGAScores['Epithelial Score'].values.astype(float))
        numMinES = np.min([np.min(dfLocalScores['Epithelial Score'].values.astype(float)),
                           numMinTCGAEpiScore,
                           np.min(dfCCLEScores['Epithelial Score'].values.astype(float))
                           ])

        numMaxES = np.max([np.max(dfLocalScores['Epithelial Score'].values.astype(float)),
                           np.max(dfTCGAScores['Epithelial Score'].values.astype(float)),
                           np.max(dfCCLEScores['Epithelial Score'].values.astype(float))
                           ])

        numMinMS = np.min([np.min(dfLocalScores['Mesenchymal Score'].values.astype(float)),
                           np.min(dfTCGAScores['Mesenchymal Score'].values.astype(float)),
                           np.min(dfCCLEScores['Mesenchymal Score'].values.astype(float))
                           ])

        numMaxTCGAMesScore = np.max(dfTCGAScores['Mesenchymal Score'].values.astype(float))
        numMaxMS = np.max([np.max(dfLocalScores['Mesenchymal Score'].values.astype(float)),
                           numMaxTCGAMesScore,
                           np.max(dfCCLEScores['Mesenchymal Score'].values.astype(float))
                           ])

        numMinScore = np.min([numMinES, numMinMS])
        numMaxScore = np.max([numMaxES, numMaxMS])

        handAxHex = handAxIn.hexbin(dfTCGAScores['Epithelial Score'].values.astype(float),
                                    dfTCGAScores['Mesenchymal Score'].values.astype(float),
                                    cmap=plt.cm.inferno,
                                    bins='log',
                                    gridsize=50,
                                    alpha=0.5,
                                    lw=0.1,
                                    extent=[numMinScore-0.13, numMaxScore+0.08,
                                            numMinScore-0.13, numMaxScore+0.08],
                                    rasterized=True)

        numOutlineMinEpiScore = numMinTCGAEpiScore-0.02
        numOutlineWidth = 0.1 - numOutlineMinEpiScore
        numOutlineHeight = numMaxTCGAMesScore+0.02 - 0.15
        handAxIn.add_patch(
            matplotlib.patches.Rectangle(
                [numOutlineMinEpiScore, 0.15],
                numOutlineWidth, numOutlineHeight,
                edgecolor='w', lw=1.,
                facecolor=None, fill=False,
                zorder=numScatterZOrder-2))
        handAxIn.add_patch(
            matplotlib.patches.Rectangle(
                [numOutlineMinEpiScore, 0.15],
                numOutlineWidth, numOutlineHeight,
                edgecolor='r', lw=0.5,
                facecolor=None, fill=False,
                zorder=numScatterZOrder-2))

        for iCellLine in range(np.shape(dfCCLEScores)[0]):
            strCellLine = dfCCLEScores.index.tolist()[iCellLine]

            strSubtype = dictBrCaLineToType[strCellLine]
            strColor = listSubtypePlotColors[4]
            for iSubtype in range(len(listOfListsCellLineSubtypes)):
                if strSubtype in listOfListsCellLineSubtypes[iSubtype]:
                    strColor = listSubtypePlotColors[iSubtype]

            plt.scatter(dfCCLEScores['Epithelial Score'].iloc[iCellLine],
                        dfCCLEScores['Mesenchymal Score'].iloc[iCellLine],
                        c=strColor, marker='^', s=30,
                        edgecolors=['black'],
                        linewidths=0.5,
                        zorder=numScatterZOrder)

        for iSampleSet in range(len(listSamplesToPlot)):
            strSampleSet = listSamplesToPlot[iSampleSet]

            listLocalSamplesToPlot = [strSample for strSample in dfLocalScores.index.tolist()
                                      if strSampleSet in strSample]
            for strSample in listLocalSamplesToPlot:
                plt.scatter(dfLocalScores['Epithelial Score'].loc[strSample],
                            dfLocalScores['Mesenchymal Score'].loc[strSample],
                            c='g', marker='^', s=30, edgecolors=['black'],
                            linewidths=0.5,
                            zorder=numScatterZOrder+1)
            strLineShort = strSampleSet.split('_')[0]
            strCondShort = strSampleSet.partition('_')[2]

            strLine = dictLineLabel[strLineShort]
            strCond = dictCondLabel[strCondShort]

            numMeanEpiScore = np.mean(
                dfLocalScores['Epithelial Score'].reindex(listLocalSamplesToPlot).values.astype(float))
            numMeanMesScore = np.mean(
                dfLocalScores['Mesenchymal Score'].reindex(listLocalSamplesToPlot).values.astype(float))

            handAxIn.annotate(
                strLine + '\n' + strCond,
                xy=(numMeanEpiScore, numMeanMesScore), xycoords='data',
                xytext=(numMeanEpiScore + dictOfDictOffsets[strLineShort][strCondShort][0],
                        numMeanMesScore + dictOfDictOffsets[strLineShort][strCondShort][1]),
                textcoords='data',
                size=Plot.numFontSize*0.7, annotation_clip=False,
                horizontalalignment='center', verticalalignment='center', zorder=numScatterZOrder-1,
                bbox=dict(boxstyle="round", fc='w', ec=(0.6, 0.6, 0.6), lw=2, alpha=1.0),
                arrowprops=dict(arrowstyle="wedge,tail_width=0.6",
                                fc=(1.0, 1.0, 1.0), ec=(0.6, 0.6, 0.6),
                                patchA=None,
                                relpos=(0.5, 0.5),
                                connectionstyle="arc3", lw=2, alpha=0.7, zorder=6)
            )

        numMinLimit = numMinScore-0.04
        numMaxLimit = numMaxScore+0.03
        handAxIn.set_xlim([numMinLimit, numMaxLimit])
        handAxIn.set_ylim([numMinLimit, numMaxLimit])

        handAxIn.set_ylabel('Mesenchymal score', fontsize=Plot.numFontSize*1.2)
        handAxIn.set_xlabel('Epithelial score', fontsize=Plot.numFontSize*1.2)

        for handTick in handAxIn.xaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize*0.7)

        for handTick in handAxIn.yaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize*0.7)

        # tidy up the tick locations
        arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
        handAxIn.xaxis.set_major_locator(arrayXTickLoc)

        arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
        handAxIn.yaxis.set_major_locator(arrayYTickLoc)

        numLegendPanelXStart = numMinLimit+0.01
        numLegendPanelYStart = numMinLimit+0.03
        numLegendPanelWidth = 0.29
        numLegendPanelHeight = 0.22

        arrayHexBinPlotPos = handAxIn.get_position()
        numColorBarXStart = arrayHexBinPlotPos.x0 + 0.035
        numColorBarYStart = arrayHexBinPlotPos.y0 + 0.03

        numColorBarLabelXPos = numLegendPanelXStart + 0.2*numLegendPanelWidth
        numColorBarLabelYPos = numLegendPanelYStart + 0.9*numLegendPanelHeight

        numScatterLabelXPos = numLegendPanelXStart + 0.70*numLegendPanelWidth
        numScatterLabelYPos = numColorBarLabelYPos

        numScatterLegendXPos = numLegendPanelXStart + 0.48*numLegendPanelWidth
        numScatterLegendYPos = numLegendPanelYStart + 0.88*numLegendPanelHeight

        numScatterLegendTextXOffset = 0.015
        numScatterLegendTextYOffset = -0.027

        # draw in a patch (white bounding box) as the background for the legend
        handPatch = handAxIn.add_patch(
            matplotlib.patches.Rectangle(
                [numLegendPanelXStart, numLegendPanelYStart],
                numLegendPanelWidth, numLegendPanelHeight,
                edgecolor='k', lw=1.,
                facecolor='w', fill=True))
        handPatch.set_zorder(numScatterZOrder+1)
        handAxIn.text(numLegendPanelXStart + 0.05*numLegendPanelWidth,
                    numLegendPanelYStart + 0.40*numLegendPanelHeight,
                    'log$_{10}$($n_{tumors}$)',
                    fontsize=Plot.numFontSize*0.7,
                    ha='center', va='center',
                    rotation=90, zorder=numScatterZOrder+3)

        handCBarPos=handFigIn.add_axes([numColorBarXStart, numColorBarYStart,
                                         0.01, 0.08])
        handSigColorBar = handFigIn.colorbar(handAxHex, cax=handCBarPos)
        handSigColorBar.ax.tick_params(labelsize=Plot.numFontSize*0.7,
                                       length=0.25, width=0.1)

        handSigColorBar.ax.set_yticklabels([])
        structCBarPos = handCBarPos.get_position()
        handFigIn.text(structCBarPos.x0 + 3.5*structCBarPos.width,
                       structCBarPos.y0 + 0.95*structCBarPos.height,
                       'Most\ntumors',
                       ha='center', va='center',
                       fontsize=Plot.numFontSize*0.7,
                       fontstyle='italic')
        handFigIn.text(structCBarPos.x0 + 3.5*structCBarPos.width,
                       structCBarPos.y0 + 0.05*structCBarPos.height,
                       'No\ntumors',
                       ha='center', va='center',
                       fontsize=Plot.numFontSize*0.7,
                       fontstyle='italic')

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handSigColorBar.ax.spines[strAxLoc].set_linewidth(0.1)

        handAxIn.text(numColorBarLabelXPos, numColorBarLabelYPos,
                      'TCGA sample\ndensity',
                    fontsize=Plot.numFontSize*0.7, horizontalalignment='center', verticalalignment='center',
                    weight='bold', zorder=numScatterZOrder+3)
        handAxIn.text(numScatterLabelXPos, numScatterLabelYPos,
                      'Cell line\nclassification',
                    fontsize=Plot.numFontSize*0.7, horizontalalignment='center', verticalalignment='center',
                    weight='bold', zorder=numScatterZOrder+3)

        for iType in range(numCellLineSubtypes):
            handAxIn.scatter(numScatterLegendXPos,
                             numScatterLegendYPos+numScatterLegendTextYOffset*(iType+1),
                             c=listSubtypePlotColors[iType],
                             clip_on=False,
                             marker='^',
                             s=numCellLineMarkerSize, edgecolor='k',
                             lw=numCellLineMarkerLineWidth,
                             zorder=numScatterZOrder+3)
            handAxIn.text(numScatterLegendXPos+numScatterLegendTextXOffset,
                          numScatterLegendYPos+numScatterLegendTextYOffset*(iType+1),
                          listCellLineSubtypesToDisp[iType],
                          fontsize=Plot.numFontSize*0.7,
                          verticalalignment='center',
                          horizontalalignment='left',
                          zorder=numScatterZOrder+3)
        handAxIn.scatter(numScatterLegendXPos,
                         numScatterLegendYPos + numScatterLegendTextYOffset * (numCellLineSubtypes + 1),
                         c='g',
                         clip_on=False,
                         marker='^',
                         s=numCellLineMarkerSize, edgecolor='k',
                         lw=numCellLineMarkerLineWidth,
                         zorder=numScatterZOrder + 3)
        handAxIn.text(numScatterLegendXPos + numScatterLegendTextXOffset,
                      numScatterLegendYPos + numScatterLegendTextYOffset * (numCellLineSubtypes + 1),
                      'EpiCRISPR samples',
                      fontsize=Plot.numFontSize * 0.7,
                      verticalalignment='center',
                      horizontalalignment='left',
                      zorder=numScatterZOrder + 3)

        return flagResult

    def epi_mes_volcano(flagResult=False,
                        handAxInMDAMB231='undefined',
                        handAxInSUM159='undefined'):

        numAdjPValThresh = 0.05

        dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()
        dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(), dictENSGToHGNC.keys()))

        dfMergedRNA = Process.diff_expr_data()
        listDataGenes = dfMergedRNA.index.tolist()
        setDataGenes = set(listDataGenes)

        for strGene in setDataGenes.difference(set(dictENSGToHGNC.keys())):
            dictENSGToHGNC[strGene] = strGene

        dictEpiMesGenes = Process.tan2012_cell_line_genes()
        listEpiGenes = dictEpiMesGenes['epi_genes']
        listMesGenes = dictEpiMesGenes['mes_genes']

        listEpiGenesENSG = [dictHGNCToENSG[strGene] for strGene in listEpiGenes
                            if strGene in dictHGNCToENSG.keys()]
        listMesGenesENSG = [dictHGNCToENSG[strGene] for strGene in listMesGenes
                            if strGene in dictHGNCToENSG.keys()]

        listOutputGeneOrder = Process.fig5_rnaseq_gene_lists()['HeatmapOrder']

        arrayMaxAbsLogFC = np.max(np.abs(dfMergedRNA['MDAMB231:logFC'].values.astype(float)))

        handAxInMDAMB231.scatter(dfMergedRNA['MDAMB231:logFC'].reindex(listEpiGenesENSG).values.astype(float),
                       -np.log10(dfMergedRNA['MDAMB231:adj.P.Val'].reindex(listEpiGenesENSG).values.astype(
                           float)),
                       lw=0.0,
                       s=4,
                       color='green',
                       alpha=0.9,
                       label='Epithelial',
                                 zorder=5)
        handAxInMDAMB231.scatter(dfMergedRNA['MDAMB231:logFC'].reindex(listMesGenesENSG).values.astype(float),
                       -np.log10(dfMergedRNA['MDAMB231:adj.P.Val'].reindex(listMesGenesENSG).values.astype(
                           float)),
                       lw=0.0,
                       s=4,
                       color='purple',
                       alpha=0.9,
                       label='Mesenchymal',
                                 zorder=5)
        handAxInMDAMB231.scatter(dfMergedRNA['MDAMB231:logFC'].values.astype(float),
                                 -np.log10(dfMergedRNA['MDAMB231:adj.P.Val'].values.astype(float)),
                                 lw=0.0,
                                 s=4,
                                 color='0.7',
                                 alpha=0.4,
                                 label='Other',
                                 zorder=4)
        handAxInMDAMB231.set_xlim([arrayMaxAbsLogFC*-1.05, arrayMaxAbsLogFC*1.05])

        # hide the right and top spines
        handAxInMDAMB231.spines['right'].set_visible(False)
        handAxInMDAMB231.spines['top'].set_visible(False)


        listSigGenes = [dfMergedRNA.index.tolist()[i] for i in
                        np.where(dfMergedRNA['MDAMB231:adj.P.Val'].values.astype(float) < 1E-10)[0]]

        listGenesToLabel = list(set(
            [strGene for strGene in listOutputGeneOrder
             if np.bitwise_and(dfMergedRNA['MDAMB231:adj.P.Val'].loc[strGene].astype(float) < numAdjPValThresh,
                               strGene in listEpiGenesENSG+listMesGenesENSG)] + \
            listSigGenes))


        listHandTextMDAMB231 = [handAxInMDAMB231.text(
            dfMergedRNA['MDAMB231:logFC'].loc[strGene].astype(float),
            -np.log10(dfMergedRNA['MDAMB231:adj.P.Val'].loc[strGene].astype(float)),
            dictENSGToHGNC[strGene],
            fontsize=Plot.numFontSize,
            ha='center')
            for strGene in listGenesToLabel]
        adjust_text(listHandTextMDAMB231,
                    #arrowprops=dict(arrowstyle=None)
                    )

        handAxInMDAMB231.set_xticks([-5, -2.5, 0, 2.5, 5])

        handAxInMDAMB231.set_ylabel('-log$_{10}$(adj. $p$-value)', fontsize=Plot.numFontSize)
        handAxInMDAMB231.set_xlabel('log$_{2}$(fold change)', fontsize=Plot.numFontSize)
        handAxInMDAMB231.set_title('MDA-MB-231', fontsize=Plot.numFontSize*1.25)

        for handTick in handAxInMDAMB231.yaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize)

        for handTick in handAxInMDAMB231.xaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize)

        arrayMaxAbsLogFC = np.max(np.abs(dfMergedRNA['SUM159:logFC'].values.astype(float)))

        handAxInSUM159.scatter(dfMergedRNA['SUM159:logFC'].reindex(listEpiGenesENSG).values.astype(float),
                       -np.log10(dfMergedRNA['SUM159:adj.P.Val'].reindex(listEpiGenesENSG).values.astype(
                           float)),
                       lw=0.0,
                       s=4,
                       color='green',
                       alpha=0.9,
                       label='Epithelial',
                               zorder=5)
        handAxInSUM159.scatter(dfMergedRNA['SUM159:logFC'].reindex(listMesGenesENSG).values.astype(float),
                       -np.log10(dfMergedRNA['SUM159:adj.P.Val'].reindex(listMesGenesENSG).values.astype(
                           float)),
                       lw=0.0,
                       s=4,
                       color='purple',
                       alpha=0.9,
                       label='Mesenchymal',
                               zorder=5)
        handAxInSUM159.scatter(dfMergedRNA['SUM159:logFC'].values.astype(float),
                               -np.log10(dfMergedRNA['SUM159:adj.P.Val'].values.astype(float)),
                               lw=0.0,
                               s=4,
                               color='0.7',
                               alpha=0.4,
                               label='Other',
                               zorder=4)
        handAxInSUM159.set_xlim([arrayMaxAbsLogFC*-1.05, arrayMaxAbsLogFC*1.05])

        listSigGenes = [dfMergedRNA.index.tolist()[i] for i in
                        np.where(dfMergedRNA['SUM159:adj.P.Val'].values.astype(float) < 1E-6)[0]]

        listGenesToLabel = list(set(
            [strGene for strGene in listOutputGeneOrder
             if np.bitwise_and(dfMergedRNA['SUM159:adj.P.Val'].loc[strGene].astype(float) < numAdjPValThresh,
                               strGene in listEpiGenesENSG+listMesGenesENSG)] + \
            listSigGenes))



        listHandTextSUM159 = [handAxInSUM159.text(
            dfMergedRNA['SUM159:logFC'].loc[strGene].astype(float),
            -np.log10(dfMergedRNA['SUM159:adj.P.Val'].loc[strGene].astype(float)),
            dictENSGToHGNC[strGene],
            fontsize=Plot.numFontSize,
            ha='center')
            for strGene in listGenesToLabel]
        adjust_text(listHandTextSUM159,
                    #arrowProps=dict(arrowstyle=None)
                    )

        handAxInSUM159.set_xlabel('log$_{2}$(fold change)', fontsize=Plot.numFontSize)
        handAxInSUM159.set_ylabel('-log$_{10}$(adj. $p$-value)', fontsize=Plot.numFontSize)
        handAxInSUM159.set_title('SUM159', fontsize=Plot.numFontSize*1.25)

        # hide the right and top spines
        handAxInSUM159.spines['right'].set_visible(False)
        handAxInSUM159.spines['top'].set_visible(False)

        for handTick in handAxInSUM159.yaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize)

        handAxInSUM159.set_xticks([-5, -2.5, 0, 2.5, 5])
        for handTick in handAxInSUM159.xaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize)

        plt.legend(loc='lower right',
                   bbox_to_anchor=(1.25, 2.3),
                   fontsize=Plot.numFontSize,
                   scatterpoints=1,
                   ncol=1,
                   facecolor='white',
                   framealpha=1.0)

        return flagResult

    def tcga_histograms(flagResult=False,
                        arrayGridSpecIn=[],
                        listGenesToPlot=[]):


        numOutGeneRow = 0
        numOutGeneCol = 0
        for iGene in range(len(listGenesToPlot)):
            strOutGene = listGenesToPlot[iGene]
            handAx = plt.subplot(arrayGridSpecIn[numOutGeneRow, numOutGeneCol])
            if numOutGeneCol == 0:
                flagLabelY = True
            else:
                flagLabelY = False
            if numOutGeneRow == 2:
                flagLabelX = True
            else:
                flagLabelX = False

            _ = PlotFunc.tcga_sel_gene_hist(handAxIn=handAx,
                                            strGeneIn=strOutGene,
                                            flagLabelYAxis=flagLabelY,
                                            flagLabelXAxis=flagLabelX)
            numOutGeneCol += 1
            if numOutGeneCol >= 2:
                numOutGeneRow += 1
                numOutGeneCol=0

        return flagResult

    def tcga_sel_gene_hist(flagResult=False,
                           handAxIn='undefined',
                           strGeneIn='undefined',
                           flagLabelXAxis=False,
                           flagLabelYAxis=False
                           ):

        handAx2 = handAxIn.twinx()
        dfTCGABrCa = Process.tcga_brca()

        dfTCGABrCaScores = Process.tcga_scores()

        arraySampleIsOfInt = np.bitwise_and(
            dfTCGABrCaScores['Epithelial Score'].values.astype(float) < 0.10,
            dfTCGABrCaScores['Mesenchymal Score'].values.astype(float) > 0.15)

        arraySampleOfIntIndices = np.where(arraySampleIsOfInt)[0]
        listSampleOfInt = [dfTCGABrCaScores.index.tolist()[i] for i in arraySampleOfIntIndices]
        arrayOtherSampleIndicess = np.where(~arraySampleIsOfInt)[0]
        listOtherSamples = [dfTCGABrCaScores.index.tolist()[i] for i in arrayOtherSampleIndicess]


        strTCGAGene = [strGene for strGene in dfTCGABrCa.index.tolist()
                       if strGene.startswith(strGeneIn+'|')][0]

        sliceData = np.log2(dfTCGABrCa.loc[strTCGAGene]+1)
        numMinVal = np.min(sliceData.values.astype(float))
        numMaxVal = np.max(sliceData.values.astype(float))
        numRange = numMaxVal - numMinVal

        arrayHistBins = np.linspace(start=numMinVal-0.05*numRange,
                                    stop=numMaxVal+0.05*numRange,
                                    num=20)

        handAxIn.hist(sliceData[listOtherSamples].values.astype(float),
                      bins=arrayHistBins,
                      zorder=4,
                      alpha=0.8,
                      color='0.6')
        handAxIn.set_xlim([numMinVal-0.05*numRange, numMaxVal+0.05*numRange])

        for handTick in handAxIn.yaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize*0.7)
        for handTick in handAxIn.xaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize*0.7)

        handAx2.hist(sliceData[listSampleOfInt].values.astype(float),
                     bins=arrayHistBins,
                     zorder=5,
                     alpha=0.5,
                     color='#ec1c24')
        handAx2.set_xlim([numMinVal-0.05*numRange, numMaxVal+0.05*numRange])

        handAx2.tick_params(axis='y', labelsize=Plot.numFontSize*0.7, labelcolor='#ec1c24')
        # for handTick in handAx2.yaxis.get_major_ticks():
        #     handTick.label.set_fontsize(Plot.numFontSize*0.7)

        handAxIn.set_title(strGeneIn, fontsize=Plot.numFontSize, fontstyle='italic')

        handAxIn.spines['top'].set_visible(False)
        handAx2.spines['top'].set_visible(False)

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxIn.spines[strAxLoc].set_linewidth(0.1)

        if flagLabelXAxis:
            handAxIn.set_xlabel('Abundance\n(log$_{2}$(TPM+1))', fontsize=Plot.numFontSize)

        if flagLabelYAxis:
            handAxIn.set_ylabel('Frequency', fontsize=Plot.numFontSize)

        return flagResult

    def rnaseq_heatmap_and_annot(flagResult=False,
                                 handAxInHeatmap='undefined',
                                 handAxInHMCMap='undefined',
                                 handAxInAnnot='undefined'):

        dictOutRNASeqCond = {'SUM159:logFC':'SUM159',
                             'MDAMB231:logFC':'MDA-\nMB-231'}

        dictGOLabel = {'GO:0070160':'Tight junction',
                       'GO:0005913':'Adherens junction',
                       'GO:0005911':'Cell-cell junction'}
        listOutputSelGO = list(dictGOLabel.keys())

        dfTFs = Process.transcription_factors()
        listTFsHGNC = dfTFs['ENSG'].values.tolist()
        dictEpiMes = Process.tan2012_cell_line_genes()

        dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()

        dfMergedRNA = Process.diff_expr_data()
        listDataGenes = dfMergedRNA.index.tolist()
        setDataGenes = set(listDataGenes)

        for strGene in setDataGenes.difference(set(dictENSGToHGNC.keys())):
            dictENSGToHGNC[strGene] = strGene

        listFCOutCols = ['MDAMB231:logFC', 'SUM159:logFC']

        listOutputGeneOrder = Process.fig5_rnaseq_gene_lists()['HeatmapOrder']

        numMaxAbsFC = np.max(np.abs(
            np.ravel(dfMergedRNA[listFCOutCols].reindex(listOutputGeneOrder).values.astype(float))))
        handRNASeqHM = handAxInHeatmap.matshow(dfMergedRNA[listFCOutCols].reindex(listOutputGeneOrder),
                                      vmin=-numMaxAbsFC, vmax=numMaxAbsFC,
                                      cmap=plt.cm.PRGn, aspect='auto')

        handAxInHeatmap.set_xticks([])
        handAxInHeatmap.set_yticks([])
        for iGene in range(len(listOutputGeneOrder)):
            strENSG = listOutputGeneOrder[iGene]
            if dictENSGToHGNC[strENSG] == dictENSGToHGNC[strENSG]:
                strGeneOut = dictENSGToHGNC[strENSG]
            else:
                strGeneOut = strENSG

            handAxInHeatmap.text(-0.7, iGene,
                        strGeneOut,
                        ha='right', va='center',
                        fontsize=Plot.numFontSize*0.65,
                        fontstyle='italic')

            if iGene < len(listOutputGeneOrder)-1:
                handAxInHeatmap.axhline(y=iGene+0.5,
                               xmin=0.0, xmax=1.0,
                               color='0.5', lw=0.25)

        for iCond in range(len(listFCOutCols)):
            handAxInHeatmap.text(iCond, -0.7,
                        dictOutRNASeqCond[listFCOutCols[iCond]],
                        ha='center', va='bottom',
                        fontsize=Plot.numFontSize)

            if iCond < len(listFCOutCols)-1:
                handAxInHeatmap.axvline(x=iCond+0.5,
                               ymin=0.0, ymax=1.0,
                               color='0.5', lw=0.25)

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxInHeatmap.spines[strAxLoc].set_linewidth(0.1)

        handSigColorBar = plt.colorbar(handRNASeqHM, cax=handAxInHMCMap,
                                           orientation='horizontal')
        handSigColorBar.ax.tick_params(width=0.5, length=2,
                                       labelsize=Plot.numFontSize * 0.8)

        arrayTickLoc = plt.MaxNLocator(3)
        handSigColorBar.locator = arrayTickLoc
        handSigColorBar.update_ticks()

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAxInHMCMap.spines[strAxLoc].set_linewidth(0.1)

        handAxInHeatmap.text(0.5,
                             len(listOutputGeneOrder)+4.5,
                             'RNA-seq log$_{2}$FC',
                     ha='center', va='bottom',
                     fontsize=Plot.numFontSize)

        dfGeneOntology = Process.go_rnaseq_diffexpr_genes()
        listOutputGeneOrderHGNC = [dictENSGToHGNC[strGene] for strGene in listOutputGeneOrder]
        dfGeneAnnot = dfGeneOntology[listOutputSelGO].reindex(listOutputGeneOrderHGNC)
        dfGeneAnnot['Transcription factor'] = pd.Series([strGene in listTFsHGNC for strGene in listOutputGeneOrderHGNC],
                                      index=listOutputGeneOrderHGNC)
        dfGeneAnnot['Epithelial gene'] = pd.Series([strGene in dictEpiMes['epi_genes'] for strGene in listOutputGeneOrderHGNC],
                                      index=listOutputGeneOrderHGNC)
        dfGeneAnnot['Mesenchymal gene'] = pd.Series([strGene in dictEpiMes['mes_genes'] for strGene in listOutputGeneOrderHGNC],
                                      index=listOutputGeneOrderHGNC)

        listGeneAnnotCols = dfGeneAnnot.columns.tolist()
        for iCol in range(len(listGeneAnnotCols)):
            if listGeneAnnotCols[iCol] in dictGOLabel.keys():
                listGeneAnnotCols[iCol] = listGeneAnnotCols[iCol] + '\n' + dictGOLabel[listGeneAnnotCols[iCol]]

        handAxInAnnot.matshow(np.nan_to_num(dfGeneAnnot.values.astype(float)),
                       cmap=plt.cm.Greys,
                       vmin=0, vmax=1,
                       aspect='auto'
                       )
        handAxInAnnot.set_xticks([])
        handAxInAnnot.set_yticks([])
        for iCol in range(len(listGeneAnnotCols)):
            handAxInAnnot.text(iCol-0.3, -1,
                        listGeneAnnotCols[iCol],
                        ha='left', va='bottom',
                        fontsize=Plot.numFontSize * 0.7,
                        rotation=70
                        )

            if iCol < len(listGeneAnnotCols)-1:
                handAxInAnnot.axvline(x=iCol+0.5,
                               ymin=0.0, ymax=1.0,
                               color='0.5', lw=0.25)

        for iRow in range(np.shape(dfGeneAnnot)[0]):
            if iRow < np.shape(dfGeneAnnot)[0]-1:
                handAxInAnnot.axhline(y=iRow+0.5,
                               xmin=0.0, xmax=1.0,
                               color='0.5', lw=0.25)

        return flagResult

class Plot:

    strOutputLoc = PathDir.pathOutFolder
    listFileFormats = ['png', 'pdf']
    numFontSize = 7
    numScatterMarkerSize = 3

    def figure_five(flagResult=False):

        tupleFigSize = (6.5, 9.5)

        numVolcanoHeight = 0.17
        numVolcanoWidth = 0.37

        numHexbinHeight = 0.33
        numHexbinWidth = numHexbinHeight * (tupleFigSize[1] / tupleFigSize[0])

        numHeatMapPanelHeight = 0.43
        numCMapHeight = 0.0075

        arrayGridSpec = matplotlib.gridspec.GridSpec(
            nrows=3, ncols=2,
            left=0.65, right=0.95,
            bottom=0.05, top=0.38,
            hspace=0.60, wspace=0.65
        )

        numABStrYPos = 0.93
        numACStrXPos = 0.02

        numDStrXPos = 0.61
        numCDStrYPos = 0.40

        numFig5StrYPos = 0.95

        dictPanelLoc = {'Volcano:MDA-MB-231':[0.09, 0.90-numVolcanoHeight, numVolcanoWidth, numVolcanoHeight],
                        'Volcano:SUM159':[0.09, 0.48, numVolcanoWidth, numVolcanoHeight],
                        'HeatMap:RNA-seq':[0.64, 0.47, 0.14, numHeatMapPanelHeight],
                        'HeatMap_cmap:RNA-seq':[0.66, 0.455, 0.10, numCMapHeight],
                        'HeatMap:RNA-seq_GO':[0.79, 0.47, 0.18, numHeatMapPanelHeight],
                        'Hexbin_Landscape':[0.08, 0.05, numHexbinWidth, numHexbinHeight]
                        }

        handFig = plt.figure(figsize=tupleFigSize)

        # # # # # #       #       #       #       #       #       #       #
        # Volcano plots

        # create the axes and pass to the associated plotting function
        handAxMDAMB231 = handFig.add_axes(dictPanelLoc['Volcano:MDA-MB-231'])
        handAxSUM159 = handFig.add_axes(dictPanelLoc['Volcano:SUM159'])
        _ = PlotFunc.epi_mes_volcano(handAxInMDAMB231=handAxMDAMB231,
                                     handAxInSUM159=handAxSUM159)
        handFig.text(numACStrXPos,
                     numABStrYPos,
                     'a',
                     ha='left',
                     va='center',
                     fontsize=Plot.numFontSize*1.5,
                     fontweight='bold')
        handFig.text(numACStrXPos,
                     numFig5StrYPos,
                     'Fig. 5',
                     ha='left',
                     va='center',
                     fontsize=Plot.numFontSize*1.5,
                     fontweight='bold')

        # # # # # #       #       #       #       #       #       #       #
        # RNA-seq logFC & GO results

        handAxHMCMap = handFig.add_axes(dictPanelLoc['HeatMap_cmap:RNA-seq'])
        handAxHeatmap = handFig.add_axes(dictPanelLoc['HeatMap:RNA-seq'])
        handAxAnnot = handFig.add_axes(dictPanelLoc['HeatMap:RNA-seq_GO'])
        _ = PlotFunc.rnaseq_heatmap_and_annot(handAxInHeatmap=handAxHeatmap,
                                              handAxInHMCMap=handAxHMCMap,
                                              handAxInAnnot=handAxAnnot)
        structAxPos = handAxHeatmap.get_position()
        handFig.text(structAxPos.x0-0.5*structAxPos.width,
                     numABStrYPos,
                     'b',
                     ha='left',
                     va='center',
                     fontsize=Plot.numFontSize*1.5,
                     fontweight='bold')

        # # # # # # # # #       #       #       #       #       #       #
        # Hexbin landscape
        handAx = handFig.add_axes(dictPanelLoc['Hexbin_Landscape'])

        _ = PlotFunc.es_ms_landscape(handAxIn=handAx,
                                     handFigIn=handFig)
        handFig.text(numACStrXPos,
                     numCDStrYPos,
                     'c',
                     ha='left',
                     va='center',
                     fontsize=Plot.numFontSize*1.5,
                     fontweight='bold')

        # # # # # # # # #       #       #       #       #       #       #
        # Histograms

        _ = PlotFunc.tcga_histograms(
            arrayGridSpecIn=arrayGridSpec,
            listGenesToPlot=['ZEB1', 'ESRP1', 'F11R', 'MAP7', 'CDS1', 'SH2D3A'])
        handFig.text(numDStrXPos,
                     numCDStrYPos,
                     'd',
                     ha='left',
                     va='center',
                     fontsize=Plot.numFontSize*1.5,
                     fontweight='bold')


        pathOut = os.path.join(Plot.strOutputLoc, 'figure_5')
        for strFormat in Plot.listFileFormats:
            handFig.savefig(os.path.join(pathOut, 'Figure5.'+strFormat),
                            ext=strFormat, dpi=300)
        plt.close(handFig)

        return flagResult

    def off_targets(flagResult=False):

        numMaxXTicks = 4
        numMaxYTicks = 4

        dictENSGToHGNC = BiomartFunctions.IdentMappers.defineEnsemblGeneToHGNCSymbolDict()
        dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(),dictENSGToHGNC.keys()))

        dfOffTargets = Process.off_targets()
        dfGuides = Process.guides()

        listUniqueGuides = [dfGuides['Sequence'].iloc[i] + '_' + dfGuides['PAM'].iloc[i] for i in range(np.shape(dfGuides)[0])]

        arrayGridSpec = matplotlib.gridspec.GridSpec(nrows=1, ncols=4,
                                                     left=0.06, right=0.97,
                                                     bottom=0.13, top=0.90,
                                                     wspace=0.2)

        dfMerged = Process.diff_expr_data()
        listLogFCCols = [strCol for strCol in dfMerged.columns.tolist() if ':logFC' in strCol]
        # listAdjPValCols = [strCol for strCol in dfMerged.columns.tolist() if ':padj' in strCol]
        listAllDiffExprConds = [strCol.split(':logFC')[0] for strCol in listLogFCCols]

        listScoreStrings = dfOffTargets['Binds_score'].values.tolist()
        listScores = []
        for strScore in listScoreStrings:
            if len(strScore) > 0:
                listScores.append(np.float(strScore))
            else:
                listScores.append(0.0)
        arrayScores = np.array(listScores)

        arraySortedByScoreIndices = np.argsort(arrayScores)[::-1]

        # dfOut = dfOffTargets[['Guide_PAM', 'Binds_HGNC', 'Binds_score', 'Binds_numMisMatch']].reindex(arraySortedByScoreIndices)
        # listGenomicLoc = []
        # for iRow in arraySortedByScoreIndices:
        #     listGenomicLoc.append('chr' + '{}'.format(dfOffTargets['Binds_chr'].iloc[iRow]) + ':' +
        #                           '{}'.format(dfOffTargets['Binds_start'].iloc[iRow]) + '-' +
        #                           '{}'.format(dfOffTargets['Binds_end'].iloc[iRow]) +
        #                           ' (' + dfOffTargets['Binds_strand'].iloc[iRow] + ')')
        # dfOut['Position'] = pd.Series(listGenomicLoc, index=dfOut.index.tolist())
        #
        # listGeneFC231AllGuides = []
        # # listGeneFC231Guide4 = []
        # listGeneFC159AllGuides = []
        # # listGeneFC159Guide4 = []
        # for iRow in range(len(arraySortedByScoreIndices)):
        #     strGene = dfOut['Binds_HGNC'].iloc[iRow]
        #     if not strGene == 'NoGene':
        #         strGeneENSG = dictHGNCToENSG[strGene]
        #         if strGeneENSG in dfMerged.index.tolist():
        #             listGeneFC231AllGuides.append(dfMerged['MDAMB231:logFC'].loc[strGeneENSG])
        #             # listGeneFC231Guide4.append(dfMerged['MDAMB231-G4_vs_emptyVect:log2FoldChange'].loc[strGeneENSG])
        #             listGeneFC159AllGuides.append(dfMerged['SUM159:logFC'].loc[strGeneENSG])
        #             # listGeneFC159Guide4.append(dfMerged['SUM159-G4_vs_emptyVect:log2FoldChange'].loc[strGeneENSG])
        #         else:
        #             listGeneFC231AllGuides.append('-')
        #             # listGeneFC231Guide4.append('-')
        #             listGeneFC159AllGuides.append('-')
        #     else:
        #         listGeneFC231AllGuides.append('-')
        #         # listGeneFC231Guide4.append('-')
        #         listGeneFC159AllGuides.append('-')
        #         # listGeneFC159Guide4.append('-')

        # dfOut['MDA-MB-231:logFC:g4_vs_EVC'] = pd.Series(listGeneFC231Guide4, index=dfOut.index.tolist())
        # dfOut['MDA-MB-231:logFC:gAll_vs_EVC'] = pd.Series(listGeneFC231AllGuides, index=dfOut.index.tolist())
        # dfOut['SUM159:logFC:gAll_vs_EVC'] = pd.Series(listGeneFC159AllGuides, index=dfOut.index.tolist())
        # dfOut['SUM159:logFC:g4_vs_EVC'] = pd.Series(listGeneFC159Guide4, index=dfOut.index.tolist())

        # dfOut.to_csv(os.path.join(Plot.strOutputLoc, 'Off_targets_to_check.tsv'), sep='\t', header=True, index=False)

        for strCellLine in Process.listLines:

            listDiffExprConds = [strCol for strCol in listAllDiffExprConds if strCellLine in strCol]

            arrayFlatFC = np.ravel(np.nan_to_num(
                dfMerged[[strCol for strCol in listLogFCCols if strCellLine in strCol]].values.astype(float)))
            numMaxAbsVal = np.max(np.abs(arrayFlatFC))

            for strDiffExpr in listDiffExprConds:
                handFig = plt.figure()
                handFig.set_size_inches(w=9,h=3.5)

                strLogFC = strDiffExpr + ':logFC'
                strPVal = strDiffExpr + ':adj.P.Val'

                arrayGeneToPlot = np.bitwise_and(dfMerged[strLogFC].notnull(),
                                                 dfMerged[strPVal].notnull())

                for iGuide in range(len(listUniqueGuides)):
                    listOffTargets = dfOffTargets['Binds_HGNC'][dfOffTargets['Guide_PAM'] == listUniqueGuides[iGuide]].tolist()
                    listOffTargetGenes = [strTarget
                                          for strTarget in listOffTargets
                                          if not np.bitwise_or(strTarget == 'NoGene', strTarget == 'ZEB1')]
                    listOffTargetGenesENSG = [dictHGNCToENSG[strGene] for strGene in listOffTargetGenes]

                    handAx = plt.subplot(arrayGridSpec[iGuide])

                    handAx.scatter(dfMerged[strLogFC].loc[arrayGeneToPlot].values.astype(float),
                                   -np.log10(dfMerged[strPVal].loc[arrayGeneToPlot].values.astype(float)),
                                   s=Plot.numScatterMarkerSize,
                                   c='0.5',
                                   alpha=0.2,
                                   edgecolors=None)

                    handAx.scatter(dfMerged[strLogFC].loc[listOffTargetGenesENSG].values.astype(float),
                                   -np.log10(dfMerged[strPVal].loc[listOffTargetGenesENSG].values.astype(float)),
                                   s=Plot.numScatterMarkerSize,
                                   edgecolors='r',
                                   lw=1,
                                   c=None)

                    handAx.scatter(dfMerged[strLogFC].loc[dictHGNCToENSG['ZEB1']].astype(float),
                                   -np.log10(dfMerged[strPVal].loc[dictHGNCToENSG['ZEB1']].astype(float)),
                                   s=Plot.numScatterMarkerSize,
                                   edgecolors=[0.0, 1.0, 0.0],
                                   lw=1,
                                   c=None)

                    handAx.set_xlim([-numMaxAbsVal*1.03, numMaxAbsVal*1.03])

                    handAx.set_xlabel('log$_{2}$FC', fontsize=Plot.numFontSize)
                    arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
                    handAx.xaxis.set_major_locator(arrayXTickLoc)
                    arrayXTickLoc = plt.MaxNLocator(numMaxYTicks)
                    handAx.xaxis.set_major_locator(arrayXTickLoc)

                    if iGuide == 0:
                        handAx.set_ylabel('-log$_{10}$($p$-value)', fontsize=Plot.numFontSize)
                        for handTick in handAx.yaxis.get_major_ticks():
                            handTick.label.set_fontsize(Plot.numFontSize)
                    else:
                        handAx.set_ylabel('')
                        handAx.set_yticklabels([])

                    handAx.set_title(listUniqueGuides[iGuide], fontsize=Plot.numFontSize)
                    for handTick in handAx.xaxis.get_major_ticks():
                        handTick.label.set_fontsize(Plot.numFontSize)

                handFig.text(x=0.5, y=0.99, s=strDiffExpr,
                             ha='center', va='top', fontsize=Plot.numFontSize*1.3)

                for strExt in Plot.listFileFormats:
                    handFig.savefig(os.path.join(Plot.strOutputLoc,
                                                 'ZEB1_offTarg_'+strCellLine+'_'+strDiffExpr+'.'+strExt),
                                    ext=strExt, dpi=300)
                plt.close(handFig)

        return flagResult

    def emt_tfs(flagResult=False):

        listTFs = ['ATF2', 'ATF3', 'ETS1', 'FOSL1',
                   'FOXA1', 'FOXA2', 'FOXC2', 'FOXO4',
                   'GRHL2', 'GSC', 'KLF8', 'MEF2C', 'MYC',
                   'SNAI1', 'SNAI2', 'SOX9', 'TCF3', 'TCF4', 'TWIST1',
                   'TWIST2', 'ZEB1', 'ZEB2']

        # load dictionaries for mapping between ENSG and HGNC
        dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()
        dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(), dictENSGToHGNC.keys()))

        listTFsENSG = [dictHGNCToENSG[strGene] for strGene in listTFs]

        # load the data
        dfMergedRNA = Process.diff_expr_data()
        # listDataGenes = dfMergedRNA.index.tolist()
        listDataColumns = dfMergedRNA.columns.tolist()
        listPValCols = [strCol for strCol in listDataColumns if 'adj.P.Val' in strCol]
        listCellLines = [strCol.split(':adj.P.Val')[0] for strCol in listPValCols]

        listlogFCCols = [strCol for strCol in listDataColumns if ':logFC' in strCol]

        dfTFRNA = dfMergedRNA.reindex(listTFsENSG).copy(deep=True)
        dfTFRNA.index = listTFs

        handFig = plt.figure(figsize=(6,4))

        arrayLogFCData = dfTFRNA[listlogFCCols].values.astype(float)
        numMaxAbsLogFC = np.max(np.abs(np.ravel(arrayLogFCData[~np.isnan(arrayLogFCData)])))

        handAx = handFig.add_axes([0.25, 0.15, 0.70, 0.80])
        handAx.matshow(arrayLogFCData,
                       cmap=plt.cm.PRGn,
                       vmin=-numMaxAbsLogFC,
                       vmax=numMaxAbsLogFC,
                       aspect='auto')

        handAx.set_yticklabels([])
        for iGene in range(len(listTFs)):
            handAx.text(-0.55,
                        iGene,
                        listTFs[iGene],
                        ha='right', va='center',
                        fontstyle='italic',
                        fontsize=5)

        handAx.set_xticklabels([])
        for iLine in range(len(listCellLines)):
            handAx.text(iLine,
                        len(listTFs)+1,
                        listCellLines[iLine],
                        ha='center', va='top',
                        fontsize=5)

        for iLine in range(len(listCellLines)):
            strLine = listCellLines[iLine]
            for iGene in range(len(listTFs)):
                strTF = listTFs[iGene]
                if np.isnan(dfTFRNA.loc[strTF, f'{strLine}:adj.P.Val']):
                    handAx.scatter(iLine, iGene, marker='o', color='k', s=5)
                else:
                    if dfTFRNA.loc[strTF, f'{strLine}:adj.P.Val'].astype(float) < 0.05:
                        a=1
                    else:
                        handAx.scatter(iLine, iGene, marker='o', color='k', s=5)

        for strFormat in Plot.listFileFormats:
            handFig.savefig(os.path.join(Plot.strOutputLoc, 'emt_tfs.' + strFormat),
                            ext=strFormat, dpi=300)
        plt.close(handFig)

        return flagResult

class Output:

    def merged_results(flagResult=False):

        listFilesToMerge = ['voom-limma_MDAMB231_G4-EVC_diffExpr.csv',
                            'voom-limma_MDAMB231_GAll-EVC_diffExpr.csv',
                            'voom-limma_SUM159_G4-EVC_diffExpr.csv',
                            'voom-limma_SUM159_GAll-EVC_diffExpr.csv']

        listColsToDrop = ['AveExpr', 't', 'P.Value']

        dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()

        listDFToMerge = []
        for strFile in listFilesToMerge:
            strCond = strFile.split('voom-limma_')[1].split('_diffExpr.csv')[0]
            strLine = strCond.split('_')[0]
            strComp = strCond.split('_')[1]
            strComp = strComp.replace('EVC', 'NoGuide')
            strCompClean = strComp.replace('-', '_vs_')
            dfIn = pd.read_csv(os.path.join(PathDir.pathProcRNAData, strFile),
                               sep=',', index_col=0, header=0)
            dfIn.drop(columns=listColsToDrop, inplace=True)

            listRenamedCol = [f'{strLine}_{strCompClean}:{strCol}' for strCol in dfIn.columns.tolist()]
            dfIn.columns = listRenamedCol
            listDFToMerge.append(dfIn)

        dfMerged = pd.concat(listDFToMerge, axis=1, sort=True)
        for strGene in set(dfMerged.index.tolist()).difference(set(dictENSGToHGNC.keys())):
            dictENSGToHGNC[strGene] = 'failed_map'

        listHGNC = [dictENSGToHGNC[strGene] for strGene in dfMerged.index.tolist()]

        listColumns = dfMerged.columns.tolist()
        dfMerged['HGNC'] = pd.Series(listHGNC, index=dfMerged.index.tolist())

        dfMerged[['HGNC'] + listColumns].to_csv(os.path.join(PathDir.pathProcRNAData, 'MergedDiffExpr.tsv'),
                                                sep='\t', header=True, index=True)

        return flagResult

# _ = Process.all_epi_mes_scores()
# _ = Plot.figure_five()
_ = Plot.emt_tfs()

# _ = Output.merged_results()






# dfDiffExpr = Process.diff_expr_data()
# listCommonGenes = Process.common_rna_genes()

# dfTCGARNA = Process.tcga_brca()
# dfTCGAScores = Process.tcga_scores(flagPerformExtraction=True)

# dfCCLERNA = Process.ccle_brca()
# dfCCLEScores = Process.ccle_scores(flagPerformExtraction=True)

# dictBrCaLineSubtype = Process.ccle_brca_subtypes()

# dfLocalScores = Process.local_scores()

# _ = Process.go_rnaseq_diffexpr_genes()

# _ = Process.tcga_brca()


# _ = Plot.off_targets()

# _ = PlotFunc.es_ms_landscape()



# _ = Plot.es_ms_landscape(
#     strDataLoc=PathDir.pathProcResults,
#     strFileName='ES_vs_MS_landscape',
#     listQuantFileNames=Process.listQuantFiles,
#     listLocalCellLines = Process.listLines,
#     listOfListsConds=Process.listOfListsConds,
#     dictOfDictXYOffsets=dictOfDictOffsets)


# _ = Plot.heatmap()

# _ = Plot.diff_expr_genes_vs_tcga()

# _ = Plot.sig_meth_vs_rna()

# _ = Plot.ppi()


# _ = Process.tcga_scores()


# dfTCGA = TCGAFunctions.PanCancer.extract_mess_rna()



