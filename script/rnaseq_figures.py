from adjustText import adjust_text
# import BiomartFunctions
# import CCLETools
# import Cellosaurus
import copy
import csv
# import DepMapTools
# import ENSEMBLTools
# import HGNCFunctions
# import GeneOntology
# import GeneSetScoring
# import IlluminaFunctions
# from lifelines import CoxPHFitter
# from lifelines.estimation import KaplanMeierFitter
# from lifelines.statistics import logrank_test as KMlogRankTest
import matplotlib
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
# import miRBaseFunctions
import networkx as nx
import numpy as np
import os
import pandas as pd
import pickle
# import ProteinInteractionTools
import scipy.cluster.hierarchy as SciPyClus
import scipy.stats as scs
import sys
# import TCGAFunctions

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

class Process:

    listQuantFiles = [
        'Waryah_Dec2017_MDAMB231_ZEB1-epiCRISPR_QuantGeneLevel.csv',
        'Waryah_Sept2017_SUM159_ZEB1-epiCRISPR_QuantGeneLevel.csv']

    listDiffExprFiles = [
        'voom-limma_MDAMB231_GAll-EVC_diffExpr.csv',
        'voom-limma_SUM159_GAll-EVC_diffExpr.csv']
    listLines = ['MDAMB231',
                 'SUM159']
    listLinesForDisp = ['MDA-MB-231',
                        'SUM159']
    listOfListsConds = [['NTC', 'NTC', 'NTC',
                         'EVC', 'EVC', 'EVC',
                         'g4', 'g4', 'g4',
                         'gAll', 'gAll', 'gAll'],
                        ['NTC', 'NTC', 'NTC',
                         'EVC', 'EVC', 'EVC',
                         'g4', 'g4', 'g4',
                         'gAll', 'gAll', 'gAll']]

    def quant_data(flagResult=False):

        listDFToMerge = []

        for iFile in range(len(Process.listQuantFiles)):
            strFileName = Process.listQuantFiles[iFile]

            dfIn = pd.read_table(os.path.join(PathDir.pathProcResults, strFileName),
                                 sep=',', header=0, index_col=0)

            if strFileName.startswith('Differentially expressed genes'):
                listColumns = dfIn.columns.tolist()
                listColToDrop = [strCol for strCol in listColumns if strCol not in Process.listOfListsConds[iFile]]
                dfIn.drop(columns=listColToDrop, inplace=True)
                listIndex = dfIn.index.tolist()
                listIndexClean = [strIndex.split('.')[0] for strIndex in listIndex]
                dfIn.index = listIndexClean
                listColRenamed = []
                for strCol in Process.listOfListsConds[iFile]:

                    if strCol.startswith('EVC'):
                        strColNew = strCol[-1] + '_SUM159-Mult_EVC'
                    elif strCol.startswith('AllTF4.'):
                        strColNew = strCol[-1] + '_SUM159-Mult_AllTF'

                    listColRenamed.append(strColNew)

                dfIn.rename(columns=dict(zip(Process.listOfListsConds[iFile], listColRenamed)), inplace=True)
                dfIn = dfIn[~dfIn.index.duplicated(keep='first')]


            listDFToMerge.append(dfIn)

        dfMerged = pd.concat(listDFToMerge, axis=1)

        return dfMerged

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
                    flagPerformExtraction=False):

        strTempFileName = 'TCGA-BRCA-EpiMesScores.pickle'

        if not os.path.exists(os.path.join(PathDir.pathProcResults, strTempFileName)):
            flagPerformExtraction = True


        if flagPerformExtraction:

            dfTCGABrCa = Process.tcga_brca()
            listTCGAGenes = dfTCGABrCa.index.tolist()
            listTCGASamples = dfTCGABrCa.columns.tolist()
            numSamples = len(listTCGASamples)

            listSharedGenes = Process.common_rna_genes()
            setOutGenes = set(listSharedGenes)

            listTCGAOutGenes = [strGene for strGene in listTCGAGenes if strGene.split('|')[0] in setOutGenes]

            # dictEpiMesCellLine = Process.tan2012_cell_line_genes()
            # listEpiCellLineGenes = dictEpiMesCellLine['epi_genes']
            # listMesCellLineGenes = dictEpiMesCellLine['mes_genes']
            dictEpiMesTissue = GeneSetScoring.ExtractList.tan2012_tumour_genes()
            listEpiTissueGenes = dictEpiMesTissue['epi_genes']
            listMesTissueGenes = dictEpiMesTissue['mes_genes']

            # create lists of the cell line/tissue epithelial/mesenchymal gene lists for scoring
            # listOutputEpiCellLineGenes = list(set(listEpiCellLineGenes).intersection(setOutGenes))
            # listOutputMesCellLineGenes = list(set(listMesCellLineGenes).intersection(setOutGenes))
            listOutputEpiTissueGenes = list(set(listEpiTissueGenes).intersection(setOutGenes))
            listOutputMesTissueGenes = list(set(listMesTissueGenes).intersection(setOutGenes))

            listOutputEpiTissueGenesMatched = [strGene for strGene in listTCGAGenes
                                               if strGene.split('|')[0] in listOutputEpiTissueGenes]
            listOutputMesTissueGenesMatched = [strGene for strGene in listTCGAGenes
                                               if strGene.split('|')[0] in listOutputMesTissueGenes]

            arrayTCGAEpiScores = np.zeros(numSamples, dtype=float)
            arrayTCGAMesScores = np.zeros(numSamples, dtype=float)
            for iSample in range(numSamples):
                print('Patient ' + '{}'.format(iSample))
                strSample = listTCGASamples[iSample]
                arrayTCGAEpiScores[iSample] = \
                    GeneSetScoring.FromInput.single_sample_rank_score(
                        listAllGenes=listTCGAOutGenes,
                        arrayTranscriptAbundance=dfTCGABrCa[strSample].reindex(listTCGAOutGenes).values.astype(float),
                        listUpGenesToScore=listOutputEpiTissueGenesMatched,
                        flagApplyNorm=True)
                arrayTCGAMesScores[iSample] = \
                    GeneSetScoring.FromInput.single_sample_rank_score(
                        listAllGenes=listTCGAOutGenes,
                        arrayTranscriptAbundance=dfTCGABrCa[strSample].reindex(listTCGAOutGenes).values.astype(float),
                        listUpGenesToScore=listOutputMesTissueGenesMatched,
                        flagApplyNorm=True)

            dfScores = pd.DataFrame({'Epithelial Score':arrayTCGAEpiScores,
                                     'Mesenchymal Score':arrayTCGAMesScores},
                                    index=listTCGASamples)
            dfScores.to_pickle(os.path.join(PathDir.pathProcResults, strTempFileName))

        else:

            dfScores = pd.read_pickle(os.path.join(PathDir.pathProcResults, strTempFileName))

        return dfScores

    def tcga_brca(flagResult=False,
                  flagPerformExtraction=False):

        strTempFileName = 'TCGA_BrCa_PreProc_RNA.pickle'

        if not os.path.exists(os.path.join(PathDir.pathProcResults, strTempFileName)):
            flagPerformExtraction = True

        if flagPerformExtraction:
            #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #
            # extract the TCGA pan-cancer RNA-seq data
            dfTCGA = TCGAFunctions.PanCancer.extract_mess_rna()
            listTCGARNASamples = dfTCGA.columns.tolist()

            # extract the TCGA pan-cancer patient metadata
            dfMeta = TCGAFunctions.PanCancer.extract_clinical_data()
            dfMeta.set_index('bcr_patient_barcode', inplace=True)

            # identify patients which are flagged as the breast cancer cohort
            listBRCAPatients = dfMeta[dfMeta['type']=='BRCA'].index.tolist()

            # extract primary tumour (index 01) samples from the full sample list
            listBRCASamples = [strSample for strSample in listTCGARNASamples
                               if np.bitwise_and(strSample[0:len('TCGA-NN-NNNN')] in listBRCAPatients,
                                                 strSample[13:15]=='01')]

            #take this subset
            dfTCGABrCa = dfTCGA[listBRCASamples]
            dfTCGABrCa.to_pickle(os.path.join(PathDir.pathProcResults, strTempFileName))
        else:
            dfTCGABrCa = pd.read_pickle(os.path.join(PathDir.pathProcResults, strTempFileName))

        return dfTCGABrCa

    def ccle_brca(flagResult=False,
                  flagPerformExtraction=False):

        dfCCLE = DepMapTools.Load.all_rnaseq_data()
        listCellLines = dfCCLE.index.tolist()
        listBrCaLines = [strLine for strLine in listCellLines if '_BREAST' in strLine]

        return dfCCLE.reindex(listBrCaLines)

    def ccle_scores(flagResult=False,
                    flagPerformExtraction=False):

        strTempFileName = 'CCLE-BRCA-EpiMesScores.pickle'

        if not os.path.exists(os.path.join(PathDir.pathProcResults, strTempFileName)):
            flagPerformExtraction = True

        if flagPerformExtraction:

            dfCCLEBrCa = Process.ccle_brca()
            listCCLEGenes = dfCCLEBrCa.columns.tolist()
            listCCLELines = dfCCLEBrCa.index.tolist()
            numCellLines = len(listCCLELines)

            listSharedGenes = Process.common_rna_genes()
            setOutGenes = set(listSharedGenes)

            listCCLEOutGenes = [strGene for strGene in listCCLEGenes if strGene.split(' (')[0] in setOutGenes]

            dictEpiMesCellLine = Process.tan2012_cell_line_genes()
            listEpiCellLineGenes = dictEpiMesCellLine['epi_genes']
            listMesCellLineGenes = dictEpiMesCellLine['mes_genes']

            # create lists of the cell line/tissue epithelial/mesenchymal gene lists for scoring
            listOutputEpiCellLineGenes = list(set(listEpiCellLineGenes).intersection(setOutGenes))
            listOutputMesCellLineGenes = list(set(listMesCellLineGenes).intersection(setOutGenes))


            listOutputEpiCellLineGenesMatched = [strGene for strGene in listCCLEGenes
                                               if strGene.split(' (')[0] in listOutputEpiCellLineGenes]
            listOutputMesCellLineGenesMatched = [strGene for strGene in listCCLEGenes
                                               if strGene.split(' (')[0] in listOutputMesCellLineGenes]

            arrayCCLEEpiScores = np.zeros(numCellLines, dtype=float)
            arrayCCLEMesScores = np.zeros(numCellLines, dtype=float)
            for iCellLine in range(numCellLines):
                print('Cell Line ' + '{}'.format(iCellLine))
                # strLine = listCCLELines[iCellLine]
                arrayCCLEEpiScores[iCellLine] = \
                    GeneSetScoring.FromInput.single_sample_rank_score(
                        listAllGenes=listCCLEOutGenes,
                        arrayTranscriptAbundance=dfCCLEBrCa[listCCLEOutGenes].iloc[iCellLine].values.astype(float),
                        listUpGenesToScore=listOutputEpiCellLineGenesMatched,
                        flagApplyNorm=True)
                arrayCCLEMesScores[iCellLine] = \
                    GeneSetScoring.FromInput.single_sample_rank_score(
                        listAllGenes=listCCLEOutGenes,
                        arrayTranscriptAbundance=dfCCLEBrCa[listCCLEOutGenes].iloc[iCellLine].values.astype(float),
                        listUpGenesToScore=listOutputMesCellLineGenesMatched,
                        flagApplyNorm=True)

            dfScores = pd.DataFrame({'Epithelial Score':arrayCCLEEpiScores,
                                     'Mesenchymal Score':arrayCCLEMesScores},
                                    index=listCCLELines)
            dfScores.to_pickle(os.path.join(PathDir.pathProcResults, strTempFileName))

        else:

            dfScores = pd.read_pickle(os.path.join(PathDir.pathProcResults, strTempFileName))

        return dfScores

    def local_scores(flagResult=False,
                    flagPerformExtraction=False):

        strTempFileName = 'LocalData-EpiMesScores.pickle'

        if not os.path.exists(os.path.join(PathDir.pathProcResults, strTempFileName)):
            flagPerformExtraction = True

        if flagPerformExtraction:

            dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()
            dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(), dictENSGToHGNC.keys()))

            listSharedGenes = Process.common_rna_genes()
            listSharedGenesENSG = [dictHGNCToENSG[strGene] for strGene in listSharedGenes]

            dfAbund = Process.quant_data()
            dfAbund = dfAbund.reindex(listSharedGenesENSG)

            arrayRowHasNaN = np.any(dfAbund.isnull().values.astype(bool), axis=1)
            listRowHasNaN = [dfAbund.index.tolist()[i] for i in np.where(arrayRowHasNaN)[0]]
            dfAbund.drop(index=listRowHasNaN, inplace=True)

            listDataGenes = dfAbund.index.tolist()
            for strGene in set(listDataGenes).difference(set(dictENSGToHGNC.keys())):
                dictENSGToHGNC[strGene] = strGene
            listDataGenesHGNC = [dictENSGToHGNC[strGene] for strGene in listDataGenes]
            dfAbund.index = listDataGenesHGNC
            listDataSamples = dfAbund.columns.tolist()
            numSamples = len(listDataSamples)


            setOutGenes = set(listSharedGenes)

            listDataOutGenes = [strGene for strGene in listDataGenesHGNC if strGene in setOutGenes]

            dictEpiMesCellLine = Process.tan2012_cell_line_genes()
            listEpiCellLineGenes = dictEpiMesCellLine['epi_genes']
            listMesCellLineGenes = dictEpiMesCellLine['mes_genes']

            # create lists of the cell line/tissue epithelial/mesenchymal gene lists for scoring
            listOutputEpiCellLineGenes = list(set(listEpiCellLineGenes).intersection(setOutGenes))
            listOutputMesCellLineGenes = list(set(listMesCellLineGenes).intersection(setOutGenes))


            listOutputEpiCellLineGenesMatched = [strGene for strGene in listDataOutGenes
                                               if strGene in listOutputEpiCellLineGenes]
            listOutputMesCellLineGenesMatched = [strGene for strGene in listDataOutGenes
                                               if strGene in listOutputMesCellLineGenes]

            arrayCCLEEpiScores = np.zeros(numSamples, dtype=float)
            arrayCCLEMesScores = np.zeros(numSamples, dtype=float)
            for iCellLine in range(numSamples):
                print('Cell Line ' + '{}'.format(iCellLine))
                strSample = listDataSamples[iCellLine]
                arrayCCLEEpiScores[iCellLine] = \
                    GeneSetScoring.FromInput.single_sample_rank_score(
                        listAllGenes=listDataOutGenes,
                        arrayTranscriptAbundance=dfAbund[strSample].reindex(listDataOutGenes).values.astype(float),
                        listUpGenesToScore=listOutputEpiCellLineGenesMatched,
                        flagApplyNorm=True)
                arrayCCLEMesScores[iCellLine] = \
                    GeneSetScoring.FromInput.single_sample_rank_score(
                        listAllGenes=listDataOutGenes,
                        arrayTranscriptAbundance=dfAbund[strSample].reindex(listDataOutGenes).values.astype(float),
                        listUpGenesToScore=listOutputMesCellLineGenesMatched,
                        flagApplyNorm=True)

            dfScores = pd.DataFrame({'Epithelial Score':arrayCCLEEpiScores,
                                     'Mesenchymal Score':arrayCCLEMesScores},
                                    index=listDataSamples)
            dfScores.to_pickle(os.path.join(PathDir.pathProcResults, strTempFileName))

        else:

            dfScores = pd.read_pickle(os.path.join(PathDir.pathProcResults, strTempFileName))

        return dfScores

    def ccle_brca_subtypes(flagResult=False):

        dfMeta = DepMapTools.Load.cell_line_metadata()
        dfMeta.set_index('CCLE_Name', inplace=True)

        listBrCaLines = [strLine for strLine in dfMeta.index.tolist() if '_BREAST' in strLine]
        listSubtype = dfMeta['lineage_molecular_subtype'].reindex(listBrCaLines).values.tolist()

        for iLine in range(len(listBrCaLines)):
            if not listSubtype[iLine] == listSubtype[iLine]:
                listSubtype[iLine] = 'unknown'

        return dict(zip(listBrCaLines, listSubtype))

    def output_genes(flagResult=False):

        numOutBothUpRegGenes = 24
        numOutBothDownRegGenes = 8

        numPerCondPerLineExtraGenes = 5


        dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()
        dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(), dictENSGToHGNC.keys()))

        dfMergedRNA = Process.diff_expr_data()
        listDataGenes = dfMergedRNA.index.tolist()

        listCondsOut = ['SUM159', 'MDAMB231']
        dfRanks = pd.DataFrame(data=np.zeros((len(listDataGenes), len(listCondsOut)), dtype=float),
                               index=listDataGenes,
                               columns=listCondsOut)

        for iCond in range(len(listCondsOut)):
            strCond = listCondsOut[iCond]
            arrayLogFC = np.nan_to_num(dfMergedRNA[f'{strCond}:logFC'].values.astype(float))
            arrayAdjPVal = dfMergedRNA[f'{strCond}:adj.P.Val'].values.astype(float)
            arrayAdjPVal[np.isnan(arrayAdjPVal)] = 1.0
            listGenesRanked = [listDataGenes[i] for i in np.argsort(np.product((arrayLogFC, -np.log10(arrayAdjPVal)), axis=0))]
            dfRanks.loc[listGenesRanked, strCond] = np.arange(start=1, stop=len(listGenesRanked)+1)

        arrayProdRankAcrossCond = np.product(dfRanks.values.astype(float), axis=1)
        arraySortedByProdRank = np.argsort(arrayProdRankAcrossCond)
        listSortedByProdRank = [listDataGenes[i] for i in arraySortedByProdRank]

        listOutputDownGenes = listSortedByProdRank[0:numOutBothDownRegGenes]
        listOutputUpGenes = listSortedByProdRank[-numOutBothUpRegGenes:]

        listExtraOutputUpGenes = []
        listExtraOutputDownGenes = []
        for iCond in range(len(listCondsOut)):
            listExtraUpForCond = []
            iUp = np.max(dfRanks.iloc[:,iCond].values.astype(float))
            while len(listExtraUpForCond) < numPerCondPerLineExtraGenes:
                strGeneToTest = dfRanks[dfRanks.iloc[:,iCond]==iUp].index.tolist()[0]
                if not strGeneToTest in listOutputUpGenes:
                    listExtraUpForCond.append(strGeneToTest)
                iUp -= 1
            listExtraOutputUpGenes += listExtraUpForCond

            listExtraDownForCond = []
            iDown = 1
            while len(listExtraDownForCond) < numPerCondPerLineExtraGenes:
                strGeneToTest = dfRanks[dfRanks.iloc[:,iCond]==iDown].index.tolist()[0]
                if not strGeneToTest in listOutputDownGenes:
                    listExtraDownForCond.append(strGeneToTest)
                iDown += 1
            listExtraOutputDownGenes += listExtraDownForCond

        listOutputGeneOrder = listOutputDownGenes + \
                              listExtraDownForCond + \
                              listExtraOutputUpGenes + \
                              listOutputUpGenes

        return listOutputGeneOrder


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

        dfGeneLists = pd.read_csv(
            os.path.join(PathDir.pathRefData, 'Thiery_generic_EMT_sig_cellLine.txt'), 
            sep='\t', header=0, index_col=None)

        listEpiGenes = dfGeneLists['cellLine_sig'][dfGeneLists['epi_mes'] == 'epi'].values.tolist()
        listMesGenes = dfGeneLists['cellLine_sig'][dfGeneLists['epi_mes'] == 'mes'].values.tolist()

        return {'epi_genes': listEpiGenes, 'mes_genes': listMesGenes}

class PlotFunc:

    def heat_map_with_sig_overlay_GO_enrich_and_DepMap(
            flagResult=False,
            strOutFilename='undefined',
            listGenesForDisp=['undefined'],
            listDiffExprCondsToShow=['undefined']):

        dfDepMap = \
            DepMapTools.Extract.depmap_data_by_genes_and_cell_lines(
                listGenesOfInt=listGenesForDisp,
                listCellLinesOfIntCCLE=['SUM159PT_BREAST',
                                        'MDAMB231_BREAST',
                                        'BT549_BREAST',
                                        'HCC1395_BREAST',
                                        'HS578T_BREAST'])


        dictENSGToHGNC = BiomartFunctions.IdentMappers.defineEnsemblGeneToHGNCSymbolDict()
        dictENSGToHGNC.pop('ENSG00000276776', None)
        dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(), dictENSGToHGNC.keys()))

        dfCellLineDiffExpr = Process.diff_expr_data()

        listAllGenesENSG = dfCellLineDiffExpr.index.tolist()
        for strGene in list(set(listAllGenesENSG).difference(set(dictENSGToHGNC.keys()))):
            dictENSGToHGNC[strGene] = strGene
            dictHGNCToENSG[strGene] = strGene

        listGenesForDispNoHGNC = [strGene for strGene in listGenesForDisp if strGene[0:len('ENSG')] == 'ENSG']

        #ENSG00000276776 should be ENSG00000165929
        #ENSG00000230439
        for strGene in listGenesForDispNoHGNC:
            dictHGNCToENSG[strGene] = strGene

        listAllGenesHGNC = [dictENSGToHGNC[strGene] for strGene in listAllGenesENSG]
        dfCellLineDiffExpr['Symbol:HGNC'] = pd.Series(listAllGenesHGNC, index=dfCellLineDiffExpr.index.tolist())

        dfGeneOntology = GeneOntology.Map.all_transcripts_with_traversal()

        listLogFCCols = [strCol + ':log2FoldChange' for strCol in listDiffExprCondsToShow]
        listPValCols = [strCol + ':padj' for strCol in listDiffExprCondsToShow]

        listGenesForDispENSG = [dictHGNCToENSG[strGene] for strGene in listGenesForDisp]

        listOfDfGOEnrich = GeneOntology.EnrichmentTest.calc_relative_enrichment(
            listOfListsMessRNAsToCheck=[listGenesForDisp],
            listAllDataMessRNAs=listAllGenesHGNC,
            stringListFormat='HGNC',
            numMinGenesInCategory=3,
            numMaxGenesInCategory=500)

        dfSigUpDiffExprGO = \
            listOfDfGOEnrich[0][
                np.bitwise_and(listOfDfGOEnrich[0]['GOEnrichPVal'] < 1E-3,
                               listOfDfGOEnrich[0]['GOObsNum'] > 2)].copy(deep=True)

        arrayGOCatRankedByEnrich = np.argsort(dfSigUpDiffExprGO['GOEnrichPVal'].values.astype(float))

        listGOCatsEnriched = [dfSigUpDiffExprGO.index.tolist()[i] for i in arrayGOCatRankedByEnrich]

        if len(listGOCatsEnriched) > 25:
            listGOCatsForDisp = [listGOCatsEnriched[i] for i in range(25)]
        else:
            listGOCatsForDisp = listGOCatsEnriched

        dfGOCatMembership = pd.DataFrame(data=np.zeros((len(listGenesForDisp), len(listGOCatsForDisp)),
                                                       dtype=float),
                                         index=listGenesForDisp,
                                         columns=listGOCatsForDisp)

        for strCat in listGOCatsForDisp:
            listGenesInThisGOCat = []
            if strCat in dfGeneOntology.columns.tolist():
                listGenesInThisGOCat = [dfGeneOntology.index.tolist()[i] for i in
                                        np.where(dfGeneOntology[strCat])[0]]
                listMatchedGenesInThisCat = list(set(listGenesInThisGOCat).intersection(set(listGenesForDisp)))
            dfGOCatMembership[strCat].loc[listMatchedGenesInThisCat] = 1.0


        arrayLogFCData = dfCellLineDiffExpr[listLogFCCols].loc[listGenesForDispENSG].values.astype(float)
        numMaxAbsFC = np.max(np.abs(np.nan_to_num(np.ravel(arrayLogFCData))))

        arrayPValData = np.nan_to_num(np.ravel(dfCellLineDiffExpr[listPValCols].loc[listGenesForDispENSG]))
        numMinNonZeroPVal = np.min(arrayPValData[arrayPValData > 0])

        arrayHorizLineSpacing = np.arange(start=3, stop=len(listGenesForDisp), step=4)

        handFig = plt.figure()
        handFig.set_size_inches(w=8, h=8)


        numLHSPos = 0.11

        numFCDataWidth = 0.10
        numGOCatWidth = 0.50
        numWidthSpacerFCToGO = 0.01

        numGOLHS = numLHSPos + numFCDataWidth + numWidthSpacerFCToGO + 0.05
        numDepMapLHS = numGOLHS + numGOCatWidth + 0.06

        handAx = handFig.add_axes([numLHSPos, 0.02, numFCDataWidth, 0.85])

        handHM = handAx.matshow(arrayLogFCData,
                                cmap=plt.cm.PRGn,
                                vmin=-numMaxAbsFC,
                                vmax=numMaxAbsFC,
                                aspect='auto',
                                interpolation=None)

        for iGene in range(len(listGenesForDisp)):
            strGene = listGenesForDisp[iGene]
            strENSG = listGenesForDispENSG[iGene]
            handAx.text(-0.6, iGene, strGene, fontstyle='italic',
                        ha='right', va='center', fontsize=Plot.numFontSize * 0.4)
            handAx.text(1.55, iGene, strGene, fontstyle='italic',
                        ha='left', va='center', fontsize=Plot.numFontSize * 0.4)

            for iCol in range(len(listDiffExprCondsToShow)):
                strColPVal = listDiffExprCondsToShow[iCol] + ':padj'
                if np.isnan(dfCellLineDiffExpr[strColPVal].loc[strENSG]):
                    numPVal = 1
                elif dfCellLineDiffExpr[strColPVal].loc[strENSG] == 0:
                    numPVal = numMinNonZeroPVal
                else:
                    numPVal = dfCellLineDiffExpr[strColPVal].loc[strENSG].astype(float) + \
                              numMinNonZeroPVal

                if np.bitwise_and(np.log10(numPVal) > -12, np.log10(numPVal) < -1):
                    handAx.scatter(x=np.float(iCol), y=np.float(iGene),
                                   marker='o',
                                   s=(np.log10(numPVal) + 12) * 1.2,
                                   c='0.6',
                                   edgecolors='0.6',
                                   alpha=1.0)
                elif np.log10(numPVal) > -1:
                    handAx.scatter(x=np.float(iCol), y=np.float(iGene),
                                   marker='o',
                                   s=(np.log10(numPVal) + 12) * 1.2,
                                   c='k')

        for iGene in arrayHorizLineSpacing:
            handAx.axhline(y=iGene+0.5, xmin=0.0, xmax=1.0, lw=0.75, color='w', zorder=10)
            handAx.axhline(y=iGene+0.5, xmin=0.0, xmax=1.0, lw=0.5, color='k', zorder=10)

        handAx.set_xticks([])
        handAx.set_yticks([])

        for iCol in range(len(listDiffExprCondsToShow)):
            strDiffExprCond = listDiffExprCondsToShow[iCol]
            strCellLine = strDiffExprCond.split('-')[0]
            strConds = strDiffExprCond.split('-')[1]
            strCondOne = strConds.split('_vs_')[0]
            strCondTwo = strConds.split('_vs_')[1]
            strToDisp = strCellLine + '\n' + strCondOne + '\nvs. ' + strCondTwo
            handAx.text(iCol-0.4, -0.6, strToDisp,
                        ha='left', va='bottom', fontsize=Plot.numFontSize * 0.5,
                        rotation=60)

        handPValLegend = handFig.add_axes([0.02, 0.25, 0.01, 0.20])
        handPValLegend.scatter(x=0.0, y=0.0,
                               s=(np.log10(1) + 12) * 1.2,
                               marker='o',
                               c='k')
        handPValLegend.scatter(x=0.0, y=1.0,
                               s=(np.log10(1E-3) + 12) * 1.2,
                               marker='o',
                               c='0.6')
        handPValLegend.scatter(x=0.0, y=2.0,
                               s=(np.log10(1E-6) + 12) * 1.2,
                               marker='o',
                               c='0.6')
        handPValLegend.scatter(x=0.0, y=3.0,
                               s=(np.log10(1E-9) + 12) * 1.2,
                               marker='o',
                               c='0.6')
        handPValLegend.set_ylim([-0.2, 3.2])
        handPValLegend.set_xticks([])
        handPValLegend.yaxis.set_ticks_position('right')
        handPValLegend.yaxis.set_label_position('left')
        handPValLegend.set_ylabel('log$_{10}$(adj. $p$-val)', fontsize=Plot.numFontSize*0.6)
        handPValLegend.set_yticks([0, 1, 2, 3])
        handPValLegend.set_yticklabels(['0', '-3', '-6', '-9'], fontsize=Plot.numFontSize*0.6)
        # for handTick in handPValLegend.yaxis.get_major_ticks():
        #     handTick.label.set_fontsize(Plot.numFontSize*0.6)


        handHMCMapAx = handFig.add_axes([0.02, 0.55, 0.01, 0.20])
        handAbundColorBar = handFig.colorbar(handHM,
                                             cax=handHMCMapAx,
                                             ticks=[-numMaxAbsFC, 0, numMaxAbsFC],
                                             format='%02.1f')
        handAbundColorBar.ax.tick_params(labelsize=Plot.numFontSize*0.6)
        handHMCMapAx.yaxis.set_label_position('left')
        handHMCMapAx.set_ylabel('log$_{2}$(FC)', fontsize=Plot.numFontSize*0.6)


        handAx = handFig.add_axes([numGOLHS, 0.02, numGOCatWidth, 0.85])

        handHM = handAx.matshow(dfGOCatMembership.values.astype(float),
                                cmap=plt.cm.gray_r,
                                vmin=0,
                                vmax=1,
                                aspect='auto',
                                interpolation=None)

        handAx.set_xticks([])
        handAx.set_yticks([])

        for iGene in arrayHorizLineSpacing:
            handAx.axhline(y=iGene+0.5, xmin=0.0, xmax=1.0, lw=0.75, color='w', zorder=10)
            handAx.axhline(y=iGene+0.5, xmin=0.0, xmax=1.0, lw=0.5, color='k', zorder=10)

        for iCol in range(len(listGOCatsForDisp)):
            strGONum = listGOCatsForDisp[iCol]
            strGOCat = dfSigUpDiffExprGO['GOCat'].loc[strGONum]
            handAx.text(iCol, np.float(len(listGenesForDisp))/2, strGONum + ' :: ' + strGOCat,
                        ha='center', va='center', fontsize=Plot.numFontSize * 0.6,
                        rotation=90,
                        color='k',
                        path_effects=[PathEffects.withStroke(linewidth=2, foreground="w")],
                        zorder=11)


        handAx = handFig.add_axes([numDepMapLHS, 0.02, 0.08, 0.85])
        handDepMapHM = handAx.matshow(dfDepMap.values.astype(float),
                                cmap=plt.cm.BrBG,
                                vmin=-1.5,
                                vmax=1.5,
                                aspect='auto',
                                interpolation=None)

        for iGene in range(len(listGenesForDisp)):
            strGene = listGenesForDisp[iGene]
            handAx.text(-0.55, iGene, strGene, fontstyle='italic',
                        ha='right', va='center', fontsize=Plot.numFontSize * 0.4)

        for iGene in arrayHorizLineSpacing:
            handAx.axhline(y=iGene+0.5, xmin=0.0, xmax=1.0, lw=0.75, color='w', zorder=10)
            handAx.axhline(y=iGene+0.5, xmin=0.0, xmax=1.0, lw=0.5, color='k', zorder=10)

        handAx.set_xticks([])
        handAx.set_yticks([])

        for iCol in range(np.shape(dfDepMap)[1]):
            strCol = dfDepMap.columns.tolist()[iCol]
            handAx.text(iCol-0.4, -0.5, strCol,
                        ha='left', va='bottom', fontsize=Plot.numFontSize * 0.5,
                        rotation=60,
                        color='k')

        handDepMapHMCMapAx = handFig.add_axes([numDepMapLHS+0.1, 0.55, 0.01, 0.20])
        handAbundColorBar = handFig.colorbar(handDepMapHM,
                                             cax=handDepMapHMCMapAx,
                                             ticks=[-1.5, 0, 1.5],extend='both',
                                             format='%02.1f')
        handAbundColorBar.ax.tick_params(labelsize=Plot.numFontSize*0.6)
        handHMCMapAx.yaxis.set_label_position('left')
        handHMCMapAx.set_title('Score', fontsize=Plot.numFontSize*0.6)

        handFig.savefig(os.path.join(Plot.strOutputLoc, strOutFilename),
                        ext='png', dpi=300)

        plt.close(handFig)

        return flagResult

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
                             'SUM159_gRNA_All',
                             'MDA231_EVC',
                             'MDA231_gRNA_All',
                             'SUM159-Mult_EVC',
                             'SUM159-Mult_AllTF']

        numMaxXTicks = 5
        numMaxYTicks = 5

        numCellLineMarkerSize = 35
        numCellLineMarkerLineWidth = 1.0

        numScatterZOrder = 11

        dictLineLabel = {'SUM159':'SUM159',
                         'MDA231':'MDA-MB-231',
                         'SUM159-Mult':'SUM159^'}
        dictCondLabel = {'EVC': 'Empty\nvector',
                         'gRNA_All': 'ZEB1 gRNAs',
                         'AllTF': 'All TFs'}

        dictOfDictOffsets = {'SUM159': {},
                             'MDA231': {},
                             'SUM159-Mult': {}}
        dictOfDictOffsets['SUM159']['EVC'] = (-0.08, 0.02)
        dictOfDictOffsets['SUM159']['gRNA_All'] = (-0.03, -0.07)
        dictOfDictOffsets['MDA231']['EVC'] = (0.04, 0.10)
        dictOfDictOffsets['MDA231']['gRNA_All'] = (-0.07, -0.07)
        dictOfDictOffsets['SUM159-Mult']['EVC'] = (0.01, 0.10)
        dictOfDictOffsets['SUM159-Mult']['AllTF'] = (-0.08, -0.05)

        dfTCGAScores = Process.tcga_scores()
        dfCCLEScores = Process.ccle_scores()
        dfLocalScores = Process.local_scores(flagPerformExtraction=False)

        dictBrCaLineToType = Process.ccle_brca_subtypes()

        numMinES = np.min([np.min(dfLocalScores['Epithelial Score'].values.astype(float)),
                           np.min(dfTCGAScores['Epithelial Score'].values.astype(float)),
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

        numMaxMS = np.max([np.max(dfLocalScores['Mesenchymal Score'].values.astype(float)),
                           np.max(dfTCGAScores['Mesenchymal Score'].values.astype(float)),
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
                                    extent=[numMinScore-0.13, numMaxScore+0.08,
                                            numMinScore-0.13, numMaxScore+0.08])

        for iCellLine in range(np.shape(dfCCLEScores)[0]):
            strCellLine = dfCCLEScores.index.tolist()[iCellLine]

            strSubtype = dictBrCaLineToType[strCellLine]
            strColor = listSubtypePlotColors[4]
            for iSubtype in range(len(listOfListsCellLineSubtypes)):
                if strSubtype in listOfListsCellLineSubtypes[iSubtype]:
                    strColor = listSubtypePlotColors[iSubtype]

            plt.scatter(dfCCLEScores['Epithelial Score'].iloc[iCellLine],
                        dfCCLEScores['Mesenchymal Score'].iloc[iCellLine],
                        c=strColor, marker='^', s=25,
                        edgecolors=['k'],
                        zorder=numScatterZOrder)

        for iSampleSet in range(len(listSamplesToPlot)):
            strSampleSet = listSamplesToPlot[iSampleSet]

            listLocalSamplesToPlot = [strSample for strSample in dfLocalScores.index.tolist()
                                      if strSample.endswith(strSampleSet)]
            for strSample in listLocalSamplesToPlot:
                plt.scatter(dfLocalScores['Epithelial Score'].loc[strSample],
                            dfLocalScores['Mesenchymal Score'].loc[strSample],
                            c='g', marker='^', s=25, edgecolors=['k'],
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
                horizontalalignment='center', verticalalignment='center', zorder=6,
                bbox=dict(boxstyle="round", fc='w', ec=(0.6, 0.6, 0.6), lw=2, alpha=1.0),
                arrowprops=dict(arrowstyle="wedge,tail_width=0.6",
                                fc=(1.0, 1.0, 1.0), ec=(0.6, 0.6, 0.6),
                                patchA=None,
                                relpos=(0.5, 0.5),
                                connectionstyle="arc3", lw=2, alpha=0.7, zorder=6)
            )
            handAxIn.set_xlim([numMinScore-0.10, numMaxScore+0.05])
            handAxIn.set_ylim([numMinScore-0.10, numMaxScore+0.05])


        handAxIn.set_ylabel('Mesenchymal score', fontsize=Plot.numFontSize*0.7)
        handAxIn.set_xlabel('Epithelial score', fontsize=Plot.numFontSize*0.7)

        for handTick in handAxIn.xaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize*0.7)

        for handTick in handAxIn.yaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize*0.7)

            # tidy up the tick locations
        arrayXTickLoc = plt.MaxNLocator(numMaxXTicks)
        handAxIn.xaxis.set_major_locator(arrayXTickLoc)


        arrayYTickLoc = plt.MaxNLocator(numMaxYTicks)
        handAxIn.yaxis.set_major_locator(arrayYTickLoc)

        arrayHexBinPlotPos = handAxIn.get_position()
        numColorBarXStart = arrayHexBinPlotPos.x0 + 0.03*arrayHexBinPlotPos.width
        numColorBarYStart = arrayHexBinPlotPos.y0 + 0.05*arrayHexBinPlotPos.height

        numLegendPanelXStart = numMinScore-0.095
        numLegendPanelYStart = numMinScore-0.095
        numLegendPanelWidth = 0.30
        numLegendPanelHeight = 0.24

        numColorBarLabelXPos = numMinScore - 0.03
        numColorBarLabelYPos = numMinScore + 0.11

        numScatterLabelXPos = numMinScore + 0.11
        numScatterLabelYPos = numColorBarLabelYPos

        numScatterLegendXPos = numMinScore + 0.05
        numScatterLegendYPos = numMinScore + 0.10

        numScatterLegendYSpacing = 0.045*(numMaxScore - numMinScore)

        numScatterLegendHMLESystemXOffset = 0.015 * (numMaxScore - numMinScore)
        numScatterLegendTextXOffset = 0.015
        numScatterLegendTextYOffset = -0.03

        # draw in a patch (white bounding box) as the background for the legend
        handPatch = handAxIn.add_patch(matplotlib.patches.Rectangle([numLegendPanelXStart, numLegendPanelYStart],
                                                      numLegendPanelWidth, numLegendPanelHeight,
                                                      edgecolor='k', lw=1.,
                                                      facecolor='w', fill=True))
        handPatch.set_zorder(numScatterZOrder+1)
        handAxIn.text(numLegendPanelXStart + 0.25*numLegendPanelWidth,
                    numLegendPanelYStart + 0.40*numLegendPanelHeight,
                    'log$_{10}$($n_{tumours}$)',
                    fontsize=Plot.numFontSize*0.7,
                    ha='center', va='center',
                    rotation=90, zorder=numScatterZOrder+3)

        arrayCBarPos=handFigIn.add_axes([numColorBarXStart,numColorBarYStart,0.02,0.08])
        handSigColorBar = handFigIn.colorbar(handAxHex,cax=arrayCBarPos)
        handSigColorBar.ax.tick_params(labelsize=Plot.numFontSize*0.7)

        arrayTickLoc = plt.MaxNLocator(5)
        handSigColorBar.locator = arrayTickLoc
        handSigColorBar.update_ticks()

        listOutTickLabels = ['']*5
        listOutTickLabels[0] = 'Low'
        listOutTickLabels[-1] = 'High'

        handSigColorBar.ax.set_yticklabels(listOutTickLabels)


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

        listEpiGenesENSG = [dictHGNCToENSG[strGene] for strGene in listEpiGenes if strGene in dictHGNCToENSG.keys()]
        listMesGenesENSG = [dictHGNCToENSG[strGene] for strGene in listMesGenes if strGene in dictHGNCToENSG.keys()]

        listOutputGeneOrder = Process.output_genes()

        arrayMaxAbsLogFC = np.max(np.abs(dfMergedRNA['MDAMB231:logFC'].values.astype(float)))

        handAxInMDAMB231.scatter(dfMergedRNA['MDAMB231:logFC'].values.astype(float),
                       -np.log10(dfMergedRNA['MDAMB231:adj.P.Val'].values.astype(float)),
                       lw=0.0,
                       s=4,
                       color='0.7',
                       alpha=0.4,
                       label='All')
        handAxInMDAMB231.scatter(dfMergedRNA['MDAMB231:logFC'].reindex(listEpiGenesENSG).values.astype(float),
                       -np.log10(dfMergedRNA['MDAMB231:adj.P.Val'].reindex(listEpiGenesENSG).values.astype(
                           float)),
                       lw=0.0,
                       s=4,
                       color='green',
                       alpha=0.9,
                       label='Epithelial')
        handAxInMDAMB231.scatter(dfMergedRNA['MDAMB231:logFC'].reindex(listMesGenesENSG).values.astype(float),
                       -np.log10(dfMergedRNA['MDAMB231:adj.P.Val'].reindex(listMesGenesENSG).values.astype(
                           float)),
                       lw=0.0,
                       s=4,
                       color='purple',
                       alpha=0.9,
                       label='Mesenchymal')
        handAxInMDAMB231.set_xlim([arrayMaxAbsLogFC*-1.05, arrayMaxAbsLogFC*1.05])

        # hide the right and top spines
        handAxInMDAMB231.spines['right'].set_visible(False)
        handAxInMDAMB231.spines['top'].set_visible(False)

        listHandTextMDAMB231 = [handAxInMDAMB231.text(
            dfMergedRNA['MDAMB231:logFC'].loc[strGene].astype(float),
            -np.log10(dfMergedRNA['MDAMB231:adj.P.Val'].loc[strGene].astype(float)),
            dictENSGToHGNC[strGene],
            fontsize=Plot.numFontSize * 0.70,
            ha='center')
            for strGene in listOutputGeneOrder
            if np.bitwise_and(dfMergedRNA['MDAMB231:adj.P.Val'].loc[strGene].astype(float) < 0.05,
                              strGene in listEpiGenesENSG+listMesGenesENSG)]
        adjust_text(listHandTextMDAMB231,
                    arrowProps=dict(arrowstyle=None))

        handAxInMDAMB231.set_ylabel('-log$_{10}$(adj. $p$-value)', fontsize=Plot.numFontSize*0.7)
        handAxInMDAMB231.set_title('MDA-MB-231', fontsize=Plot.numFontSize)

        for handTick in handAxInMDAMB231.yaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize * 0.7)

        for handTick in handAxInMDAMB231.xaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize * 0.7)

        arrayMaxAbsLogFC = np.max(np.abs(dfMergedRNA['SUM159:logFC'].values.astype(float)))

        handAxInSUM159.scatter(dfMergedRNA['SUM159:logFC'].values.astype(float),
                       -np.log10(dfMergedRNA['SUM159:adj.P.Val'].values.astype(float)),
                       lw=0.0,
                       s=4,
                       color='0.7',
                       alpha=0.4,
                       label='Other')
        handAxInSUM159.scatter(dfMergedRNA['SUM159:logFC'].reindex(listEpiGenesENSG).values.astype(float),
                       -np.log10(dfMergedRNA['SUM159:adj.P.Val'].reindex(listEpiGenesENSG).values.astype(
                           float)),
                       lw=0.0,
                       s=4,
                       color='green',
                       alpha=0.9,
                       label='Epithelial')
        handAxInSUM159.scatter(dfMergedRNA['SUM159:logFC'].reindex(listMesGenesENSG).values.astype(float),
                       -np.log10(dfMergedRNA['SUM159:adj.P.Val'].reindex(listMesGenesENSG).values.astype(
                           float)),
                       lw=0.0,
                       s=4,
                       color='purple',
                       alpha=0.9,
                       label='Mesenchymal')
        handAxInSUM159.set_xlim([arrayMaxAbsLogFC*-1.05, arrayMaxAbsLogFC*1.05])


        listHandTextSUM159 = [handAxInSUM159.text(
            dfMergedRNA['SUM159:logFC'].loc[strGene].astype(float),
            -np.log10(dfMergedRNA['SUM159:adj.P.Val'].loc[strGene].astype(float)),
            dictENSGToHGNC[strGene],
            fontsize=Plot.numFontSize * 0.70,
            ha='center')
            for strGene in listOutputGeneOrder
            if np.bitwise_and(dfMergedRNA['SUM159:adj.P.Val'].loc[strGene].astype(float) < 0.05,
                              strGene in listEpiGenesENSG+listMesGenesENSG)]
        adjust_text(listHandTextSUM159,
                    arrowProps=dict(arrowstyle=None)
                    )

        handAxInSUM159.set_xlabel('log$_{2}$(fold change)', fontsize=Plot.numFontSize*0.7)
        handAxInSUM159.set_ylabel('-log$_{10}$(adj. $p$-value)', fontsize=Plot.numFontSize*0.7)
        handAxInSUM159.set_title('SUM159', fontsize=Plot.numFontSize)

        # hide the right and top spines
        handAxInSUM159.spines['right'].set_visible(False)
        handAxInSUM159.spines['top'].set_visible(False)

        for handTick in handAxInSUM159.yaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize * 0.7)

        for handTick in handAxInSUM159.xaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize * 0.7)

        plt.legend(loc='lower center',
                   bbox_to_anchor=(0.5, 1.13),
                   fontsize=Plot.numFontSize * 0.6,
                   scatterpoints=1,
                   ncol=3,
                   facecolor='white',
                   framealpha=1.0)

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

        sliceData = dfTCGABrCa.loc[strTCGAGene]
        numMinVal = np.min(sliceData.values.astype(float))
        numMaxVal = np.max(sliceData.values.astype(float))
        numRange = numMaxVal - numMinVal

        arrayHistBins = np.linspace(start=numMinVal-0.05*numRange,
                                    stop=numMaxVal+0.05*numRange,
                                    num=30)

        handAxIn.hist(sliceData[listOtherSamples].values.astype(float),
                      bins=arrayHistBins,
                      zorder=4,
                      alpha=0.7,
                      color='0.6')
        handAxIn.set_xlim([numMinVal-0.05*numRange, numMaxVal+0.05*numRange])

        for handTick in handAxIn.yaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize*0.7)
        for handTick in handAxIn.xaxis.get_major_ticks():
            handTick.label.set_fontsize(Plot.numFontSize*0.7)

        handAx2.hist(sliceData[listSampleOfInt].values.astype(float),
                     bins=arrayHistBins,
                     zorder=5,
                     alpha=0.7,
                     color='#ec1c24')
        handAx2.set_xlim([numMinVal-0.05*numRange, numMaxVal+0.05*numRange])
        handAx2.tick_params(axis='y', labelsize=Plot.numFontSize*0.7, labelcolor='#ec1c24')
        # for handTick in handAx2.yaxis.get_major_ticks():
        #     handTick.label.set_fontsize(Plot.numFontSize*0.7)

        handAxIn.set_title(strGeneIn, fontsize=Plot.numFontSize*0.7)

        if flagLabelXAxis:
            handAxIn.set_xlabel('Abundance', fontsize=Plot.numFontSize*0.7)

        if flagLabelYAxis:
            handAxIn.set_ylabel('Frequency', fontsize=Plot.numFontSize*0.7)

        return flagResult


class Plot:

    strOutputLoc = PathDir.pathOutFolder
    listFileFormats = ['png', 'pdf']
    numFontSize = 7
    numScatterMarkerSize = 3

    def figure_five(flagResult=False):

        tupleFigSize = (6.5, 9.5)

        numVolcanoHeight = 0.17
        numVolcanoWidth = 0.35

        numHexbinHeight = 0.33
        numHexbinWidth = numHexbinHeight * (tupleFigSize[1] / tupleFigSize[0])

        numHeatMapPanelHeight = 0.43
        numHeatMapPanelXLabelPos = 0.41
        numCMapHeight = 0.0075

        listOutputSelGO = ['GO:0070160',
                           'GO:0005913',
                           'GO:0005911']

        dictGOLabel = {'GO:0070160':'Tight junction',
                       'GO:0005913':'Adherens junction',
                       'GO:0005911':'Cell-cell junction'}

        arrayGridSpec = matplotlib.gridspec.GridSpec(
            nrows=3, ncols=2,
            left=0.65, right=0.95,
            bottom=0.05, top=0.38,
            hspace=0.50, wspace=0.65
        )

        dictPanelLoc = {'Volcano:MDA-MB-231':[0.07, 0.90-numVolcanoHeight, numVolcanoWidth, numVolcanoHeight],
                        'Volcano:SUM159':[0.07, 0.48, numVolcanoWidth, numVolcanoHeight],
                        'HeatMap:RNA-seq':[0.64, 0.47, 0.14, numHeatMapPanelHeight],
                        'HeatMap_cmap:RNA-seq':[0.66, 0.455, 0.10, numCMapHeight],
                        'HeatMap:RNA-seq_GO':[0.79, 0.47, 0.18, numHeatMapPanelHeight],
                        'Hexbin_Landscape':[0.07, 0.05, numHexbinWidth, numHexbinHeight]
                        }

        dictOutRNASeqCond = {'SUM159:logFC':'SUM159',
                             'MDAMB231:logFC':'MDA-MB-231'}

        dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()

        dfMergedRNA = Process.diff_expr_data()
        listDataGenes = dfMergedRNA.index.tolist()
        setDataGenes = set(listDataGenes)

        for strGene in setDataGenes.difference(set(dictENSGToHGNC.keys())):
            dictENSGToHGNC[strGene] = strGene

        listFCOutCols = ['MDAMB231:logFC', 'SUM159:logFC']

        listOutputGeneOrder = Process.output_genes()

        # dfGeneOntology = GeneOntology.Map.all_transcripts_with_traversal()
        #
        # dfTF = GeneSetScoring.ExtractList.transcription_factors()
        # listTFs = dfTF['ENSG'].values.tolist()
        # dictEpiMes = GeneSetScoring.ExtractList.tan2012_tumour_genes()


        handFig = plt.figure(figsize=tupleFigSize)

        # # # # # #       #       #       #       #       #       #       #
        # Volcano plots

        handAxMDAMB231 = handFig.add_axes(dictPanelLoc['Volcano:MDA-MB-231'])
        handAxSUM159 = handFig.add_axes(dictPanelLoc['Volcano:SUM159'])

        _ = PlotFunc.epi_mes_volcano(handAxInMDAMB231=handAxMDAMB231,
                                     handAxInSUM159=handAxSUM159)

        # # # # # #       #       #       #       #       #       #       #
        # RNA-seq logFC

        handAx = handFig.add_axes(dictPanelLoc['HeatMap:RNA-seq'])

        numMaxAbsFC = np.max(np.abs(
            np.ravel(dfMergedRNA[listFCOutCols].reindex(listOutputGeneOrder).values.astype(float))))
        handRNASeqHM = handAx.matshow(dfMergedRNA[listFCOutCols].reindex(listOutputGeneOrder),
                       vmin=-numMaxAbsFC, vmax=numMaxAbsFC,
                       cmap=plt.cm.PRGn, aspect='auto')

        handAx.set_xticks([])
        handAx.set_yticks([])
        for iGene in range(len(listOutputGeneOrder)):
            strENSG = listOutputGeneOrder[iGene]
            if dictENSGToHGNC[strENSG] == dictENSGToHGNC[strENSG]:
                strGeneOut = dictENSGToHGNC[strENSG]
            else:
                strGeneOut = strENSG

            handAx.text(-0.7, iGene,
                        strGeneOut,
                        ha='right', va='center',
                        fontsize=Plot.numFontSize*0.65,
                        fontstyle='italic')

            if iGene < len(listOutputGeneOrder)-1:
                handAx.axhline(y=iGene+0.5,
                               xmin=0.0, xmax=1.0,
                               color='0.5', lw=0.25)

        for iCond in range(len(listFCOutCols)):
            handAx.text(iCond-0.2, -0.5,
                        dictOutRNASeqCond[listFCOutCols[iCond]],
                        ha='left', va='bottom',
                        fontsize=Plot.numFontSize*0.7,
                        rotation=70)

            if iCond < len(listFCOutCols)-1:
                handAx.axvline(x=iCond+0.5,
                               ymin=0.0, ymax=1.0,
                               color='0.5', lw=0.25)


        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAx.spines[strAxLoc].set_linewidth(0.1)

        handCBarAx = handFig.add_axes(dictPanelLoc['HeatMap_cmap:RNA-seq'])
        handSigColorBar = handFig.colorbar(handRNASeqHM, cax=handCBarAx,
                                           orientation='horizontal')
        handSigColorBar.ax.tick_params(labelsize=Plot.numFontSize * 0.8)

        arrayTickLoc = plt.MaxNLocator(3)
        handSigColorBar.locator = arrayTickLoc
        handSigColorBar.update_ticks()

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handCBarAx.spines[strAxLoc].set_linewidth(0.1)

        structAxPos = handCBarAx.get_position()
        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     numHeatMapPanelXLabelPos, 'RNA-seq log$_{2}$FC',
                     ha='center', va='bottom',
                     fontsize=Plot.numFontSize*0.7)


        # # # # # # # # # #       #       #       #       #       #       #
        # # RNA-seq GO analysis
        # handAx = handFig.add_axes(dictPanelLoc['HeatMap:RNA-seq_GO'])
        # listOutputGeneOrderHGNC = [dictENSGToHGNC[strGene] for strGene in listOutputGeneOrder]
        # dfGeneAnnot = dfGeneOntology[listOutputSelGO].reindex(listOutputGeneOrderHGNC)
        # dfGeneAnnot['Transcription factor'] = pd.Series([strGene in listTFs for strGene in listOutputGeneOrder],
        #                               index=listOutputGeneOrderHGNC)
        # dfGeneAnnot['Epithelial gene'] = pd.Series([strGene in dictEpiMes['epi_genes'] for strGene in listOutputGeneOrderHGNC],
        #                               index=listOutputGeneOrderHGNC)
        # dfGeneAnnot['Mesenchymal gene'] = pd.Series([strGene in dictEpiMes['mes_genes'] for strGene in listOutputGeneOrderHGNC],
        #                               index=listOutputGeneOrderHGNC)
        #
        # listGeneAnnotCols = dfGeneAnnot.columns.tolist()
        # for iCol in range(len(listGeneAnnotCols)):
        #     if listGeneAnnotCols[iCol] in dictGOLabel.keys():
        #         listGeneAnnotCols[iCol] = listGeneAnnotCols[iCol] + '\n' + dictGOLabel[listGeneAnnotCols[iCol]]
        #
        # handAx.matshow(np.nan_to_num(dfGeneAnnot.values.astype(float)),
        #                cmap=plt.cm.Greys,
        #                vmin=0, vmax=1,
        #                aspect='auto'
        #                )
        # handAx.set_xticks([])
        # handAx.set_yticks([])
        # for iCol in range(len(listGeneAnnotCols)):
        #     handAx.text(iCol-0.3, -1,
        #                 listGeneAnnotCols[iCol],
        #                 ha='left', va='bottom',
        #                 fontsize=Plot.numFontSize * 0.7,
        #                 rotation=70
        #                 )
        #
        #     if iCol < len(listGeneAnnotCols)-1:
        #         handAx.axvline(x=iCol+0.5,
        #                        ymin=0.0, ymax=1.0,
        #                        color='0.5', lw=0.25)
        #
        # for iRow in range(np.shape(dfGeneAnnot)[0]):
        #     if iRow < np.shape(dfGeneAnnot)[0]-1:
        #         handAx.axhline(y=iRow+0.5,
        #                        xmin=0.0, xmax=1.0,
        #                        color='0.5', lw=0.25)
        #
        # # # # # # # # # #       #       #       #       #       #       #
        # # Hexbin landscape
        # handAx = handFig.add_axes(dictPanelLoc['Hexbin_Landscape'])
        #
        # _ = PlotFunc.es_ms_landscape(handAxIn=handAx,
        #                              handFigIn=handFig)
        #
        # # # # # # # # # #       #       #       #       #       #       #
        # # Histograms
        # listOutGenes = ['ZEB1', 'ESRP1',
        #                 'F11R', 'MAP7',
        #                 'CDS1', 'SH2D3A']
        # numOutGeneRow = 0
        # numOutGeneCol = 0
        # for iGene in range(len(listOutGenes)):
        #     strOutGene = listOutGenes[iGene]
        #     handAx = plt.subplot(arrayGridSpec[numOutGeneRow, numOutGeneCol])
        #     if numOutGeneCol == 0:
        #         flagLabelY = True
        #     else:
        #         flagLabelY = False
        #     if numOutGeneRow == 2:
        #         flagLabelX = True
        #     else:
        #         flagLabelX = False
        #
        #     _ = PlotFunc.tcga_sel_gene_hist(handAxIn=handAx,
        #                                     strGeneIn=strOutGene,
        #                                     flagLabelYAxis=flagLabelY,
        #                                     flagLabelXAxis=flagLabelX)
        #     numOutGeneCol += 1
        #     if numOutGeneCol >= 2:
        #         numOutGeneRow += 1
        #         numOutGeneCol=0


        pathOut = os.path.join(Plot.strOutputLoc, 'figure_5')
        for strFormat in Plot.listFileFormats:
            handFig.savefig(os.path.join(pathOut, 'Figure5.'+strFormat),
                            ext=strFormat, dpi=300)
        plt.close(handFig)

        return flagResult

    def figure_six(flagResult=False):

        tupleFigSize = (6.5, 9.5)

        numVolcanoHeight = 0.17
        numVolcanoWidth = 0.30

        numHexbinHeight = 0.33
        numHexbinWidth = numHexbinHeight * (tupleFigSize[1] / tupleFigSize[0])

        numHeatMapPanelHeight = 0.40
        numHeatMapPanelXLabelPos = 0.44
        numCMapHeight = 0.0075

        arrayGridSpec = matplotlib.gridspec.GridSpec(
            nrows=3, ncols=2,
            left=0.65, right=0.95,
            bottom=0.07, top=0.40,
            hspace=0.50, wspace=0.65
        )

        dictPanelLoc = {'Volcano:MDA-MB-231':[0.07, 0.90-numVolcanoHeight, numVolcanoWidth, numVolcanoHeight],
                        'Volcano:SUM159':[0.07, 0.50, numVolcanoWidth, numVolcanoHeight],
                        'HeatMap:RNA-seq':[0.60, 0.50, 0.07, numHeatMapPanelHeight],
                        'HeatMap_cmap:RNA-seq':[0.60, 0.485, 0.07, numCMapHeight],
                        'HeatMap:DNAme':[0.70, 0.50, 0.09, numHeatMapPanelHeight],
                        'HeatMap_cmap:DNAme':[0.71, 0.485, 0.07, numCMapHeight],
                        'HeatMap_cmap:DNAme_type':[0.71, 0.905, 0.07, numCMapHeight],
                        'HeatMap:ATAC-seq':[0.81, 0.50, 0.07, numHeatMapPanelHeight],
                        'HeatMap_cmap:ATAC-seq':[0.82, 0.485, 0.05, numCMapHeight],
                        'HeatMap:ChIP-seq':[0.90, 0.50, 0.07, numHeatMapPanelHeight],
                        'HeatMap_cmap:ChIP-seq':[0.91, 0.485, 0.05, numCMapHeight],
                        'Hexbin_Landscape':[0.07, 0.07, numHexbinWidth, numHexbinHeight]
                        }

        dictOutRNASeqCond = {'SUM159:logFC':'SUM159',
                             'MDAMB231:logFC':'MDA-MB-231'}

        dictENSGToHGNC = Process.dict_gtf_ensg_to_hgnc()
        dictHGNCToENSG = dict(zip(dictENSGToHGNC.values(), dictENSGToHGNC.keys()))

        dfMergedRNA = Process.diff_expr_data()
        listDataGenes = dfMergedRNA.index.tolist()
        setDataGenes = set(listDataGenes)

        listFCOutCols = ['MDAMB231:logFC', 'SUM159:logFC']

        for strGene in setDataGenes.difference(set(dictENSGToHGNC.keys())):
            dictENSGToHGNC[strGene] = strGene

        dfPublicChIPSeq = Process.chip_seq()

        dfLocalDNAMe = Process.dna_me_from_sep()
        dfLocalDNAMe.set_index('Probes', inplace=True)

        dfATACSeq = Process.atac_seq(flagPerformExtraction=True)

        listOutputGeneOrder = Process.output_genes()


        listChIPColsOut = ['Peak Score', 'Annotation', 'Gene Name']
        listOfListsMatchedChIP = []
        numMaxPeak = 0
        for strGene in listOutputGeneOrder:
            if dictENSGToHGNC[strGene] == dictENSGToHGNC[strGene]:
                strHGNC = dictENSGToHGNC[strGene]
                if strHGNC in dfPublicChIPSeq['Gene Name'].values.tolist():
                    sliceChIPForGene = dfPublicChIPSeq[dfPublicChIPSeq['Gene Name']==strHGNC]
                    listOfListsMatchedChIP.append(sliceChIPForGene[listChIPColsOut].copy(deep=True))
                else:
                    listOfListsMatchedChIP.append([np.nan])
            else:
                listOfListsMatchedChIP.append([np.nan])

        # listMatchedChIPLen = [len(listOfListsMatchedChIP[i]) for i in range(len(listOfListsMatchedChIP))]
        # set(listMatchedChIPLen)
        # Out[5]: {1, 2, 3, 5, 6}
        # ---> 30 min common product
        dfChIPForDisp = pd.DataFrame(data=np.zeros((len(listOutputGeneOrder), 30), dtype=float),
                                     index=listOutputGeneOrder)
        for iGene in range(len(listOutputGeneOrder)):
            if isinstance(listOfListsMatchedChIP[iGene], pd.DataFrame):
                numPeaks = len(listOfListsMatchedChIP[iGene])
                numWidthPerPeak = 30/numPeaks
                for iPeak in range(numPeaks):
                    dfChIPForDisp.iloc[iGene,np.int(iPeak*numWidthPerPeak):np.int((iPeak+1)*numWidthPerPeak)] = \
                        listOfListsMatchedChIP[iGene]['Peak Score'].iloc[iPeak]
            else:
                dfChIPForDisp.iloc[iGene, :] = np.nan

            listDNAMeColsOut = ['logFC', 'adj.P.Val', 'UCSC_RefGene_Name', 'UCSC_RefGene_Group']
            listOfListsMatchedDNAMe = []
            numMaxPeak = 0
            for strGene in listOutputGeneOrder:
                if dictENSGToHGNC[strGene] == dictENSGToHGNC[strGene]:
                    strHGNC = dictENSGToHGNC[strGene]
                    if strHGNC in dfLocalDNAMe['UCSC_RefGene_Name'].values.tolist():
                        sliceDNAMeForGene = dfLocalDNAMe[dfLocalDNAMe['UCSC_RefGene_Name'] == strHGNC]
                        listOfListsMatchedDNAMe.append(sliceDNAMeForGene[listDNAMeColsOut].copy(deep=True))
                    else:
                        listOfListsMatchedDNAMe.append([np.nan])
                else:
                    listOfListsMatchedDNAMe.append([np.nan])


        listDNAMeGroupOrders = ['TSS200', 'TSS1500', '5\'UTR', '1stExon', 'Body', 'ExonBnd', '3\'UTR']
        # listMatchedDNAMeLen = [len(listOfListsMatchedDNAMe[i]) for i in range(len(listOfListsMatchedDNAMe))]
        # set(listMatchedDNAMeLen)
        # Out[5]: {1, 2, 3, 5, 6}
        # ---> 30 min common product
        dfDNAMeForDisp = pd.DataFrame(data=np.zeros((len(listOutputGeneOrder), 500), dtype=float),
                                      index=listOutputGeneOrder)
        listProbeTypeFreq = []
        for iGene in range(len(listOutputGeneOrder)):
            if isinstance(listOfListsMatchedDNAMe[iGene], pd.DataFrame):
                dfForGene = listOfListsMatchedDNAMe[iGene]
                numProbes = np.shape(dfForGene)[0]
                numWidthPerPeak = 500 / numProbes
                listProbeOrder = []
                listProbeTypeFreqs = []
                for strGroup in listDNAMeGroupOrders:
                    listProbesFromGroup = dfForGene[dfForGene['UCSC_RefGene_Group']==strGroup].index.tolist()
                    listProbeOrder += listProbesFromGroup
                    listProbeTypeFreqs.append(len(listProbesFromGroup))
                listProbeTypeFreq.append(listProbeTypeFreqs)

                for iProbe in range(len(listProbeOrder)):
                    strProbe = listProbeOrder[iProbe]
                    dfDNAMeForDisp.iloc[iGene, np.int(iProbe*numWidthPerPeak):np.int((iProbe+1)*numWidthPerPeak)]=dfForGene['logFC'].loc[strProbe]
            else:
                dfDNAMeForDisp.iloc[iGene, :] = np.nan
                listProbeTypeFreq.append([0]*len(listDNAMeGroupOrders))


        dfATACSeqForDisp = pd.DataFrame(data=np.zeros((len(listOutputGeneOrder), 1),
                                                      dtype=float),
                                        index=listOutputGeneOrder)
        for iOutGene in range(len(listOutputGeneOrder)):
            strGene = listOutputGeneOrder[iOutGene]
            arrayGeneMatch = dfATACSeq['ENSG'].str.match(strGene)
            for iMatch in range(len(arrayGeneMatch)):
                if not arrayGeneMatch[iMatch] == arrayGeneMatch[iMatch]:
                    arrayGeneMatch[iMatch] = False

            arrayATACSeqIndex = np.where(arrayGeneMatch)[0]

            if len(arrayATACSeqIndex) == 1:
                dfATACSeqForDisp.loc[strGene] = np.log2(dfATACSeq['No_gRNA_count'].iloc[arrayATACSeqIndex[0]] -
                                                        dfATACSeq['All_gRNA_count'].iloc[arrayATACSeqIndex[0]])
            elif len(arrayATACSeqIndex) > 1:
                a=1
            else:
                dfATACSeqForDisp.loc[strGene] = np.nan

        handFig = plt.figure(figsize=tupleFigSize)

        # # # # # #       #       #       #       #       #       #       #
        # Volcano plots

        handAxMDAMB231 = handFig.add_axes(dictPanelLoc['Volcano:MDA-MB-231'])
        handAxSUM159 = handFig.add_axes(dictPanelLoc['Volcano:SUM159'])

        _ = PlotFunc.epi_mes_volcano(handAxInMDAMB231=handAxMDAMB231,
                                     handAxInSUM159=handAxSUM159)

        # # # # # #       #       #       #       #       #       #       #
        # RNA-seq logFC

        handAx = handFig.add_axes(dictPanelLoc['HeatMap:RNA-seq'])

        numMaxAbsFC = np.max(np.abs(
            np.ravel(dfMergedRNA[listFCOutCols].reindex(listOutputGeneOrder).values.astype(float))))
        handRNASeqHM = handAx.matshow(dfMergedRNA[listFCOutCols].reindex(listOutputGeneOrder),
                       vmin=-numMaxAbsFC, vmax=numMaxAbsFC,
                       cmap=plt.cm.PRGn, aspect='auto')

        handAx.set_xticks([])
        handAx.set_yticks([])
        for iGene in range(len(listOutputGeneOrder)):
            strENSG = listOutputGeneOrder[iGene]
            if dictENSGToHGNC[strENSG] == dictENSGToHGNC[strENSG]:
                strGeneOut = dictENSGToHGNC[strENSG]
            else:
                strGeneOut = strENSG

            handAx.text(-0.7, iGene,
                        strGeneOut,
                        ha='right', va='center',
                        fontsize=Plot.numFontSize*0.65,
                        fontstyle='italic')

            if iGene < len(listOutputGeneOrder)-1:
                handAx.axhline(y=iGene+0.5,
                               xmin=0.0, xmax=1.0,
                               color='0.5', lw=0.25)

        for iCond in range(len(listFCOutCols)):
            handAx.text(iCond-0.2, -0.5,
                        dictOutRNASeqCond[listFCOutCols[iCond]],
                        ha='left', va='bottom',
                        fontsize=Plot.numFontSize*0.7,
                        rotation=70)

            if iCond < len(listFCOutCols)-1:
                handAx.axvline(x=iCond+0.5,
                               ymin=0.0, ymax=1.0,
                               color='0.5', lw=0.25)


        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAx.spines[strAxLoc].set_linewidth(0.1)

        handCBarAx = handFig.add_axes(dictPanelLoc['HeatMap_cmap:RNA-seq'])
        handSigColorBar = handFig.colorbar(handRNASeqHM, cax=handCBarAx,
                                           orientation='horizontal')
        handSigColorBar.ax.tick_params(labelsize=Plot.numFontSize * 0.8)

        arrayTickLoc = plt.MaxNLocator(3)
        handSigColorBar.locator = arrayTickLoc
        handSigColorBar.update_ticks()

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handCBarAx.spines[strAxLoc].set_linewidth(0.1)

        structAxPos = handCBarAx.get_position()
        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     numHeatMapPanelXLabelPos, 'RNA-seq log$_{2}$FC',
                     ha='center', va='bottom',
                     fontsize=Plot.numFontSize*0.7)

        #       #       #       #       #       #       #       #       #
        # DNAme
        handAx = handFig.add_axes(dictPanelLoc['HeatMap:DNAme'])

        arrayColorNorm = matplotlib.colors.Normalize(vmin=0,
                                                     vmax=9)
        arrayColorsForMap = matplotlib.cm.ScalarMappable(norm=arrayColorNorm,
                                                         cmap=matplotlib.cm.tab10)


        CMapDNAme = copy.copy(matplotlib.cm.BrBG)
        CMapDNAme.set_bad('0.5', 1.)
        numMaxAbsFC = np.max(np.abs(
            np.ravel(np.nan_to_num(dfDNAMeForDisp.values.astype(float)))))
        handDNAmeHM = handAx.imshow(dfDNAMeForDisp.values.astype(float),
                       vmin=-numMaxAbsFC, vmax=numMaxAbsFC,
                       cmap=CMapDNAme, aspect='auto')

        handAx.set_xticks([])
        handAx.set_yticks([])

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handAx.spines[strAxLoc].set_linewidth(0.1)

        numRelGeneHeight = 1.0/len(listOutputGeneOrder)
        for iGene in range(len(listOutputGeneOrder)):
            if iGene < len(listOutputGeneOrder)-1:
                handAx.axhline(y=iGene+0.5,
                               xmin=0.0, xmax=1.0,
                               color='0.5', lw=1.5)

            numProbesForGene = np.sum(listProbeTypeFreq[iGene])
            if numProbesForGene > 0:
                numProbeHMWidth = 500./numProbesForGene
                for iProbe in range(numProbesForGene):
                    handAx.axvline(x=numProbeHMWidth*iProbe,
                                   ymin=1.0-((iGene+1)*numRelGeneHeight),
                                   ymax=1.0-(iGene*numRelGeneHeight),
                                   color='0.5', lw=0.25)

            numTotalProbes = np.sum(listProbeTypeFreq[iGene])

            if numTotalProbes > 0:
                numWidthPerProbe = 1.0 / numTotalProbes
                numProbeOutCount = 0
                for iProbeGroup in range(len(listProbeTypeFreq[iGene])):
                    if listProbeTypeFreq[iGene][iProbeGroup] > 0:
                        numProbesForType = listProbeTypeFreq[iGene][iProbeGroup]
                        if numProbeOutCount == 0:
                            handAx.axhline(y=iGene-0.33,
                                           xmin=0.0,
                                           xmax=((numProbeOutCount+numProbesForType)*numWidthPerProbe),
                                           color=arrayColorsForMap.to_rgba(iProbeGroup), lw=1.0,
                                           zorder=10)
                            numProbeOutCount += numProbesForType
                        else:
                            handAx.axhline(y=iGene-0.33,
                                           xmin=(numProbeOutCount*numWidthPerProbe)+0.01,
                                           xmax=((numProbeOutCount+numProbesForType)*numWidthPerProbe),
                                           color=arrayColorsForMap.to_rgba(iProbeGroup), lw=1.0,
                                           zorder=10)
                            numProbeOutCount += numProbesForType




        handCBarAx = handFig.add_axes(dictPanelLoc['HeatMap_cmap:DNAme_type'])
        arrayTypeForDisp = np.zeros((1, len(listDNAMeGroupOrders)), dtype=float)
        arrayTypeForDisp[0,:] = np.arange(len(listDNAMeGroupOrders), dtype=float)
        handCBarAx.matshow(arrayTypeForDisp,
                           cmap=plt.cm.tab10,
                           vmin=0,
                           vmax=9,
                           aspect='auto')
        handCBarAx.set_xticks([])
        handCBarAx.set_yticks([])

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handCBarAx.spines[strAxLoc].set_linewidth(0.1)

        structAxPos = handCBarAx.get_position()
        for iType in range(len(listDNAMeGroupOrders)):
            handFig.text(structAxPos.x0+((iType+0.25)/len(listDNAMeGroupOrders))*structAxPos.width,
                         structAxPos.y0 + 1.5*structAxPos.height,
                         listDNAMeGroupOrders[iType],
                         ha='left', va='bottom',
                         rotation=90,
                         fontsize=Plot.numFontSize*0.5)

        handCBarAx = handFig.add_axes(dictPanelLoc['HeatMap_cmap:DNAme'])
        handSigColorBar = handFig.colorbar(handDNAmeHM, cax=handCBarAx,
                                           orientation='horizontal')
        handSigColorBar.ax.tick_params(labelsize=Plot.numFontSize * 0.8)

        arrayTickLoc = plt.MaxNLocator(3)
        handSigColorBar.locator = arrayTickLoc
        handSigColorBar.update_ticks()

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handCBarAx.spines[strAxLoc].set_linewidth(0.1)

        structAxPos = handCBarAx.get_position()
        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     numHeatMapPanelXLabelPos, 'DNAme log$_{2}$FC',
                     ha='center', va='bottom',
                     fontsize=Plot.numFontSize*0.7)

        # # # # # #       #       #       #       #       #       #       #
        # ATAC-seq
        handAx = handFig.add_axes(dictPanelLoc['HeatMap:ATAC-seq'])

        numMaxPeak = np.max(np.nan_to_num(np.ravel(dfATACSeqForDisp.values.astype(float))))
        handATACseq = handAx.matshow(dfATACSeqForDisp.values.astype(float),
                       vmin=0, vmax=numMaxPeak,
                       cmap=plt.cm.plasma, aspect='auto')

        handAx.set_xticks([])
        handAx.set_yticks([])

        handCBarAx = handFig.add_axes(dictPanelLoc['HeatMap_cmap:ATAC-seq'])
        handSigColorBar = handFig.colorbar(handATACseq, cax=handCBarAx,
                                           orientation='horizontal')
        handSigColorBar.ax.tick_params(labelsize=Plot.numFontSize * 0.8)

        arrayTickLoc = plt.MaxNLocator(3)
        handSigColorBar.locator = arrayTickLoc
        handSigColorBar.update_ticks()

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handCBarAx.spines[strAxLoc].set_linewidth(0.1)

        structAxPos = handCBarAx.get_position()
        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     numHeatMapPanelXLabelPos, 'ATAC-seq\n'+r'log$_{2}$(${\Delta}$reads)',
                     ha='center', va='bottom',
                     fontsize=Plot.numFontSize*0.7)

        # # # # # #       #       #       #       #       #       #       #
        # ChIP-seq
        handAx = handFig.add_axes(dictPanelLoc['HeatMap:ChIP-seq'])

        numMaxPeak = np.max(np.nan_to_num(np.ravel(dfChIPForDisp.values.astype(float))))
        handChIPSeq = handAx.matshow(np.log2(dfChIPForDisp.values.astype(float)+1),
                       vmin=0, vmax=np.log2(numMaxPeak+1),
                       cmap=plt.cm.plasma, aspect='auto')

        handAx.set_xticks([])
        handAx.set_yticks([])

        handCBarAx = handFig.add_axes(dictPanelLoc['HeatMap_cmap:ChIP-seq'])
        handSigColorBar = handFig.colorbar(handChIPSeq, cax=handCBarAx,
                                           orientation='horizontal')
        handSigColorBar.ax.tick_params(labelsize=Plot.numFontSize * 0.8)

        arrayTickLoc = plt.MaxNLocator(3)
        handSigColorBar.locator = arrayTickLoc
        handSigColorBar.update_ticks()

        for strAxLoc in ['bottom', 'left', 'right', 'top']:
            handCBarAx.spines[strAxLoc].set_linewidth(0.1)

        structAxPos = handCBarAx.get_position()
        handFig.text(structAxPos.x0+0.5*structAxPos.width,
                     numHeatMapPanelXLabelPos, 'x $et al.$ ChIP-seq\n'+r'?log$_{2}$(${\Delta}$reads)?',
                     ha='center', va='bottom',
                     fontsize=Plot.numFontSize*0.7)

        # # # # # # # # #       #       #       #       #       #       #
        # Hexbin landscape
        handAx = handFig.add_axes(dictPanelLoc['Hexbin_Landscape'])

        _ = PlotFunc.es_ms_landscape(handAxIn=handAx,
                                     handFigIn=handFig)

        # # # # # # # # #       #       #       #       #       #       #
        # Histograms
        listOutGenes = ['ZEB1', 'ESRP1',
                        'F11R', 'MAP7',
                        'CDS1', 'SH2D3A']
        numOutGeneRow = 0
        numOutGeneCol = 0
        for iGene in range(len(listOutGenes)):
            strOutGene = listOutGenes[iGene]
            handAx = plt.subplot(arrayGridSpec[numOutGeneRow, numOutGeneCol])
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


        for strFormat in Plot.listFileFormats:
            handFig.savefig(os.path.join(Plot.strOutputLoc, 'Figure6.'+strFormat),
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

    def comb_tf_volcano(flagResult=False):

        dfData = Process.comb_tf_data()

        arrayLogFCData = dfData['Log2 Fold change'].values.astype(float)
        arrayFDR = dfData['FDR step up'].values.astype(float)
        numMinDFR = np.min(arrayFDR[arrayFDR > 0])
        arrayFDR[arrayFDR == 0.0] = numMinDFR
        arrayLogFDR = np.nan_to_num(-np.log10(arrayFDR))

        arrayIsGeneToLabel = arrayLogFDR > 150
        arrayGeneToLabelIndices = np.where(arrayIsGeneToLabel)[0]
        # listGenesToLabel = [dfData['gene_name'].tolist()[i] for i in arrayGeneToLabelIndices]

        numMaxAbsLogFC = np.max(np.abs(arrayLogFCData))
        arrayXRange = np.array([-1.05*numMaxAbsLogFC, 1.05*numMaxAbsLogFC], dtype=float)

        handFig = plt.figure(figsize=(5,5))

        handAx = handFig.add_axes([0.15, 0.15, 0.84, 0.80])

        handAx.scatter(arrayLogFCData,
                       arrayLogFDR,
                       s=5,
                       alpha=0.5,
                       lw=0.0,
                       color='0.5')

        arrayYLim = handAx.get_ylim()
        handAx.set_xlim(arrayXRange)

        handAx.set_xlabel('log$_{2}$(fold change)')
        handAx.set_ylabel('-log$_{10}$(FDR)')

        # hide the right and top spines
        handAx.spines['right'].set_visible(False)
        handAx.spines['top'].set_visible(False)

        listHandText = [handAx.text(
            arrayLogFCData[iGene],
            arrayLogFDR[iGene],
            dfData['gene_name'].iloc[iGene],
            fontsize=Plot.numFontSize * 0.70,
            style='italic',
            ha='center', va='center')
            for iGene in arrayGeneToLabelIndices]

        adjust_text(listHandText,
                    force_text=1.1,
                    force_points=1.1,
                    force_objects=1.1,
                    arrowprops=dict(arrowstyle='-',
                                    color='k', lw=0.5,
                                    connectionstyle="arc3",
                                    alpha=0.5),
                    #arrowProps=dict(arrowstyle=None)
                    )

        handFig.savefig(os.path.join(PathDir.pathOutFolder, 'CombTF_volcano.png'),
                        dpi=300)
        handFig.savefig(os.path.join(PathDir.pathOutFolder, 'CombTF_volcano.pdf'),
                        dpi=300)
        plt.close(handFig)

        a=1

        return flagResult

# dfDiffExpr = Process.diff_expr_data()
# listCommonGenes = Process.common_rna_genes()

# dfTCGARNA = Process.tcga_brca()
# dfTCGAScores = Process.tcga_scores(flagPerformExtraction=True)

# dfCCLERNA = Process.ccle_brca()
# dfCCLEScores = Process.ccle_scores(flagPerformExtraction=True)

# dictBrCaLineSubtype = Process.ccle_brca_subtypes()

# dfLocalScores = Process.local_scores()


_ = Plot.figure_five()
# _ = Plot.figure_six()

# _ = Plot.off_targets()

# _ = Plot.comb_tf_volcano()

# _ = PlotFunc.es_ms_landscape()



# _ = Plot.es_ms_landscape(
#     strDataLoc=PathDir.pathProcResults,
#     strFileName='ES_vs_MS_landscape',
#     listQuantFileNames=Process.listQuantFiles,
#     listLocalCellLines = Process.listLines,
#     listOfListsConds=Process.listOfListsConds,
#     dictOfDictXYOffsets=dictOfDictOffsets)


# _ = Output.merged_results()

# _ = Plot.heatmap()

# _ = Plot.diff_expr_genes_vs_tcga()

# _ = Plot.sig_meth_vs_rna()

# _ = Plot.ppi()


# _ = Process.tcga_scores()


# dfTCGA = TCGAFunctions.PanCancer.extract_mess_rna()