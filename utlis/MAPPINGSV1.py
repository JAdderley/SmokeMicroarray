import pandas as pd
import numpy as np
import networkx as nx
import itertools
import random
from tqdm import tqdm
from itertools import chain


def process_microarray_data(ArrayData, NetConnections, network_output_filename, 
                            ErrorThreshold = 1, LowSignalCutOff = False, PanNormaliser = False, 
                            walk_number = 1000000, save_dir = 'MAPPINGS/', walklength = 1):

    pd.options.mode.chained_assignment = None  # default='warn'
    
    # ErrorThreshold = 1  # Lower = more stringent (0.01 - 10 (above 1 not recommended, 1 = 1:1 total error to change)
    # LowSignalCutOff = 750  # Change to desired value, older arrays 1000 recommended, newer 750 recommended
    # PanNormaliser = True  # Do you want the phospho signals normalised by pan-signals?
    # walk_number = 1000000 # How many randomtrails to generate (standard = 1 million)

    def DatasetOrganisation(Data, LowSignalCutOff, ErrorThreshold):

        if not isinstance(LowSignalCutOff, bool): 
            # Rename columns to easier names
            print("Read this")
            Data.columns = ['UniprotID', 'Protein', 'Phosphosite', 'URaw', 'UError', 'IRaw', 'IError']

            # Remove pan-specific signals
            Data = Data[~Data['Phosphosite'].str.contains("Pan", na=False)]

            # Remove Low intensity signals (If neither of the signals are above 1000 units)
            Data = Data[(Data['URaw'] >= LowSignalCutOff) | (Data['IRaw'] >= LowSignalCutOff)]

            # Identify signals where the total error is greater than set threshold

            Data['PercentCFC'] = Data['IRaw'] / Data['URaw'] * 100 - 100
            Data['InversePercentCFC'] = Data['URaw'] / Data['IRaw'] * 100 - 100

            conditions = [
                (Data['PercentCFC'] >= 0) & (Data['PercentCFC'] * ErrorThreshold > (Data['UError'] + Data['IError'])),
                (Data['PercentCFC'] >= 0) & (Data['PercentCFC'] * ErrorThreshold <= (Data['UError'] + Data['IError'])),
                (Data['PercentCFC'] < 0) & (Data['InversePercentCFC'] * ErrorThreshold > (Data['UError'] + Data['IError'])),
                (Data['PercentCFC'] < 0) & (Data['InversePercentCFC'] * ErrorThreshold <= (Data['UError'] + Data['IError']))]

            choices = ['Low', 'High', 'Low', 'High']
            Data['ErrorVerdict'] = np.select(conditions, choices)

            # Remove these high error to change ratio signals
            Data = Data[Data['ErrorVerdict'].str.contains("Low", case=False, na=False)]

        else:
            # Rename columns to easier names
            Data.columns = ['UniprotID', 'Protein', 'Phosphosite', 'URaw', 'IRaw']

            # Remove pan-specific signals
            Data = Data[~Data['Phosphosite'].str.contains("Pan", na=False)]

        # Remove special characters from Phosphosite column
        Data = Data.replace("\+", ",", regex=True)  # replaces + with , \ needed as + is a special character

        # Split dual + triple phosphosite signals into individual rows with same signal value (Toxo Data)
        DataMulti = Data[Data['Phosphosite'].str.contains(",", case=False, na=False)]  # Multi phosphosites only
        DataMulti['Phosphosite'] = DataMulti['Phosphosite'].str.split(",")  # puts phosphosites into a list using ","
        DataMulti = DataMulti.explode('Phosphosite').reset_index(drop=True)  # splits phosphosite to separate rows
        DataSingle = Data[~Data['Phosphosite'].str.contains(",", case=True, na=False)]  # Single phosphosite only
        Data = [DataMulti, DataSingle]
        Data = pd.concat(Data).reset_index(drop=True)  # join multi and single and reset index

        # Creates Reference for mapping to network
        Data['Concat'] = Data['UniprotID'].map(str) + '/' + Data['Phosphosite'].map(str)

        Data['Log2FoldChange'] = Data['IRaw'] - Data['URaw']

        # If multiple R/T/S values are available for a given phosphorylation site average values
        Data = Data.groupby(['Concat'], as_index=False).agg(
            {'Log2FoldChange': 'mean', 'UniprotID': 'first', 'Protein': 'first',
            'Phosphosite': 'first'})

        # Create INV for negative runs
        Data['INVLog2FoldChange'] = 1 / Data['Log2FoldChange']

        # Creates Reference for mapping to network
        Data['SubID_phosphosite'] = Data['UniprotID'].map(str) + '/' + Data['Phosphosite'].map(str)

        # Determine CDF value for each edge (used in trail termination chance)
        NegDFwithCDF = getFoldChangeCDF(Data[Data['Log2FoldChange'] < 0], Positive=False)
        PosDFwithCDF = getFoldChangeCDF(Data[Data['Log2FoldChange'] >= 0], Positive=True)

        return PosDFwithCDF, NegDFwithCDF
    
    def getFoldChangeCDF(FCdf, label='Log2FoldChange', Positive=True):
        stats_df = FCdf.groupby(label)[label].agg('count').pipe(pd.DataFrame).rename(columns={label: 'frequency'})

        # PDF
        stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])

        # CDF
        if Positive:
            stats_df['cdf'] = stats_df['pdf'].cumsum()

        else:
            stats_df['cdf'] = 1 - stats_df['pdf'].cumsum()

        stats_df = stats_df.reset_index()

        return FCdf.join(stats_df.set_index(label), on=label)

    def Merger(Network, PosDataCorrected, NegDataCorrected):

        PosMappedNetwork = pd.merge(PosDataCorrected, Network, how='right', on='SubID_phosphosite')  # merge network to data

        PosMappedNetwork.dropna(subset=['Log2FoldChange'], inplace=True)  # remove unlinked data
        PosMappedNetwork = PosMappedNetwork.drop(['SubID_phosphosite', 'UniprotID', 'Protein', 'Phosphosite_x'], axis=1)
        PosMappedNetwork = PosMappedNetwork[
            ['Kinase', 'Substrate', 'Phosphosite_y', 'Substrate_effect', 'Log2FoldChange',
            'Kinase_uniprot_ID', 'Substrate_uniprot_ID', 'cdf']]
        PosMappedNetwork = PosMappedNetwork.rename({'Phosphosite_y': 'Phosphosite'}, axis=1)

        # Remove duplicate rows
        PosMappedNetwork['Concat'] = PosMappedNetwork['Kinase_uniprot_ID'].map(str) + '/' + \
                                PosMappedNetwork['Substrate_uniprot_ID'].map(str) + '/' + \
                                PosMappedNetwork['Phosphosite'].map(str)  # make reference column

        PosMappedNetwork = PosMappedNetwork.drop_duplicates(subset=['Concat'], keep='first')  # remove duplicate, keep first
        PosMappedNetwork = PosMappedNetwork.drop(['Concat'], axis=1)

        PosMappedNetwork['INVLog2FoldChange'] = -PosMappedNetwork['Log2FoldChange']  # INV reference for negative walks

        
        ########################################################################################################################
        NegMappedNetwork = pd.merge(NegDataCorrected, Network, how='right', on='SubID_phosphosite')  # merge network to data

        NegMappedNetwork.dropna(subset=['Log2FoldChange'], inplace=True)  # remove unlinked data
        NegMappedNetwork = NegMappedNetwork.drop(['SubID_phosphosite', 'UniprotID', 'Protein', 'Phosphosite_x'], axis=1)
        NegMappedNetwork = NegMappedNetwork[
            ['Kinase', 'Substrate', 'Phosphosite_y', 'Substrate_effect', 'Log2FoldChange',
            'Kinase_uniprot_ID', 'Substrate_uniprot_ID', 'cdf']]
        NegMappedNetwork = NegMappedNetwork.rename({'Phosphosite_y': 'Phosphosite'}, axis=1)

        # Remove duplicate rows
        NegMappedNetwork['Concat'] = NegMappedNetwork['Kinase_uniprot_ID'].map(str) + '/' + \
                                NegMappedNetwork['Substrate_uniprot_ID'].map(str) + '/' + \
                                NegMappedNetwork['Phosphosite'].map(str)  # make reference column

        NegMappedNetwork = NegMappedNetwork.drop_duplicates(subset=['Concat'], keep='first')  # remove duplicate, keep first
        NegMappedNetwork = NegMappedNetwork.drop(['Concat'], axis=1)

        NegMappedNetwork['INVLog2FoldChange'] = -NegMappedNetwork['Log2FoldChange']  # INV reference for negative walks

        ########################################################################################################################
        # Identify the largest fold change for all multi-edges between two nodes for DiGraph setup
        PosNetwork = PosMappedNetwork.sort_values('Log2FoldChange', ascending=False)
        PosNetwork = PosNetwork[~PosNetwork.duplicated(subset=['Kinase', 'Substrate'], keep='first')]
        NegNetwork = NegMappedNetwork.sort_values('INVLog2FoldChange', ascending=False)
        NegNetwork = NegNetwork[~NegNetwork.duplicated(subset=['Kinase', 'Substrate'], keep='first')]

        return PosNetwork, NegNetwork

    def RandomTrail(g, nwalks, Control, Positive, walklength):

        pbar = tqdm(total=nwalks, desc='Running Random Trails', unit='walks')  # Progress Bar
        VTX = g.nodes()  # defines VTX as all of the nodes in the graph
        walks = list()
        j = 0

        while j <= nwalks:

            walk = list()
            visited = list()

            for step in itertools.count(start=1):  # for each step of the trail

                if step == 1:  # if this is the first step

                    node = random.sample(list(VTX), 1)[0]  # select a random node from the network

                else:  # if this is not the first step of a trail use last node from previous step
                    # Determine chance for trail termination based on edge Fold change value last used

                    if Control:
                        TerminationChance = 0.20

                    if not Control:
                        TerminationChance = 0.20 * (1 - g.get_edge_data(*selectededge)['cdf'])

                    if random.random() < TerminationChance:
                        break

                    else:
                        node = nextnode  # select last node after previous step

                walkelements = node

                adjacent = list(g.edges(node))  # paths out from the node
        #######################################################################################################################
                # assesses if the substrate effect code from the last step was negative, if so end the walk

                if step != 1:  # if not the first step of walk and last edge walked substrate effect code = '-' break

                    if g.get_edge_data(*selectededge)['Substrate_effect'] == "-":

                        walk.append(walkelements)
                        break
        #######################################################################################################################
                # Stop walks from rewalking steps already used and end walks if the pathway is now a dead end

                if len(visited) > 0:  # remove visited edges from the options

                    adjacent = [x for x in adjacent if x not in visited]

                if len(adjacent) == 0:  # if now a dead end after removing visited edges, end walk

                    walk.append(walkelements)
                    break
        #######################################################################################################################
                elif len(adjacent) == 1:  # else if there is only one edge option?

                    selectededge = adjacent[0]
                    nextnode = selectededge[1]  # Use the edge

        #######################################################################################################################
                else:  # else there is more than 1 edge to choose from, therefore a weighted decision is required

                    # adding weighting to the edges based on FoldChange
                    if not Control:  # If this is not a Basal/Control network trails analysis
                        prob = list()

                        # for each adjacent edge get the associated FoldChanges
                        for i in range(0, len(adjacent)):
                            if Positive:
                                prob.append(g.get_edge_data(node, adjacent[i][1])['Log2FoldChange'])
                            else:
                                prob.append(g.get_edge_data(node, adjacent[i][1])['INVLog2FoldChange'])

                        totalprob = sum(prob)
                        if totalprob == 0:  # if the edge probabilities = 0 ie two or more edges without positive weights
                            walk.append(walkelements)  # end walk

                        else:  # else the edge selection is weighted by the FoldChange values
                            selectededge = random.choices(adjacent, weights=prob, k=1)[0]
                            nextnode = selectededge[1]  # the second node in the edge (used in next step of walk)

                    else:  # If this is a Basal/Control network trails analysis then edge choice is random

                        selectededge = random.choices(adjacent, k=1)[0]
                        nextnode = selectededge[1]
        #######################################################################################################################

                visited.append(selectededge)

                walk.append(walkelements)  # defines the walk as the sum of the walk elements

            if len(walk) >= walklength:
                walks.append(walk)

                j = j + 1
                pbar.update(1)

        pbar.close()

        return walks
    
    def edgetally(datalist, SpecificNetwork):
        # splits trails into individual edges
        
        # splits trails list into edge list
        AllTrailEdgeUse = pd.DataFrame(chain.from_iterable(zip(x, x[1:]) for x in datalist))

        # Tally up edge usage and call all associated information to edges
        AllTrailEdgeUse.columns = ['Kinase', 'Substrate']
        TrailEdgeSummaryTable = pd.pivot_table(AllTrailEdgeUse, index=['Kinase', 'Substrate'], aggfunc='size')
        TrailEdgeSummary = pd.DataFrame(TrailEdgeSummaryTable).reset_index()
        TrailEdgeSummary.rename(inplace=True, columns={TrailEdgeSummary.columns[2]: 'TotalWalks'})
        TrailEdges = pd.merge(SpecificNetwork, TrailEdgeSummary, on=['Kinase', 'Substrate'])

        return TrailEdges
    
    def ChangePercent(df1, df2, Positive):
        # Get metrics
        # Merge Basal and Biological runs then determine Change(%) from basal network
        df3 = df1.merge(df2, on=['Kinase', 'Substrate', 'Phosphosite', 'Substrate_effect', 'INVLog2FoldChange',
                                'Log2FoldChange'], how='outer')

        df3['Change (%)'] = df3['TotalWalks_y'] / df3['TotalWalks_x'] * 100 - 100

        df3 = df3[df3['Change (%)'] > 5]

        # Inverse Negative Trail Change(%) for visualisation on a single network map
        if not Positive:
            df3['Change (%)'] = -df3['Change (%)']

        return df3

    def NetworkasDiGraph(PositiveNetwork, NegativeNetwork):

        PosDiGraph = nx.from_pandas_edgelist(PositiveNetwork, source='Kinase',
                                            target='Substrate', edge_attr=['Phosphosite', 'Substrate_effect',
                                                                            'Log2FoldChange', 'INVLog2FoldChange',
                                                                            'cdf'],
                                            create_using=nx.DiGraph())

        NegDiGraph = nx.from_pandas_edgelist(NegativeNetwork, source='Kinase',
                                            target='Substrate', edge_attr=['Phosphosite', 'Substrate_effect',
                                                                            'Log2FoldChange',
                                                                            'INVLog2FoldChange', 'cdf'],
                                            create_using=nx.DiGraph())

        PosNetworkFinal = nx.to_pandas_edgelist(PosDiGraph)
        NegNetworkFinal = nx.to_pandas_edgelist(NegDiGraph)

        PosNetworkFinal.rename(inplace=True, columns={'source': 'Kinase', 'target': 'Substrate'})
        NegNetworkFinal.rename(inplace=True, columns={'source': 'Kinase', 'target': 'Substrate'})

        # Setup of directional walks using positive or negative changes in source datasets
        PosEdges = PosDiGraph.copy()
        for edge in PosDiGraph.edges():
            if PosDiGraph.get_edge_data(*edge)["Log2FoldChange"] < 0:
                PosEdges.remove_edge(*edge)

        NegEdges = NegDiGraph.copy()
        for edge in NegDiGraph.edges():
            if NegDiGraph.get_edge_data(*edge)["INVLog2FoldChange"] < 0:
                NegEdges.remove_edge(*edge)

        return PosEdges, NegEdges, PosNetworkFinal, NegNetworkFinal
    
    
    # Organisation of Dataset and determination of quartiles of the data
    if PanNormaliser == True:
        print("DatasetOrganisationNormalise is missing see MAPPINGS")
        # PosDataCorrected, NegDataCorrected = DatasetOrganisationNormalise(ArrayData, LowSignalCutOff, ErrorThreshold)
    else:
        PosDataCorrected, NegDataCorrected = DatasetOrganisation(ArrayData, LowSignalCutOff, ErrorThreshold)

    # Merging of Network with Array dataset
    # Returns mapped dataset, Positive Network and Negative Network
    PosNetwork, NegNetwork = Merger(NetConnections, PosDataCorrected, NegDataCorrected)

    # Network as DiGraph and final edgelists
    PosEdges, NegEdges, PosNetworkFinal, NegNetworkFinal = NetworkasDiGraph(PosNetwork, NegNetwork)
    print("Networks Setup: Successful")

    #######################################################################################################################
    # RandomWalks and output processing

    print("Positive Trail Analysis")
    PosWalks = RandomTrail(PosEdges, nwalks = walk_number, Control=False, Positive=True, walklength=walklength)
    print("Positive Trail Analysis (Control Data)")
    CPosWalks = RandomTrail(PosEdges, nwalks = walk_number, Control=True, Positive=True, walklength=walklength)
    print("Negative Trail Analysis")
    NegWalks = RandomTrail(NegEdges, nwalks = walk_number, Control=False, Positive=False, walklength=walklength)
    print("Negative Trail Analysis (Control Data)")
    CNegWalks = RandomTrail(NegEdges, nwalks = walk_number, Control=True, Positive=False, walklength=walklength)

    print("Walks: Completed")

    PosData = edgetally(PosWalks, PosNetworkFinal)
    NegData = edgetally(NegWalks, NegNetworkFinal)

    CPosData = edgetally(CPosWalks, PosNetworkFinal)
    CNegData = edgetally(CNegWalks, NegNetworkFinal)

    print("Trails Step Splitting and Tallying: Completed")

    PosFinal = ChangePercent(CPosData, PosData, Positive=True)
    NegFinal = ChangePercent(CNegData, NegData, Positive=False)

    # Merge to final stage networks for visualisation
    NetworkFinal = PosFinal.merge(NegFinal, how='outer')

    # Output for Cytoscape
    NetworkFinal.to_csv(save_dir + network_output_filename, index=False)



    