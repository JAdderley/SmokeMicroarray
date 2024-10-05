import re
import os
import gzip
import urllib.request
import shutil
import time
import json
import zlib
import requests
import pandas as pd
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from matplotlib.colors import LinearSegmentedColormap
from goatools.godag_plot import plot_results
from IPython.display import Image
from xml.etree import ElementTree
from urllib.parse import urlparse, parse_qs, urlencode
from requests.adapters import HTTPAdapter, Retry
from goatools.obo_parser import GODag
from goatools.base import download_ncbi_associations
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

def goe_analysis(df_for_goa, uniprot_ids, quartiles = True, log2fc_col = None, significant = None):

    logging.info(f"Initiating GOEA Analysis")

    def download_and_decompress_gene2go():
        url = "ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"
        local_gz_file = "gene2go.gz"
        local_file = "gene2go"
        expected_file_size = 1000000000  # Set an approximate expected file size in bytes
        max_retries = 3  # Maximum number of retries for downloading

        def file_exists():
            return os.path.exists(local_gz_file) and os.path.exists(local_file)

        def download_file():
            try:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=local_gz_file) as t:
                    def reporthook(block_num, block_size, total_size):
                        if total_size not in (None, -1):
                            t.total = total_size
                        t.update(block_size)
                    urllib.request.urlretrieve(url, local_gz_file, reporthook)
                logging.info("File downloaded successfully")
                return True
            except Exception as e:
                logging.error(f"Error downloading file: {e}")
                return False

        def check_file_size():
            file_size = os.path.getsize(local_gz_file)
            if file_size < expected_file_size:
                logging.warning(f"Downloaded file size ({file_size}) is smaller than expected ({expected_file_size}).")
                return False
            return True

        def is_valid_gzip():
            try:
                with gzip.open(local_gz_file, 'rb') as f:
                    f.read(1)
                return True
            except Exception as e:
                logging.error(f"Invalid gzip file: {e}")
                return False

        def decompress_file():
            try:
                with gzip.open(local_gz_file, 'rb') as f_in:
                    with open(local_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                logging.info("File decompressed successfully")
                return True
            except Exception as e:
                logging.error(f"Error decompressing file: {e}")
                return False

        if file_exists():
            logging.info("Files already exist. Skipping download and decompression.")
            return True

        retries = 0
        while retries < max_retries:
            if download_file() and check_file_size() and is_valid_gzip():
                if decompress_file():
                    return True
            retries += 1
            logging.warning(f"Retrying... ({retries}/{max_retries})")
            if os.path.exists(local_gz_file):
                os.remove(local_gz_file)

        return False

    def download_go_basic_obo():
        url = "https://current.geneontology.org/ontology/go-basic.obo"
        local_file = "go.obo"
        expected_file_size = 30000000  # Approximate expected file size in bytes
        max_retries = 3  # Maximum number of retries for downloading

        def file_exists():
            return os.path.exists(local_file)

        def download_file():
            try:
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=local_file) as t:
                    def reporthook(block_num, block_size, total_size):
                        if total_size not in (None, -1):
                            t.total = total_size
                        t.update(block_size)
                    urllib.request.urlretrieve(url, local_file, reporthook)
                logging.info("File downloaded successfully")
                return True
            except Exception as e:
                logging.error(f"Error downloading file: {e}")
                return False

        def check_file_size():
            file_size = os.path.getsize(local_file)
            if file_size < expected_file_size:
                logging.warning(f"Downloaded file size ({file_size}) is smaller than expected ({expected_file_size}).")
                return False
            return True

        if file_exists():
            logging.info("go-basic.obo already exists. Skipping download.")
            return True

        retries = 0
        while retries < max_retries:
            if download_file() and check_file_size():
                return True
            retries += 1
            logging.warning(f"Retrying download... ({retries}/{max_retries})")
            if os.path.exists(local_file):
                os.remove(local_file)

        return False

    def convert_uniprots(list_of_ids):
        POLLING_INTERVAL = 5
        API_URL = "https://rest.uniprot.org"

        retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))

        def check_response(response):
            try:
                response.raise_for_status()
            except requests.HTTPError as e:
                logging.error(f"HTTP Error: {e}, Response: {response.text}")
                raise

        def get_next_link(headers):
            re_next_link = re.compile(r'<(.+)>; rel="next"')
            if "Link" in headers:
                match = re_next_link.match(headers["Link"])
                if match:
                    return match.group(1)
            return None

        def submit_id_mapping(from_db, to_db, ids):
            request = requests.post(
                f"{API_URL}/idmapping/run",
                data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
            )
            check_response(request)
            job_id = request.json().get("jobId")
            if not job_id:
                raise Exception("Failed to retrieve jobId from response.")
            return job_id

        def check_id_mapping_results_ready(job_id):
            while True:
                request = session.get(f"{API_URL}/idmapping/status/{job_id}")
                check_response(request)
                j = request.json()
                if "jobStatus" in j:
                    status = j["jobStatus"]
                    if status == "RUNNING":
                        logging.info(f"Retrying in {POLLING_INTERVAL}s")
                        time.sleep(POLLING_INTERVAL)
                    elif status == "NEW":
                        logging.info(f"Job status: NEW. Waiting for the job to start. Retrying in {POLLING_INTERVAL}s")
                        time.sleep(POLLING_INTERVAL)
                    elif status == "FAILED":
                        logging.error("Job failed.")
                        return False
                    elif status == "FINISHED":
                        return True
                    else:
                        raise Exception(f"Unexpected job status: {status}")
                else:
                    # When 'jobStatus' is not present, the job is ready
                    return True

        def get_batch(batch_response, file_format, compressed):
            batch_url = get_next_link(batch_response.headers)
            while batch_url:
                batch_response = session.get(batch_url)
                check_response(batch_response)
                yield decode_results(batch_response, file_format, compressed)
                batch_url = get_next_link(batch_response.headers)

        def combine_batches(all_results, batch_results, file_format):
            if file_format == "json":
                for key in ("results", "failedIds"):
                    if key in batch_results and batch_results[key]:
                        all_results[key] += batch_results[key]
            elif file_format == "tsv":
                return all_results + batch_results[1:]
            else:
                return all_results + batch_results
            return all_results

        def get_id_mapping_results_link(job_id):
            url = f"{API_URL}/idmapping/details/{job_id}"
            request = session.get(url)
            check_response(request)
            return request.json()["redirectURL"]

        def decode_results(response, file_format, compressed):
            if compressed:
                decompressed = zlib.decompress(response.content, 16 + zlib.MAX_WBITS)
                if file_format == "json":
                    j = json.loads(decompressed.decode("utf-8"))
                    return j
                elif file_format == "tsv":
                    return [line for line in decompressed.decode("utf-8").split("\n") if line]
                elif file_format == "xlsx":
                    return [decompressed]
                elif file_format == "xml":
                    return [decompressed.decode("utf-8")]
                else:
                    return decompressed.decode("utf-8")
            elif file_format == "json":
                return response.json()
            elif file_format == "tsv":
                return [line for line in response.text.split("\n") if line]
            elif file_format == "xlsx":
                return [response.content]
            elif file_format == "xml":
                return [response.text]
            return response.text

        def log_progress_batches(batch_index, size, total):
            n_fetched = min((batch_index + 1) * size, total)
            logging.info(f"Fetched: {n_fetched} / {total}")

        def get_id_mapping_results_search(url):
            parsed = urlparse(url)
            query = parse_qs(parsed.query)
            file_format = query["format"][0] if "format" in query else "json"
            if "size" in query:
                size = int(query["size"][0])
            else:
                size = 500
                query["size"] = size
            compressed = (
                query["compressed"][0].lower() == "true" if "compressed" in query else False
            )
            parsed = parsed._replace(query=urlencode(query, doseq=True))
            url = parsed.geturl()
            request = session.get(url)
            check_response(request)
            results = decode_results(request, file_format, compressed)
            total = int(request.headers.get("x-total-results", 0))
            log_progress_batches(0, size, total)
            for i, batch in enumerate(get_batch(request, file_format, compressed), 1):
                results = combine_batches(results, batch, file_format)
                log_progress_batches(i, size, total)
            if file_format == "xml":
                return merge_xml_results(results)
            return results

        # Start of the main function body

        # **Add this code to log the UniProt IDs being converted**
        logging.info(f"Converting UniProt IDs: {list_of_ids}")

        if not list_of_ids:
            logging.warning("No UniProt IDs provided for conversion.")
            return pd.DataFrame(columns=['GeneID', 'UniprotID'])  # Return empty DataFrame

        # Start the ID mapping process
        job_id = submit_id_mapping(
            from_db="UniProtKB_AC-ID", to_db="GeneID", ids=list_of_ids
        )

        if check_id_mapping_results_ready(job_id):
            link = get_id_mapping_results_link(job_id)
            results = get_id_mapping_results_search(link)
        else:
            logging.error("ID mapping job did not complete successfully.")
            return pd.DataFrame(columns=['GeneID', 'UniprotID'])  # Return empty DataFrame

        # **Add this check to handle cases where no results are returned**
        if not results.get('results'):
            logging.warning("No mapping results found.")
            return pd.DataFrame(columns=['GeneID', 'UniprotID'])

        # Proceed with processing the results
        gene_ids = []
        uniprot_ids = []

        for item in results.get('results', []):
            uniprot_ids.append(item['from'])
            gene_ids.append(item['to'])

        df = pd.DataFrame({'GeneID': gene_ids, 'UniprotID': uniprot_ids})

        # **Ensure 'UniprotID' column is of type string**
        df['UniprotID'] = df['UniprotID'].astype(str)

        return df

    def get_geneIDs_of_quartiles(df_for_goa, uniprot_ids, log2fc_col):

        gene_uniprot_ids_pop = convert_uniprots(list_of_ids=list(set(df_for_goa[uniprot_ids].astype('str'))))
        gene_uniprot_ids_pop['GeneID'] = gene_uniprot_ids_pop['GeneID'].astype('float64')

        df = pd.DataFrame(list(df_for_goa[uniprot_ids].astype('str')), columns=['UniprotID'])
        gene_uniprot_ids_pop = gene_uniprot_ids_pop.merge(df, on='UniprotID', how='right')
        geneids_pop = gene_uniprot_ids_pop['GeneID']

        gene_uniprot_ids_study_top = convert_uniprots(list(set(df_for_goa[uniprot_ids][df_for_goa[log2fc_col] >= df_for_goa[log2fc_col].quantile(q=0.75)].astype('str'))))
        df = pd.DataFrame(list(df_for_goa[uniprot_ids][df_for_goa[log2fc_col] >= df_for_goa[log2fc_col].quantile(q=0.75)].astype('str')), columns=['UniprotID'])
        gene_uniprot_ids_study_top = gene_uniprot_ids_study_top.merge(df, on='UniprotID', how='right')
        geneids_study_top = gene_uniprot_ids_study_top['GeneID'].astype('float64')

        gene_uniprot_ids_study_bottom = convert_uniprots(list(set(df_for_goa[uniprot_ids][df_for_goa[log2fc_col] <= df_for_goa[log2fc_col].quantile(q=0.25)].astype('str'))))
        df = pd.DataFrame(list(df_for_goa[uniprot_ids][df_for_goa[log2fc_col] <= df_for_goa[log2fc_col].quantile(q=0.25)].astype('str')), columns=['UniprotID'])
        gene_uniprot_ids_study_bottom = gene_uniprot_ids_study_bottom.merge(df, on='UniprotID', how='right')
        geneids_study_bottom = gene_uniprot_ids_study_bottom['GeneID'].astype('float64')

        return geneids_pop, geneids_study_top, geneids_study_bottom
    
    def get_geneID_of_sig(df_for_goa, uniprot_ids, significant):

        gene_uniprot_ids_pop = convert_uniprots(list_of_ids=list(set(df_for_goa[uniprot_ids].astype('str'))))
        gene_uniprot_ids_pop['GeneID'] = gene_uniprot_ids_pop['GeneID'].astype('float64')

        df = pd.DataFrame(list(df_for_goa[uniprot_ids].astype('str')), columns=['UniprotID'])
        gene_uniprot_ids_pop = gene_uniprot_ids_pop.merge(df, on='UniprotID', how='right')
        geneids_pop = gene_uniprot_ids_pop['GeneID']

        geneids_uniprot_ids_study = convert_uniprots(list(set(df_for_goa[uniprot_ids][df_for_goa[significant] == True].astype('str'))))
        df = pd.DataFrame(list(df_for_goa[uniprot_ids][df_for_goa[significant] == True].astype('str')), columns=['UniprotID'])
        geneids_uniprot_ids_study = geneids_uniprot_ids_study.merge(df, on='UniprotID', how='right')
        geneids_study = geneids_uniprot_ids_study['GeneID'].astype('float64')

        return geneids_pop, geneids_study

    def prep_goe_analysis(geneids_pop):

        fin_gene2go = download_ncbi_associations()
        obodag = GODag("go.obo")

        objanno = Gene2GoReader(fin_gene2go, taxids=[9606])
        ns2assoc = objanno.get_ns2assc()

        goeaobj = GOEnrichmentStudyNS(
            geneids_pop,              # List of genes possible in study
            ns2assoc,                 # Namespace2association
            obodag,                   # Ontologies
            propagate_counts=False,   # Propagate counts up GO hierarchy
            alpha=0.05,               # Significance level
            methods=["fdr_bh"]        # Correction method
        )

        GO_items = []

        temp = goeaobj.ns2objgoea['BP'].assoc
        for item in temp:
            GO_items += temp[item]

        temp = goeaobj.ns2objgoea['CC'].assoc
        for item in temp:
            GO_items += temp[item]

        temp = goeaobj.ns2objgoea['MF'].assoc
        for item in temp:
            GO_items += temp[item]

        return goeaobj, GO_items

    def go_it(test_genes, study):
        logging.info(f'input genes: {len(test_genes)}')

        goea_results_all = goeaobj.run_study(test_genes, prt=None)
        goea_results_sig = [r for r in goea_results_all if r.p_fdr_bh < 0.001]

        GO = pd.DataFrame(list(map(lambda x: [
            x.GO,
            x.goterm.name,
            x.goterm.namespace,
            x.p_uncorrected,
            x.p_fdr_bh,
            x.ratio_in_study[0],
            x.ratio_in_study[1],
            GO_items.count(x.GO),
            list(x.study_items)
        ], goea_results_sig)),
            columns=['GO', 'term', 'class', 'p', 'p_corr', 'n_genes', 'n_study', 'n_go', 'study_genes']
        )

        GO = GO[GO.n_genes > 1]
        GO['n_genes/n_go'] = GO.n_genes / GO.n_go
        GO['study_pop_assessed'] = study

        return GO, goea_results_sig

    
    if download_and_decompress_gene2go():
        logging.info("Successful download and decompression of gene2go.gz")
    else:
        logging.error("Failed to download or decompress gene2go.gz file")
    
    if download_go_basic_obo():
        logging.info("Successful downloaded go.obo")
    else:
        logging.error("Failed to download go.obo")
    
    if quartiles == True:
        geneids_pop, geneids_study_top, geneids_study_bottom = get_geneIDs_of_quartiles(df_for_goa, uniprot_ids, log2fc_col)
        goeaobj, GO_items = prep_goe_analysis(geneids_pop)

        goea_results_sig_top, goea_sig_top = go_it(test_genes=geneids_study_top, study='>Q3')
        goea_results_sig_bottom, goea_sig_bottom = go_it(test_genes=geneids_study_bottom, study='<Q1')

        goea_results = pd.concat([goea_results_sig_top, goea_results_sig_bottom])
        goea_results = goea_results.drop_duplicates(subset='GO', keep=False)

        logging.info("GOEA Analysis completed.")
        
        return goea_results, goea_sig_top, goea_sig_bottom
    
    else:
        geneids_pop, geneids_study = get_geneID_of_sig(df_for_goa, uniprot_ids, significant)
        goeaobj, GO_items = prep_goe_analysis(geneids_pop)

        goea_results, goea_sig_all = go_it(test_genes=geneids_study, study='Significant Signals')

        logging.info("GOEA Analysis completed.")

        return goea_results, goea_sig_all

# Function to perform GO analysis and save results
def perform_go_analysis(final_df, comparisons, output_dir, uniprot_ids, quartiles = True, log2fc_col = None, significant = None):

    goea_results = {}
    goea_results_sig = {}
    goea_results_sig_top = {}
    goea_results_sig_bottom = {}

    if quartiles == True:

        for comparison in comparisons:
            comparison_df = final_df[final_df['Comparison'] == comparison]
            goea_results[comparison], goea_results_sig_top[comparison], goea_results_sig_bottom[comparison] = goe_analysis(
                df_for_goa=comparison_df,
                uniprot_ids=uniprot_ids,
                log2fc_col=log2fc_col
            )

            # Save the GOEA results to a CSV file
            filename = os.path.join(output_dir, comparison.replace(' ', '_') + '_goea_results.csv')
            goea_results[comparison].to_csv(filename, index=False)

            # Plot and save the results for significant GO terms
            filename_top = os.path.join(output_dir, comparison.replace(' ', '_') + '_goea_network_top')
            plot_results(filename_top + "{NS}.png", goea_results_sig_top[comparison])

            filename_bottom = os.path.join(output_dir, comparison.replace(' ', '_') + '_goea_network_bottom')
            plot_results(filename_bottom + "{NS}.png", goea_results_sig_bottom[comparison])
            
        return goea_results, goea_results_sig_top, goea_results_sig_bottom
    
    else:
        for comparison in comparisons:
            comparison_df = final_df[final_df['Comparison'] == comparison]
            goea_results[comparison], goea_results_sig[comparison] = goe_analysis(
                df_for_goa=comparison_df,
                uniprot_ids=uniprot_ids,
                log2fc_col=log2fc_col,
                significant=significant
            )

            # Save the GOEA results to a CSV file
            filename = os.path.join(output_dir, comparison.replace(' ', '_') + '_goea_results.csv')
            goea_results[comparison].to_csv(filename, index=False)

            # Plot and save the results for significant GO terms
            filename_top = os.path.join(output_dir, comparison.replace(' ', '_') + '_goea_network_all')
            plot_results(filename_top + "{NS}.png", goea_results_sig[comparison])

        return goea_results, goea_results_sig

# Function to merge GOEA results and create heatmap
def merge_and_plot_heatmap(goea_results, comparisons, output_dir, quartiles = True):
    # Merge dataframes on 'GO', 'term', and 'class' columns
    merger = pd.DataFrame()
    for i, comparison in enumerate(comparisons):
        if i == 0:
            merger = goea_results[comparison][['GO', 'term', 'class', 'study_pop_assessed']]
            merger = merger.rename(columns={'study_pop_assessed': comparison})
        else:
            merger = pd.merge(merger, goea_results[comparison][['GO', 'term', 'class', 'study_pop_assessed']],
                              on=['GO', 'term', 'class'], how='outer')
            merger = merger.rename(columns={'study_pop_assessed': comparison})
    
    if quartiles == True:
        # Map values to -1, 0, 1
        mapping = {'>Q3': 1, '<Q1': -1, np.nan: 0}
        for comparison in comparisons:
            merger[comparison] = merger[comparison].map(mapping)
    else:
        mapping = {'Significant': 1, np.nan: 0}
        for comparison in comparisons:
            merger[comparison] = merger[comparison].map(mapping)
    
    # Define custom colors
    colour_dict = ['#22b2a6', '#D3D3D3', '#FF7F00']
    cmap = LinearSegmentedColormap.from_list('Custom', colour_dict, len(colour_dict))
    
    # Specify the categories to plot separately
    categories = ['biological_process', 'molecular_function', 'cellular_component']
    
    for category in categories:
        # Filter the merger dataframe for the current category
        filtered_merger = merger[merger['class'] == category]
        
        if filtered_merger.empty:
            continue  # Skip if there is no data for the category
        
        # Create heatmap
        cluster_grid = sns.clustermap(filtered_merger[comparisons],
                                      yticklabels=filtered_merger['term'],
                                      cmap=cmap,
                                      col_cluster=False,
                                      square=True,
                                      linewidths=0.2,
                                      linecolor='black',
                                      figsize=(8, 12),
                                      cbar_pos=[0, 0, 0, 0],
                                      dendrogram_ratio=(0.0001, 0.0001))
        
        # Move x-axis labels to the top
        cluster_grid.ax_heatmap.xaxis.set_ticks_position('top')
        cluster_grid.ax_heatmap.xaxis.set_label_position('top')
        
        # Set x-axis labels to start at the tick location
        plt.setp(cluster_grid.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha='left')
        
        # Save the heatmap as an SVG file
        heatmap_file = os.path.join(output_dir, f'goea_heatmap_{category}.svg')
        cluster_grid.savefig(heatmap_file, format='svg')
        plt.close()