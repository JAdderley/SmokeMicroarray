import numpy as np
from scipy import stats
import plotly.express as px 
import matplotlib.pyplot as plt  
from matplotlib_venn import venn3
import pandas as pd     
from IPython.display import display_html


def custom_melt(data):
    # Rename the columns using the rename_columns function and melt the dataframe
    df = rename_columns(data).melt(id_vars=[('Index', 'Cat No.')])
    
    # Filter for only the rows where 'column' is 'Normalized' or 'Flag'
    df = df[df['column'].isin(['Normalized', 'Flag'])]
    
    # Rename the ('Index', 'Cat No.') column to 'antibody' for clarity
    df = df.rename(columns={('Index', 'Cat No.'): 'antibody'})
    
    # Add a 'replicate' column, which counts replicates per antibody, id, and column
    df['replicate'] = df.groupby(['antibody', 'id', 'column']).cumcount()
    
    # Pivot the table to get 'Normalized' and 'Flag' as separate columns
    df = df.pivot(index=['antibody', 'replicate', 'id'], columns='column', values='value').reset_index()
    
    return df


def rename_columns(df):
    new_cols = []
    
    # Initialize 'last' to store the most recent valid column name encountered
    last = ""
    
    # Loop through all column names in the dataframe
    for i in df.columns:
        # Check if the column name is not an unnamed column
        if "Unnamed:" not in i:
            # If valid, store it in 'last'
            last = i

        # Check if the column name contains 'Barcode: #', extract the part after 'Barcode: #'
        if 'Barcode: #' in last:
            last = last.split('Barcode: #')[1]
        # Check if the column name contains 'Barcode ', extract the part after 'Barcode '
        elif 'Barcode ' in last:
            last = last.split('Barcode ')[1]
        # Check if the column name contains 'Barcode: ', extract the part after 'Barcode: '
        elif 'Barcode: ' in last:
            last = last.split('Barcode: ')[1]
        # Check if the column name contains 'ID ', extract the part after 'ID '
        elif 'ID ' in last:
            last = last.split('ID ')[1]

        new_cols.append(last)

    # Convert the first four columns to string type (this may include 'Cat No.' column)
    df.iloc[:, :4] = df.iloc[:, :4].astype(str)
    
    # Remove leading and trailing spaces from values in the 4th column (index 3)
    df.iloc[:, 3] = df.iloc[:, 3].str.strip()

    # Create a MultiIndex object from the new columns and the first row of the dataframe.
    # The new MultiIndex will have 'id' and 'column' as its two levels.
    multi_index = pd.MultiIndex.from_tuples(list(zip(new_cols, df.iloc[0])), names=['id', 'column'])

    df = df.iloc[1:]
    df.columns = multi_index

    return df


def make_fc_antibody(G, exp):
    """
    Calculates the log2 fold change and p-value for a specific antibody between two groups.

    Parameters:
    - G: DataFrame containing data for one antibody from the base group
    - exp: DataFrame containing data from the experimental group

    Returns:
    - first: A pandas Series containing the antibody information and calculated statistics
    """

    # Extract data for the same antibody from the experimental group
    exp = exp[exp['antibody'] == G.iloc[0]['antibody']]

    # Get the 'Normalized' intensity values for the experimental group
    exp_vals = exp['Normalized']

    # Prepare a Series to store results by taking the first row of the experimental data
    first = exp.drop(columns=['replicate', 'Flag', 'Normalized']).iloc[0]

    # Get the 'Normalized' intensity values for the base group
    G_vals = G['Normalized']

    # Calculate the log2 fold change between experimental and base groups
    # Use the mean normalized values for the calculation
    first['log2fc'] = np.log2(exp_vals.mean() / G_vals.mean())

    # Perform an independent t-test (Welch's t-test) between experimental and base groups
    # equal_var=False does not assume equal population variances
    # [1] extracts the p-value from the returned tuple (t-statistic, p-value)
    first['pval'] = stats.ttest_ind(exp_vals, G_vals, equal_var=False)[1]

    # Store the sample sizes for each group
    first['N_base'] = len(exp_vals)
    first['N_exp'] = len(G_vals)

    return first


def make_fc(base, exp):
    """
    Applies the make_fc_antibody function to each antibody in the base group,
    comparing it to the experimental group, and compiles the results.

    Parameters:
    - base: DataFrame containing data for the base group
    - exp: DataFrame containing data for the experimental group

    Returns:
    - compared: A DataFrame containing comparison results for all antibodies
    """

    # Group the base DataFrame by 'antibody' and apply make_fc_antibody to each group
    # Pass the experimental group DataFrame as an additional argument
    compared = base.groupby('antibody').apply(make_fc_antibody, exp).reset_index(drop=True)

    # Select relevant columns for the final output
    compared = compared[['antibody', 'Target Name', 'Target Uniprot ID', 'Pan or P-Site',
                         'log2fc', 'pval', 'N_base', 'N_exp']]

    # Identify if the antibody is pan-specific by checking the 'Pan or P-Site' column
    compared['isPan'] = compared['Pan or P-Site'].str.lower().str.contains('pan')

    # Determine if the result is significant based on fold change and p-value thresholds
    compared['significant'] = (compared['log2fc'].abs() > 1) & (compared['pval'] < 0.05)

    return compared


def comparison_volcano(key_a, key_b, comparisons):
    # Retrieve the comparison data
    comp = comparisons.get((key_a, key_b))
    if comp is None:
        print(f"Comparison for {key_a} vs {key_b} not found.")
        return None  # Exit the function if the comparison is not found

    # Filter out pan-specific antibodies
    # comp = comp[~comp['isPan']]

    # Replace any p-values equal to zero to avoid issues with log scale
    comp['pval'] = comp['pval'].replace(0, 1e-300)

    # Compute -log10(p-value) for better visualization
    comp['-log10_pval'] = -np.log10(comp['pval'])

    # Create a new column to categorize points based on significance and fold change thresholds
    def categorize(row):
        if row['pval'] < 0.05 and row['log2fc'] > 1:
            return 'Upregulated'
        elif row['pval'] < 0.05 and row['log2fc'] < -1:
            return 'Downregulated'
        else:
            return 'Not Significant'

    comp['Expression'] = comp.apply(categorize, axis=1)

    # Create the scatter plot using Plotly Express
    fig = px.scatter(
        comp,
        x='log2fc',
        y='-log10_pval',
        hover_data=['Target Name', 'log2fc', 'pval'],
        color='Expression',
        color_discrete_map={
            'Upregulated': 'red',
            'Downregulated': 'blue',
            'Not Significant': 'grey'
        },
        title=f'Volcano Plot: {key_a} vs. {key_b}',
        labels={
            'log2fc': 'Log2 Fold Change',
            '-log10_pval': '-Log10 P-value'
        },
        template='plotly_white'
    )

    # Add threshold lines for significance and fold change
    fig.add_hline(
        y=-np.log10(0.05),
        line_dash="dash",
        line_color="grey",
        annotation_text="p = 0.05",
        annotation_position="bottom left"
    )
    fig.add_vline(
        x=1,
        line_dash="dash",
        line_color="grey",
        annotation_text="Log2FC = 1",
        annotation_position="top right"
    )
    fig.add_vline(
        x=-1,
        line_dash="dash",
        line_color="grey",
        annotation_text="Log2FC = -1",
        annotation_position="top left"
    )

    # Update layout for better aesthetics and adjust figure dimensions
    fig.update_layout(
        xaxis_title='Log2 Fold Change',
        yaxis_title='-Log10 P-value',
        legend_title='Expression',
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        width=800,  # Adjust the width as needed
        height=800  # Adjust the height as needed
    )

    # Annotate the top significant points
    # Get top upregulated genes
    top_upregulated = comp[(comp['Expression'] == 'Upregulated')].nlargest(5, '-log10_pval')
    # Get top downregulated genes
    top_downregulated = comp[(comp['Expression'] == 'Downregulated')].nlargest(5, '-log10_pval')

    # Function to add annotations
    def add_annotations(df, x_offset):
        for i, row in df.iterrows():
            fig.add_annotation(
                x=row['log2fc'],
                y=row['-log10_pval'],
                text=row['Target Name'],
                showarrow=True,
                arrowhead=1,
                ax=x_offset,
                ay=-40,
                font=dict(size=10)
            )

    # Annotate top upregulated genes
    add_annotations(top_upregulated, x_offset=-40)
    # Annotate top downregulated genes
    add_annotations(top_downregulated, x_offset=40)

    return fig


def get_significant(key_a, key_b, comparisons):
    # Retrieve the comparison DataFrame for the given group pair
    out_df = comparisons[(key_a, key_b)]

    # Filter for significant results
    out_df = out_df[out_df['significant']]

    # Create a 'Title' column for display
    out_df['Title'] = out_df['Target Name'] + ' ' + out_df['Pan or P-Site']

    # Select the columns of interest
    out_df = out_df[['Title', 'log2fc']]

    # Sort the DataFrame by 'log2fc' values
    out_df = out_df.sort_values('log2fc')

    # Reset the index to ensure it's unique and sequential
    out_df = out_df.reset_index(drop=True)

    # Set the index name for display purposes
    out_df.index.name = f"{key_a} vs. {key_b}"

    # Define the styling function
    def make_pretty(styler):
        # Format the 'log2fc' column to 2 decimal places
        styler = styler.format(precision=2)
        # Apply a color gradient to the 'log2fc' column
        styler = styler.background_gradient(subset=['log2fc'], cmap="coolwarm", vmin=-2, vmax=2)
        # Set table attributes for inline display
        styler = styler.set_table_attributes("style='display:inline'")
        return styler

    # Apply the styling and return the styled DataFrame
    return out_df.style.pipe(make_pretty)


def significance_filter_df(df, pval_cutoff, fc_abs_cutoff):
    # Filter out pan-specific antibodies (keep only specific ones)
    # df = df[~df['isPan']]
    # Create a 'Title' column by combining 'Target Name' and 'Pan or P-Site'
    df.loc[:, 'Title'] = df['Target Name'] + ' ' + df['Pan or P-Site']
    # Return the DataFrame filtered by significance criteria:
    # p-value less than cutoff and absolute log2 fold change greater than or equal to cutoff
    return df[(df['pval'] < pval_cutoff) & (df['log2fc'].abs() >= abs(fc_abs_cutoff))]


def get_overlap(comparisons, pval_cutoff, fc_abs_cutoff, names_sets, include=[set()], exclude=[set()]):
    # Calculate the union of excluded sets
    out_set = set.union(*[names_sets[i] for i in exclude], set())
    
    # Calculate the intersection of included sets
    in_set = set.intersection(*[names_sets[i] for i in include])
    
    # Determine the difference between included and excluded sets
    difference = in_set.difference(out_set)
    
    # Combine all keys from include and exclude lists
    all_keys = include + exclude
    
    # Prepare a dictionary to hold DataFrames for each comparison
    dfs = {}
    for key in all_keys:
        # Apply the significance filter to the comparison DataFrame
        df = significance_filter_df(comparisons[key], pval_cutoff = pval_cutoff, fc_abs_cutoff = fc_abs_cutoff ).copy()
        
        # Create a 'Title' column for identification
        df['Title'] = df['Target Name'] + ' ' + df['Pan or P-Site']
        
        # Filter the DataFrame to include only titles in the 'difference' set
        df = df[df['Title'].isin(difference)]
        
        # Set a multi-index with 'Title' and a unique identifier to prevent duplicates
        df = df.set_index(['Title', df.groupby('Title').cumcount()])
        
        # Store the 'log2fc' values in the dictionary with a formatted key
        dfs[" vs. ".join(key)] = df['log2fc']
    
    # Concatenate the Series into a DataFrame
    out_df = pd.concat(dfs, axis=1)
    
    # Sort the columns in reverse order for display
    out_df = out_df.reindex(sorted(out_df.columns, reverse=True), axis=1)
    
    # Define a styling function for the DataFrame
    def make_pretty(styler):
        styler = styler.format(precision=2)  # Format numbers to 2 decimal places
        styler = styler.background_gradient(cmap="coolwarm", vmin=-2, vmax=2)
        styler = styler.set_table_attributes("style='display:inline'")
        return styler
    
    # Apply the styling and return the styled DataFrame
    return out_df.style.pipe(make_pretty)


def makeVenn(key_a, key_b, comparisons, pval_cutoff=0.05, fc_abs_cutoff=1):
    # Retrieve comparison data for key_a and key_b
    fc1 = comparisons[key_a]
    fc2 = comparisons[key_b]

    # Filter out pan-specific antibodies
    # fc1 = fc1[~fc1['isPan']]
    # fc2 = fc2[~fc2['isPan']]

    # Apply the significance filters (p-value and fold-change cutoff)
    fc1_sel = fc1[(fc1['pval'] <= pval_cutoff) & (fc1['log2fc'].abs() >= abs(fc_abs_cutoff))].reset_index(drop=True)
    fc2_sel = fc2[(fc2['pval'] <= pval_cutoff) & (fc2['log2fc'].abs() >= abs(fc_abs_cutoff))].reset_index(drop=True)

    # Create 'Title' column for comparison
    fc1_sel.loc[:, 'Title'] = fc1_sel['Target Name'] + ' ' + fc1_sel['Pan or P-Site']
    fc2_sel.loc[:, 'Title'] = fc2_sel['Target Name'] + ' ' + fc2_sel['Pan or P-Site']

    # Define sets for Venn diagram (A, B, and overlap)
    A_names = set(fc1_sel['Title']).difference(set(fc2_sel['Title']))
    both_names = set(fc1_sel['Title']).intersection(set(fc2_sel['Title']))
    B_names = set(fc2_sel['Title']).difference(set(fc1_sel['Title']))

    # Subsets for A, B, and overlap
    A = fc1_sel[fc1_sel['Title'].isin(A_names)]
    B = fc2_sel[fc2_sel['Title'].isin(B_names)]
    both = fc1_sel.merge(fc2_sel, left_on='Title', right_on='Title')
    both = both[['Title', 'log2fc_x', 'log2fc_y']].rename(columns={'log2fc_x': 'log2fc_A', 'log2fc_y': 'log2fc_B'})

    # Handle duplicate antibodies by averaging log2 fold change
    A = A[['Title', 'log2fc']].groupby('Title').mean().sort_values('log2fc')
    B = B[['Title', 'log2fc']].groupby('Title').mean().sort_values('log2fc')
    both = both.groupby('Title').mean().sort_values(['log2fc_A', 'log2fc_B'])

    # Styling function for DataFrames
    def make_pretty(styler, title):
        styler.set_caption(title)
        # Use format instead of set_precision (set_precision is deprecated)
        styler = styler.format(precision=2)
        styler = styler.background_gradient(axis=None, vmin=-2, vmax=2, cmap="coolwarm")
        styler.set_table_attributes("style='display:inline'")
        return styler

    # Display the results in a Venn-like layout using HTML
    return display_html(
        A.style.pipe(make_pretty, f'{key_a[0]} vs. {key_a[1]}<br>N={len(A)}')._repr_html_() +
        both.style.pipe(make_pretty, f"Overlap <br>N={len(both)}")._repr_html_() +
        B.style.pipe(make_pretty, f'{key_b[0]} vs. {key_b[1]}<br>N={len(B)}')._repr_html_(),
        raw=True
    )
