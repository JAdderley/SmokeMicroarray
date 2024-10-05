from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

def barplot_GOEA_quartiles(dataset, hue, title='GO Enrichment Analysis'):
    # Define a dictionary that maps quartile names to colors
    colour_dict = {'>Q3': '#FF7F00',
                   '<Q1': '#20B2AA'}

    # Extract keys and values of the colour dictionary and store them in separate lists
    values = list(colour_dict.keys())
    colors = list(colour_dict.values())

    # Create a colormap using the colors from the colour dictionary
    cmap = ListedColormap(colors)

    # Calculate the number of bars to be plotted
    n_bars = len(dataset['term'].unique())

    # Calculate the height of the figure based on the number of bars to be plotted
    fig_height = 0.5 + (n_bars * 0.45)

    # Create a matplotlib figure and axis object with dynamically calculated size
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=300)

    # Create a bar plot using the Seaborn barplot function
    ax = sns.barplot(
        data=dataset,
        x='n_genes/n_go',  # Set the x-axis as the gene ratio column
        y='term',           # Set the y-axis as the GO term column
        hue=hue,            # Group the data by quartiles using the hue parameter
        palette=colour_dict,  # Use the colour dictionary to set the colours of the bars
        dodge=False,        # Align the bars of each hue
        edgecolor='black'   # Draw black lines around the bars
    )

    # Set the x-axis and y-axis labels using the set_xlabel and set_ylabel functions
    ax.set_xlabel("Gene Ratio", fontsize=20)
    ax.set_ylabel("GO Term", fontsize=20)

    # Set title
    ax.set_title(title, fontsize=20)
    ax.set_xlim(0, 1)  # Set the x-axis limit to 0-1
    
    ax.legend().remove()

    # Create a legend using the legend function
    legend = ax.legend(
        bbox_to_anchor=(0.62, 1),
        loc='upper left',
        title='Quartile Assessed',
        fontsize=20
    )
    legend.get_title().set_fontsize(20)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    output_dir = 'goea_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Clean up the title to make it a valid filename
    filename = re.sub(r'[\\/*?:"<>|]', "", title)  # Remove invalid characters
    filename = filename.replace(' ', '_')          # Replace spaces with underscores
    filename += "_GOEA_quartiles.png"                        # Append "GOEA" and file extension
    filename = os.path.join(output_dir, filename)
                            
    # Save the figure using the cleaned filename
    plt.savefig(filename, bbox_inches='tight')