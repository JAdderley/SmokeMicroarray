def mappings_data_conformer(df, control, treatment):

    df = df[df['Treatment'].isin([control, treatment])]

    def aggregate_err_and_mean(G):
        ctrl = G[G['Treatment'] == control]
        test = G[G['Treatment'] == treatment]

        ctrl_mean_norm = ctrl['Normalized'].mean()
        ctrl['abs err'] = ctrl_mean_norm - ctrl['Normalized']
        ctrl_mean_error = ((ctrl_mean_norm - ctrl['Normalized']).abs() / ctrl_mean_norm).mean()

        test_mean_norm = test['Normalized'].mean()
        test['abs err'] = test_mean_norm - test['Normalized']
        test_mean_error = ((test_mean_norm - test['Normalized']).abs() / test_mean_norm).mean()

        first = G.iloc[0][['Target Uniprot ID', 'Target Name', 'Pan or P-Site']]
        first['Control Signal'] = ctrl_mean_norm
        first['Control Signal Error'] = ctrl_mean_error
        first['Test Signal'] = test_mean_norm
        first['Test Signal Error'] = test_mean_error

        return first

    mean_normalized = df.groupby(['Target Uniprot ID', 'Pan or P-Site'], group_keys=False)\
                        .apply(aggregate_err_and_mean)\
                        .reset_index(drop=True)

    mean_normalized = mean_normalized.rename({
        'Target Uniprot ID': 'Uniprot ID',
        'Target Name': 'Protein name',
        'Pan or P-Site': 'Phosphosite'
    })

    # **Filter out rows where any of the specified columns are NaN**
    mean_normalized = mean_normalized.dropna(subset=[
        'Control Signal',
        'Control Signal Error',
        'Test Signal',
        'Test Signal Error'
    ])

    return mean_normalized