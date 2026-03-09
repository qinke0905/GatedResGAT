import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr


def read_raw_data(file_path):
    df = pd.read_excel(file_path)
    df = df.dropna(subset=['Case ID', 'Eating Time', 'Onset Time', 'Food Category'])
    return df


def generate_encoding_dicts(df, desktop_path):
    districts = sorted(df['Current Address'].unique())
    district_dict = {district: i + 1 for i, district in enumerate(districts)}

    occupations = sorted(df['Patient Occupation'].unique())
    occupation_dict = {occupation: i + 1 for i, occupation in enumerate(occupations)}

    food_categories = sorted(df['Food Category'].unique())
    food_category_dict = {category: i + 1 for i, category in enumerate(food_categories)}

    processing_types = sorted(df['Processing/Packaging Method'].unique())
    processing_dict = {ptype: i + 1 for i, ptype in enumerate(processing_types)}

    eating_places = sorted(df['Eating Place'].unique())
    eating_place_dict = {place: i + 1 for i, place in enumerate(eating_places)}

    status_dict = {'Sporadic': 0, 'Clustered': 1}

    encoding_dicts = {
        'District Code': district_dict,
        'Occupation Code': occupation_dict,
        'Food Category Code': food_category_dict,
        'Processing Method Code': processing_dict,
        'Eating Place Code': eating_place_dict,
        'Case Type Code': status_dict
    }

    with pd.ExcelWriter(f'{desktop_path}/Encoding_Dictionary.xlsx') as writer:
        for sheet_name, encoding_dict in encoding_dicts.items():
            df_dict = pd.DataFrame(list(encoding_dict.items()), columns=['Name', 'Code'])
            df_dict.to_excel(writer, sheet_name=sheet_name, index=False)

    return encoding_dicts


def create_patient_node_table(df, encoding_dicts):
    node_df = df.copy()

    node_df['Medical Institution'] = node_df['Medical Institution'].map(encoding_dicts['District Code'])
    node_df['Patient Occupation'] = node_df['Patient Occupation'].map(encoding_dicts['Occupation Code'])
    node_df['Food Category'] = node_df['Food Category'].map(encoding_dicts['Food Category Code'])
    node_df['Processing/Packaging Method'] = node_df['Processing/Packaging Method'].map(encoding_dicts['Processing Method Code'])
    node_df['Eating Place'] = node_df['Eating Place'].map(encoding_dicts['Eating Place Code'])
    node_df['Current Address'] = node_df['Current Address'].map(encoding_dicts['District Code'])

    if 'Purchase Place' in node_df.columns:
        node_df = node_df.drop(columns=['Purchase Place'])

    node_df['Patient Gender'] = node_df['Patient Gender'].map({'Female': 0, 'Male': 1})

    for symptom in ['Nausea', 'Vomiting', 'Abdominal Pain']:
        node_df[symptom] = node_df[symptom].map({'No': 0, 'Yes': 1})

    node_df['Case Type'] = node_df['Case Type'].map(encoding_dicts['Case Type Code'])

    def parse_datetime(s):
        try:
            return pd.to_datetime(s)
        except:
            try:
                return pd.to_datetime(s, format='%Y-%m-%d %H:%M:%S')
            except:
                return pd.NaT

    node_df['Eating Time'] = node_df['Eating Time'].apply(parse_datetime)
    node_df['Onset Time'] = node_df['Onset Time'].apply(parse_datetime)
    node_df = node_df.dropna(subset=['Eating Time', 'Onset Time'])

    node_df['Incubation Period'] = (node_df['Onset Time'] - node_df['Eating Time']).dt.total_seconds() / 3600
    node_df.loc[node_df['Incubation Period'] < 0, 'Incubation Period'] = 0

    drop_cols = ['Food Name'] if 'Food Name' in node_df.columns else []
    if 'Purchase Place' in node_df.columns:
        drop_cols.append('Purchase Place')
    node_df = node_df.drop(columns=drop_cols)

    return node_df


def create_patient_edge_table(node_df):
    def calculate_gaussian_bandwidths(node_df):
        latency = node_df['Incubation Period'].values
        age = node_df['Age'].values
        latency_diff = np.abs(latency[:, None] - latency[None, :])
        age_diff = np.abs(age[:, None] - age[None, :])
        upper_tri = np.triu_indices_from(latency_diff, k=1)
        sigma_latency = np.std(latency_diff[upper_tri])
        sigma_age = np.std(age_diff[upper_tri])
        sigma_latency = max(sigma_latency, 1e-6)
        sigma_age = max(sigma_age, 1e-6)
        return sigma_latency, sigma_age

    sigma_latency, sigma_age = calculate_gaussian_bandwidths(node_df)
    time_window_hour = 12
    n = len(node_df)

    case_ids = node_df['Case ID'].values
    onset_time = node_df['Onset Time'].values.astype('datetime64[s]')
    latency = node_df['Incubation Period'].values
    district = node_df['Current Address'].values
    food_category = node_df['Food Category'].values
    process_type = node_df['Processing/Packaging Method'].values
    eating_place = node_df['Eating Place'].values
    symptoms = node_df[['Nausea', 'Vomiting', 'Abdominal Pain']].values
    gender = node_df['Patient Gender'].values
    age = node_df['Age'].values
    occupation = node_df['Patient Occupation'].values
    cluster_label = node_df['Case Type'].values

    i_idx, j_idx = np.triu_indices(n, k=1)

    full_onset_diff_hour = np.abs((onset_time[i_idx] - onset_time[j_idx]) / np.timedelta64(1, 'h'))
    full_latency_diff = np.abs(latency[i_idx] - latency[j_idx])
    full_time_factor1 = np.exp(-(full_latency_diff ** 2) / (2 * sigma_latency ** 2))
    full_time_factor2 = np.exp(-full_onset_diff_hour / time_window_hour)
    full_spatial_factor = (district[i_idx] == district[j_idx]).astype(float)
    full_category_sim = (food_category[i_idx] == food_category[j_idx]).astype(float)
    full_process_sim = (process_type[i_idx] == process_type[j_idx]).astype(float)
    full_eating_sim = (eating_place[i_idx] == eating_place[j_idx]).astype(float)
    full_sym_i = symptoms[i_idx]
    full_sym_j = symptoms[j_idx]
    full_common_sym = np.sum(full_sym_i & full_sym_j, axis=1)
    full_total_sym = np.sum(full_sym_i | full_sym_j, axis=1)
    full_gender_sim = (gender[i_idx] == gender[j_idx]).astype(float)
    full_age_diff = np.abs(age[i_idx] - age[j_idx])
    full_age_sim = np.exp(-(full_age_diff ** 2) / (2 * sigma_age ** 2))
    full_occupation_sim = (occupation[i_idx] == occupation[j_idx]).astype(float)
    full_same_cluster = (cluster_label[i_idx] == 1) & (cluster_label[j_idx] == 1)

    window_mask = full_onset_diff_hour <= time_window_hour
    i_filtered = i_idx[window_mask]
    j_filtered = j_idx[window_mask]

    sts = full_time_factor1[window_mask] * full_time_factor2[window_mask] * full_spatial_factor[window_mask]
    fes = (full_category_sim[window_mask] + full_process_sim[window_mask] + full_eating_sim[window_mask]) / 3.0
    common_sym = full_common_sym[window_mask]
    total_sym = full_total_sym[window_mask]
    ss = np.divide(common_sym, total_sym, where=total_sym != 0, out=np.zeros_like(common_sym, dtype=float))
    ds = (full_gender_sim[window_mask] + full_age_sim[window_mask] + full_occupation_sim[window_mask]) / 3.0

    full_sts = full_time_factor1 * full_time_factor2 * full_spatial_factor
    full_fes = (full_category_sim + full_process_sim + full_eating_sim) / 3.0
    full_ss = np.divide(full_common_sym, full_total_sym, where=full_total_sym != 0, out=np.zeros_like(full_common_sym, dtype=float))
    full_ds = (full_gender_sim + full_age_sim + full_occupation_sim) / 3.0

    def calculate_correlations(sim_array, label_array):
        corr, _ = pointbiserialr(label_array, sim_array)
        return abs(corr)

    corr_spatial = calculate_correlations(full_sts, full_same_cluster)
    corr_food = calculate_correlations(full_fes, full_same_cluster)
    corr_symptom = calculate_correlations(full_ss, full_same_cluster)
    corr_demographic = calculate_correlations(full_ds, full_same_cluster)

    total_corr = corr_spatial + corr_food + corr_symptom + corr_demographic
    w_spatial = corr_spatial / total_corr if total_corr != 0 else 0.25
    w_food = corr_food / total_corr if total_corr != 0 else 0.25
    w_symptom = corr_symptom / total_corr if total_corr != 0 else 0.25
    w_demographic = corr_demographic / total_corr if total_corr != 0 else 0.25

    total_correlation = (w_spatial * sts + w_food * fes + w_symptom * ss + w_demographic * ds)

    edges_df = pd.DataFrame({
        'Edge ID': [f'E{i+1:03d}' for i in range(len(i_filtered))],
        'Source Case ID': case_ids[i_filtered],
        'Target Case ID': case_ids[j_filtered],
        'Spatiotemporal Similarity (STS)': np.round(sts, 2),
        'Food Exposure Similarity (FES)': np.round(fes, 2),
        'Symptom Similarity (SS)': np.round(ss, 2),
        'Demographic Similarity (DS)': np.round(ds, 2),
        'Total Similarity': np.round(total_correlation, 2)
    })

    return edges_df


def main():
    desktop_path = '***Your saving path***'

    print("1. Reading raw data...")
    raw_data = read_raw_data(f'{desktop_path}/synthetic_foodborne_disease.xlsx')
    print(f"Raw data read successfully, total {len(raw_data)} cases")

    print("\n2. Generating encoding dictionaries...")
    encoding_dicts = generate_encoding_dicts(raw_data, desktop_path)

    print("\n3. Creating patient node feature table...")
    node_table = create_patient_node_table(raw_data, encoding_dicts)
    print(f"Node table created successfully, total {len(node_table)} nodes")

    print("\n4. Creating patient edge relation table...")
    edge_table = create_patient_edge_table(node_table)

    node_table.to_excel(f'{desktop_path}/Patient_Node_Feature_Table.xlsx', index=False)
    edge_table.to_excel(f'{desktop_path}/Patient_Edge_Relation_Table.xlsx', index=False)

    print("\nData processing completed!")


if __name__ == "__main__":
    main()