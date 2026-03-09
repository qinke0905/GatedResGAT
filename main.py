import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GATConv
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score,
    f1_score, matthews_corrcoef,
    precision_recall_curve, auc as sk_auc
)
import os
from datetime import datetime
import time

def load_fold_data(train_df, val_indices, test_df, fold_idx, desktop_path):
    start_time = time.time()
    start_datetime = datetime.fromtimestamp(start_time).strftime("%H:%M:%S")

    feature_cols = [col for col in train_df.columns if col not in
                    ['Case ID', 'Eating Time', 'Onset Time', 'Case Type']]

    val_mask = np.zeros(len(train_df), dtype=bool)
    val_mask[val_indices] = True
    train_mask = ~val_mask

    train_sub_df = train_df[train_mask]
    val_df = train_df[val_mask]

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_sub_df[feature_cols].values)
    val_features = scaler.transform(val_df[feature_cols].values)
    test_features = scaler.transform(test_df[feature_cols].values)

    train_val_features = np.zeros((len(train_df), len(feature_cols)))
    train_val_features[train_mask] = train_features
    train_val_features[val_mask] = val_features
    train_val_features = torch.tensor(train_val_features, dtype=torch.float)

    test_node_features = torch.tensor(test_features, dtype=torch.float)

    train_val_labels = torch.tensor(train_df['Case Type'].values, dtype=torch.long)
    test_labels = torch.tensor(test_df['Case Type'].values, dtype=torch.long)

    id_to_idx = {id: i for i, id in enumerate(train_df['Case ID'])}
    test_id_to_idx = {id: i + len(train_df) for i, id in enumerate(test_df['Case ID'])}
    edge_df = pd.read_excel(f'{desktop_path}/Patient_Edge_Relation_Table.xlsx')
    edge_index = []
    edge_features = []

    for _, row in edge_df.iterrows():
        if row['Source Case ID'] in id_to_idx and row['Target Case ID'] in id_to_idx:
            u = id_to_idx[row['Source Case ID']]
            v = id_to_idx[row['Target Case ID']]
            edge_index.append([u, v])
            edge_index.append([v, u])
            edge_feat = [row['Spatiotemporal Similarity (STS)'], row['Food Exposure Similarity (FES)'],
                         row['Symptom Similarity (SS)'], row['Demographic Similarity (DS)']]
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0),
                                                                                                            dtype=torch.long)
    edge_features = torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.zeros((0, 4),
                                                                                                     dtype=torch.float)

    if edge_index.numel() > 0:
        max_idx = edge_index.max().item()
        if max_idx >= len(train_df):
            valid_mask = (edge_index[0] < len(train_df)) & (edge_index[1] < len(train_df))
            edge_index = edge_index[:, valid_mask]
            edge_features = edge_features[valid_mask]

    end_time = time.time()
    end_datetime = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
    load_time = end_time - start_time

    return {
        'train_val_features': train_val_features,
        'test_features': test_node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'train_val_labels': train_val_labels,
        'test_labels': test_labels,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'train_df': train_df,
        'test_df': test_df,
        'id_to_idx': id_to_idx,
        'test_id_to_idx': test_id_to_idx,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'fold': fold_idx,
        'timing': {
            'step': f'Fold {fold_idx + 1} - Data Loading',
            'timestamp': start_time,
            'start_time': start_datetime,
            'end_time': end_datetime,
            'duration': round(load_time, 2)
        }
    }


class GatedResGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.gat_layers = torch.nn.ModuleList()
        self.res_layers = torch.nn.ModuleList()
        self.gate_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()
        self.alpha_params = torch.nn.ParameterList()

        self.gat_layers.append(
            GATConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                edge_dim=4,
                heads=num_heads,
                dropout=dropout,
                add_self_loops=False,
                concat=True
            )
        )
        self.res_layers.append(torch.nn.Linear(input_dim, hidden_dim * num_heads))
        self.gate_layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_dim * num_heads * 2, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1),
                torch.nn.Sigmoid()
            )
        )
        self.norm_layers.append(torch.nn.LayerNorm(hidden_dim * num_heads))
        self.alpha_params.append(torch.nn.Parameter(torch.tensor(0.5)))

        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    in_channels=hidden_dim * num_heads,
                    out_channels=hidden_dim,
                    edge_dim=4,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                    concat=True
                )
            )
            self.res_layers.append(torch.nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads))
            self.gate_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim * num_heads * 2, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, 1),
                    torch.nn.Sigmoid()
                )
            )
            self.norm_layers.append(torch.nn.LayerNorm(hidden_dim * num_heads))
            self.alpha_params.append(torch.nn.Parameter(torch.tensor(0.5)))

        self.gat_layers.append(
            GATConv(
                in_channels=hidden_dim * num_heads,
                out_channels=output_dim,
                edge_dim=4,
                heads=num_heads,
                dropout=dropout,
                add_self_loops=False,
                concat=False
            )
        )
        self.res_layers.append(torch.nn.Linear(hidden_dim * num_heads, output_dim))
        self.gate_layers.append(
            torch.nn.Sequential(
                torch.nn.Linear(output_dim * 2, output_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(output_dim, 1),
                torch.nn.Sigmoid()
            )
        )
        self.norm_layers.append(torch.nn.LayerNorm(output_dim))
        self.alpha_params.append(torch.nn.Parameter(torch.tensor(0.5)))

        self.skip_res = torch.nn.Linear(input_dim, output_dim)
        self.beta = torch.nn.Parameter(torch.tensor(0.3))

        self.att_weights_list = []
        self.node_embedding = None

    def forward(self, x, edge_index, edge_attr):
        x0 = x
        x_current = x
        self.att_weights_list = []

        for layer_idx in range(self.num_layers):
            x_prev = x_current
            gat_layer = self.gat_layers[layer_idx]
            res_layer = self.res_layers[layer_idx]
            gate_layer = self.gate_layers[layer_idx]
            norm_layer = self.norm_layers[layer_idx]
            alpha = self.alpha_params[layer_idx]

            x_gat, (_, att_weights) = gat_layer(x_current, edge_index, edge_attr, return_attention_weights=True)
            self.att_weights_list.append((edge_index, att_weights))

            x_res = res_layer(x_prev)

            gate_input = torch.cat([x_gat, x_res], dim=1)
            g = gate_layer(gate_input)

            if layer_idx != self.num_layers - 1:
                x_fused = g * (x_gat + alpha * x_res) + (1 - g) * x_res
            else:
                skip_feat = self.skip_res(x0)
                combined = x_gat + alpha * x_res + self.beta * skip_feat
                x_fused = g * combined + (1 - g) * x_res

            x_fused = norm_layer(x_fused)
            if layer_idx != self.num_layers - 1:
                x_fused = F.relu6(x_fused)
                x_fused = F.dropout(x_fused, p=self.dropout, training=self.training)

            x_current = x_fused

        self.node_embedding = x_current

        return F.log_softmax(x_current, dim=1)


def train_single_fold(fold_data, fold_idx, save_dir, desktop_path):
    fold_timings = [fold_data['timing']]
    train_start_time = time.time()
    train_start_datetime = datetime.fromtimestamp(train_start_time).strftime("%H:%M:%S")

    input_dim = fold_data['train_val_features'].shape[1]
    model = GatedResGAT(
        input_dim=input_dim,
        hidden_dim=128,
        output_dim=2,
        num_heads=4,
        num_layers=4,
        dropout=0.1
    )

    train_labels = fold_data['train_val_labels'].numpy()[fold_data['train_mask']]
    n_pos = np.sum(train_labels == 1)
    n_neg = len(train_labels) - n_pos
    pos_weight = n_neg / n_pos if n_pos != 0 else 1.0
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float)
    criterion = torch.nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)

    best_val_pr_auc = -1
    patience = 200
    counter = 0
    best_model_state = None
    train_losses = []

    model.train()
    for epoch in range(2000):
        optimizer.zero_grad()
        out = model(fold_data['train_val_features'], fold_data['edge_index'], fold_data['edge_features'])
        loss = criterion(out[fold_data['train_mask']], fold_data['train_val_labels'][fold_data['train_mask']])
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_out = model(fold_data['train_val_features'], fold_data['edge_index'], fold_data['edge_features'])
            val_probs = torch.exp(val_out)[fold_data['val_mask'], 1].detach().numpy()
            val_labels = fold_data['train_val_labels'][fold_data['val_mask']].numpy()

            if len(np.unique(val_labels)) > 1:
                precision_curve, recall_curve, _ = precision_recall_curve(val_labels, val_probs)
                current_val_pr_auc = sk_auc(recall_curve, precision_curve)
            else:
                current_val_pr_auc = 0

        if current_val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = current_val_pr_auc
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(
                    f"Fold {fold_idx + 1}: GatedResGAT early stopping triggered at epoch {epoch + 1}, best val PR-AUC: {best_val_pr_auc:.4f}")
                break
        model.train()

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        val_out = model(fold_data['train_val_features'], fold_data['edge_index'], fold_data['edge_features'])
        val_probs = torch.exp(val_out)[fold_data['val_mask'], 1].detach().numpy()
        val_preds = val_out[fold_data['val_mask']].argmax(dim=1).numpy()
        val_labels = fold_data['train_val_labels'][fold_data['val_mask']].numpy()

        all_features = torch.cat([fold_data['train_val_features'], fold_data['test_features']], dim=0)
        test_edge_index = []
        test_edge_features = []
        edge_df = pd.read_excel(f'{desktop_path}/Patient_Edge_Relation_Table.xlsx')
        full_id_map = {**fold_data['id_to_idx'], **fold_data['test_id_to_idx']}

        for _, row in edge_df.iterrows():
            if row['Source Case ID'] in full_id_map and row['Target Case ID'] in full_id_map:
                u = full_id_map[row['Source Case ID']]
                v = full_id_map[row['Target Case ID']]
                test_edge_index.append([u, v])
                test_edge_index.append([v, u])
                edge_feat = [row['Spatiotemporal Similarity (STS)'], row['Food Exposure Similarity (FES)'],
                             row['Symptom Similarity (SS)'], row['Demographic Similarity (DS)']]
                test_edge_features.append(edge_feat)
                test_edge_features.append(edge_feat)

        test_edge_index = torch.tensor(test_edge_index,
                                       dtype=torch.long).t().contiguous() if test_edge_index else torch.zeros((2, 0),
                                                                                                              dtype=torch.long)
        test_edge_features = torch.tensor(test_edge_features, dtype=torch.float) if test_edge_features else torch.zeros(
            (0, 4), dtype=torch.float)

        test_out = model(all_features, test_edge_index, test_edge_features)
        test_probs = torch.exp(test_out)[len(fold_data['train_df']):, 1].detach().numpy()
        test_preds = test_out[len(fold_data['train_df']):].argmax(dim=1).numpy()
        test_labels = fold_data['test_labels'].numpy()

    def calculate_full_metrics(labels, probs, preds):
        metrics = {}
        if len(np.unique(labels)) > 1:
            precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
            metrics['pr_auc'] = sk_auc(recall_curve, precision_curve)
            metrics['auc'] = roc_auc_score(labels, probs)
        else:
            metrics['pr_auc'] = 0
            metrics['auc'] = 0

        metrics['f1'] = f1_score(labels, preds) if (np.sum(preds) > 0 and np.sum(labels) > 0) else 0
        metrics['recall'] = recall_score(labels, preds) if np.sum(labels) > 0 else 0
        metrics['precision'] = precision_score(labels, preds) if np.sum(preds) > 0 else 0
        metrics['mcc'] = matthews_corrcoef(labels, preds) if len(np.unique(labels)) > 1 else 0
        return metrics

    val_metrics = calculate_full_metrics(val_labels, val_probs, val_preds)
    test_metrics = calculate_full_metrics(test_labels, test_probs, test_preds)

    loss_curve_path = f'{save_dir}/Fold_{fold_idx + 1}_Training_Loss_Curve.png'
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='GatedResGAT Training Loss', color='#1f77b4', linewidth=1.5)
    plt.xlabel('Training Epochs')
    plt.ylabel('NLLLoss Value')
    plt.title(f'Fold {fold_idx + 1} GatedResGAT Training Loss Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    val_result_df = pd.DataFrame({
        'Case ID': fold_data['train_df'][fold_data['val_mask']]['Case ID'].values,
        'True Label': val_labels,
        'Cluster Probability': val_probs.round(4),
        'Predicted Label': val_preds
    })
    val_result_df.to_excel(f'{save_dir}/Fold_{fold_idx + 1}_Validation_Set_Prediction_Results.xlsx', index=False)

    test_result_df = pd.DataFrame({
        'Case ID': fold_data['test_df']['Case ID'].values,
        'True Label': test_labels,
        'Cluster Probability': test_probs.round(4),
        'Predicted Label': test_preds
    })
    test_result_df.to_excel(f'{save_dir}/Fold_{fold_idx + 1}_Test_Set_Prediction_Results.xlsx', index=False)

    eval_end_time = time.time()
    fold_timings.append({
        'step': f'Fold {fold_idx + 1} - Model Training and Evaluation',
        'start_time': train_start_datetime,
        'end_time': datetime.fromtimestamp(eval_end_time).strftime("%H:%M:%S"),
        'duration': round(eval_end_time - train_start_time, 2)
    })

    print(
        f"Fold {fold_idx + 1} completed | Val PR-AUC: {val_metrics['pr_auc']:.4f} | Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    return val_metrics, test_metrics, model, fold_timings


def run_5fold_cv(desktop_path):
    total_start_time = time.time()
    total_start_datetime = datetime.fromtimestamp(total_start_time).strftime("%H:%M:%S")
    all_timings = [{
        'step': 'Overall Process - Start',
        'start_time': total_start_datetime,
        'end_time': '-',
        'duration': 0.0
    }]

    node_df = pd.read_excel(f'{desktop_path}/Patient_Node_Feature_Table.xlsx')
    train_df, test_df = train_test_split(
        node_df,
        test_size=0.2,
        random_state=42,
        stratify=node_df['Case Type']
    )
    print(f"Data split completed: Train set {len(train_df)}, Test set {len(test_df)}")
    print(f"Train set class distribution:\n{train_df['Case Type'].value_counts(normalize=True).round(4)}")
    print(f"Test set class distribution:\n{test_df['Case Type'].value_counts(normalize=True).round(4)}")

    save_root = f'{desktop_path}/GatedResGAT_Paper_Reproduction_Results_{datetime.now().strftime("%Y%m%d%H%M")}'
    os.makedirs(save_root, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_metrics_list = []
    test_metrics_list = []
    all_fold_timings = []

    for fold_idx, (_, val_indices) in enumerate(skf.split(train_df, train_df['Case Type'])):
        print(f"\n===== Starting Fold {fold_idx + 1}/5 =====")
        fold_dir = f'{save_root}/Fold_{fold_idx + 1}'
        os.makedirs(fold_dir, exist_ok=True)

        fold_data = load_fold_data(train_df, val_indices, test_df, fold_idx, desktop_path)
        val_metrics, test_metrics, _, fold_timings = train_single_fold(fold_data, fold_idx, fold_dir, desktop_path)

        val_metrics_list.append(val_metrics)
        test_metrics_list.append(test_metrics)
        all_fold_timings.extend(fold_timings)

    total_end_time = time.time()
    total_duration = round(total_end_time - total_start_time, 2)
    all_timings.append({
        'step': 'Overall Process - End',
        'start_time': total_start_datetime,
        'end_time': datetime.fromtimestamp(total_end_time).strftime("%H:%M:%S"),
        'duration': total_duration
    })

    timing_df = pd.DataFrame(all_fold_timings).sort_values(by='start_time').reset_index(drop=True)
    timing_df.to_excel(f'{save_root}/Experiment_Time_Schedule.xlsx', index=False)

    metrics_keys = ['pr_auc', 'auc', 'f1', 'recall', 'precision', 'mcc']
    avg_val_metrics = {}
    avg_test_metrics = {}

    for key in metrics_keys:
        avg_val_metrics[key] = np.mean([m[key] for m in val_metrics_list])
        avg_val_metrics[f'{key}_std'] = np.std([m[key] for m in val_metrics_list])

        avg_test_metrics[key] = np.mean([m[key] for m in test_metrics_list])
        avg_test_metrics[f'{key}_std'] = np.std([m[key] for m in test_metrics_list])

    metrics_summary_df = pd.DataFrame({
        'Dataset': ['Validation'] * len(metrics_keys) + ['Test'] * len(metrics_keys),
        'Metric': metrics_keys + metrics_keys,
        'Mean': [avg_val_metrics[k] for k in metrics_keys] + [avg_test_metrics[k] for k in metrics_keys],
        'Std': [avg_val_metrics[f'{k}_std'] for k in metrics_keys] + [avg_test_metrics[f'{k}_std'] for k in
                                                                         metrics_keys]
    })
    metrics_summary_df['Mean ± Std'] = metrics_summary_df.apply(lambda x: f"{x['Mean']:.4f} ± {x['Std']:.4f}",
                                                                 axis=1)
    metrics_summary_df.to_excel(f'{save_root}/GatedResGAT_5-Fold_Average_Metrics_Summary.xlsx', index=False)

    print("\n" + "=" * 50)
    print("===== 5-Fold CV Validation Set Average Metrics =====")
    for key in metrics_keys:
        print(f"  {key}: {avg_val_metrics[key]:.4f} ± {avg_val_metrics[f'{key}_std']:.4f}")

    print("\n===== 5-Fold CV Test Set Average Metrics =====")
    for key in metrics_keys:
        print(f"  {key}: {avg_test_metrics[key]:.4f} ± {avg_test_metrics[f'{key}_std']:.4f}")
    print("=" * 50)
    print(f"\nAll results saved to: {save_root}")

    return avg_val_metrics, avg_test_metrics, timing_df, save_root


if __name__ == "__main__":
    desktop_path = '***Your saving path***'
    run_5fold_cv(desktop_path)