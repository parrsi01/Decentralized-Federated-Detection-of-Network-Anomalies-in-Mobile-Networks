import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from utils import simulate
from multiprocessing import Pool
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle


# from loaders import data_loader

# Ensure dataset folder exists
os.makedirs("datasets", exist_ok=True)

TOPOLOGIES = {
    "Ring": lambda n: {i: [(i-1)%n, (i+1)%n] if n > 1 else [] for i in range(n)},
    "Star": lambda n: {i: [0] if i != 0 else list(range(1, n)) for i in range(n)},
    "FullyConnected": lambda n: {i: [j for j in range(n) if j != i] for i in range(n)},
    "Random": lambda n: {i: random.sample([j for j in range(n) if j != i], 
                                    k=min(random.randint(1, 3), max(0, n - 1))) if n > 1 else []
                       for i in range(n)}
}


def prepare_data(dataset, alpha=0.5, num_agents=5, attack_types=None):
    import importlib.util
    import sys
    import os

    marl_path = os.path.join(os.path.dirname(__file__), '..', 'marl_xgb.py')
    spec = importlib.util.spec_from_file_location("marl_xgb", marl_path)
    marl_module = importlib.util.module_from_spec(spec)
    sys.modules["marl_xgb"] = marl_module
    spec.loader.exec_module(marl_module)

    result = marl_module.load_dataset(dataset, alpha=alpha, agents=num_agents, attack_filter=attack_types)
    # Fix: If the result is just a list of tuples (agent data), wrap it as expected
    if isinstance(result, tuple) and isinstance(result[0], list) and all(isinstance(x, tuple) and len(x) == 2 for x in result[0]):
        agents_data = result[0]
        X_test = result[1]
        y_test = result[2]
        feature_count = result[3]
        return agents_data, X_test, y_test, feature_count
    if isinstance(result, list) and all(isinstance(x, tuple) and len(x) == 2 for x in result):
        # Infer test data and feature count from one of the agents
        sample_X = result[0][0]
        sample_y = result[0][1]
        test_size = int(0.2 * len(sample_X))
        X_test = sample_X.iloc[:test_size]
        y_test = sample_y.iloc[:test_size]
        result = (sample_X, X_test, y_test, sample_X.shape[1])
    if not isinstance(result, tuple) or not isinstance(result[0], (pd.DataFrame, list)):
        raise ValueError(f"Expected load_dataset to return a tuple with a DataFrame or list as the first element, got: {type(result)} with first element {type(result[0]) if isinstance(result, tuple) else 'N/A'}")
    df = result[0]

    df = df[df['label'].isin(attack_types)]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df['label'])
    df['label'] = y_encoded

    X = df.drop(columns=['label'])
    # y = label_encoder.transform(df['label'])  # Ensure y and y_test are encoded

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)
    y_test = pd.Series(y_test).astype(int)

    # Dirichlet distribution for non-IID
    proportions = np.random.dirichlet([alpha]*num_agents)
    proportions = (proportions / proportions.sum() * len(X_train)).astype(int)

    idx = 0
    agents_data = []
    for size in proportions:
        if idx + size > len(X_train):
            size = len(X_train) - idx
        X_part = pd.DataFrame(X_train.iloc[idx:idx+size].copy())
        y_part = pd.Series(y_train[idx:idx+size].copy())
        idx += size

        # Apply SMOTE if necessary
        if len(np.unique(y_part)) > 1:
            sm = SMOTE()
            X_part, y_part = sm.fit_resample(X_part, y_part)
            X_part = pd.DataFrame(X_part)
            y_part = pd.Series(y_part)
        agents_data.append((X_part, y_part))

    return agents_data, X_test, y_test, X.shape[1]

def run_single_baseline(params):
    dataset, alpha, agents, attacks, loss_rate, topo_name = params
    label = "_".join([a.lower() for a in attacks])

    try:
        agents_data, X_test, y_test, fcount = prepare_data(dataset, alpha=alpha, num_agents=agents, attack_types=attacks)
        print(f"[DEBUG] Loaded agents_data type: {type(agents_data)}")
        print(f"[DEBUG] Sample agent X_train type: {type(agents_data[0][0])}")
    except Exception as e:
        print(f"❌ Failed to load {dataset} alpha={alpha} agents={agents} attacks={label}: {e}")
        return

    topo = TOPOLOGIES[topo_name](len(agents_data))
    detection_records = []
    traffic_records = []

    for _ in range(1):  # You can increase repeats here if needed
        for i, (X_train, y_train) in enumerate(agents_data):
            # Ensure y_train is properly encoded
            if isinstance(y_train, pd.DataFrame):
                if 'label' in y_train.columns:
                    y_train = y_train['label']
                else:
                    y_train = y_train.iloc[:,0]
            if isinstance(y_train, str):
                y_train = pd.Series([y_train])
            if isinstance(y_train, (pd.Series, np.ndarray)) and (y_train.dtype == 'O' or (len(y_train) > 0 and isinstance(y_train.iloc[0] if isinstance(y_train, pd.Series) else y_train[0], str))):
                y_train = pd.factorize(y_train)[0].astype(int)

            y_test_proc = pd.Series(y_test).astype(int)
            if isinstance(y_test_proc, pd.Series) and y_test_proc.dtype == object:
                y_test_proc = pd.factorize(y_test_proc)[0]

            if isinstance(X_train, str):
                raise ValueError(f"X_train is a string: {X_train}. Ensure load_dataset returns arrays, not file paths.")
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                print(f"⚠️ Skipping agent {i} due to single class in y_train for {dataset} alpha={alpha} agents={agents} attacks={label}")
                if 0 not in unique_classes:
                    print(f"⚠️ Skipping agent {i} due to class imbalance (expected class 0 missing) in y_train for {dataset} alpha={alpha} agents={agents} attacks={label}")
                detection_records.append({
                    "Dataset": dataset, "Topology": topo_name, "Alpha": alpha, "Agents": agents,
                    "AttackModel": label, "Agent": i,
                    "Accuracy": None, "Precision": None, "Recall": None, "F1": None
                })
                traffic_records.append({
                    "Dataset": dataset, "Topology": topo_name, "Alpha": alpha, "Agents": agents,
                    "AttackModel": label, "Agent": i,
                    "bandwidth": 0, "latency": 0, "packet_loss": 0
                })
                continue

            print(f"[DEBUG] Agent {i} y_train label distribution: {np.bincount(np.array(y_train))}")
            print(f"[DEBUG] y_test label distribution: {np.bincount(np.array(y_test_proc))}")

            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=1000)
            try:
                model.fit(X_train, y_train)
                print(f"✅ Model trained successfully for agent {i}")
            except ValueError as e:
                print(f"⚠️ Skipping agent {i} due to model error for {dataset} alpha={alpha} agents={agents} attacks={label}: {e}")
                detection_records.append({
                    "Dataset": dataset, "Topology": topo_name, "Alpha": alpha, "Agents": agents,
                    "AttackModel": label, "Agent": i,
                    "Accuracy": None, "Precision": None, "Recall": None, "F1": None
                })
                traffic_records.append({
                    "Dataset": dataset, "Topology": topo_name, "Alpha": alpha, "Agents": agents,
                    "AttackModel": label, "Agent": i,
                    "bandwidth": 0, "latency": 0, "packet_loss": 0
                })
                continue
            y_pred = model.predict(X_test)

            print(f"[DEBUG] Agent {i} y_pred label distribution: {np.bincount(y_pred)}")
            print(f"[DEBUG] Agent {i} - y_test unique: {np.unique(y_test_proc)}, y_pred unique: {np.unique(y_pred)}")
            print(f"[DEBUG] Agent {i} final y_test_proc unique: {np.unique(y_test_proc)}, y_pred unique: {np.unique(y_pred)}")

            acc = accuracy_score(y_test_proc, y_pred)
            prec = precision_score(y_test_proc, y_pred, zero_division=0)
            rec = recall_score(y_test_proc, y_pred, zero_division=0)
            f1 = f1_score(y_test_proc, y_pred, zero_division=0)

            detection_records.append({
                "Dataset": dataset, "Topology": topo_name, "Alpha": alpha, "Agents": agents,
                "AttackModel": label, "Agent": i,
                "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
            })

            traffic_list = simulate(loss_rate)
            traffic = traffic_list[0] if traffic_list else {'bandwidth': 0, 'latency': 0, 'packet_loss': 0}
            traffic_records.append({
                "Dataset": dataset, "Topology": topo_name, "Alpha": alpha, "Agents": agents,
                "AttackModel": label, "Agent": i,
                **traffic
            })

    os.makedirs("results/baseline", exist_ok=True)
    det_path = f"results/baseline/baseline_detection_{dataset}_{topo_name}_alpha{alpha}_agents{agents}_loss{int(loss_rate*100)}_{label}.csv"
    tra_path = f"results/baseline/baseline_traffic_{dataset}_{topo_name}_alpha{alpha}_agents{agents}_loss{int(loss_rate*100)}_{label}.csv"
    pd.DataFrame(detection_records).to_csv(det_path, index=False)
    pd.DataFrame(traffic_records).to_csv(tra_path, index=False)
    print(f"✅ Metrics saved: {det_path}")
    print(f"✅ Traffic saved: {tra_path}")



# Run selected naive baseline configurations only
def run_selected_naive_baseline(params_list):
    print("Running selected naive baseline tests...")
    with Pool(processes=8) as pool:
        list(tqdm(pool.imap_unordered(run_single_baseline, params_list), total=len(params_list)))
    print("Selected naive baseline tests completed.")


# New function as requested
def run_naive_baseline(params_list):
    run_selected_naive_baseline(params_list)


# Inline parameter sweep and main entry point
if __name__ == "__main__":
    DATASETS = ["cic", "ton", "unsw"]
    ALPHAS = [0.1, 0.3, 0.5, 1.0, 2.0]
    AGENTS = [5, 6, 7, 8, 9, 10]
    ATTACKS = [
        ["normal", "dos"],
        ["normal", "reconnaissance"],
        ["normal", "fuzzers"]
    ]
    LOSS_RATES = [0.0, 0.3, 0.5]
    TOPOLOGIES_LIST = ["Ring", "Star", "FullyConnected", "Random"]

    full_params = []
    for dataset in DATASETS:
        for alpha in ALPHAS:
            for agent in AGENTS:
                for attack in ATTACKS:
                    for loss in LOSS_RATES:
                        for topo in TOPOLOGIES_LIST:
                            full_params.append((dataset, alpha, agent, attack, loss, topo))

    run_naive_baseline(full_params)