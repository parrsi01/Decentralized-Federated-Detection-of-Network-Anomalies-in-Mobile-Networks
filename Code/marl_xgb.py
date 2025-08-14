# --- BASELINE ONLY FLAG ---
# Set to True to run only baseline (non-trust, simple averaging, no trust updates, separate logging)
BASELINE_ONLY = False  # Set to True to run only the non-scaled averaging baseline
#
# Author: Simon Parris
# Title: Decentralized and Hierarchical Federated Learning for Anomaly Detection
#
# --- Ensure baselines directory exists early ---
import os
os.makedirs("baselines", exist_ok=True)
import csv
import multiprocessing
num_cpus = multiprocessing.cpu_count()
pool_size = max(1, num_cpus - 1)
# from joblib import Parallel, delayed  # REMOVE if not used elsewhere
import numpy as np, pandas as pd, xgboost as xgb, random, os, matplotlib.pyplot as plt, seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import KMeansSMOTE

# --- Failed runs log for parameter study ---
failed_runs_log = []


#
# Global parameters for experiments
ALPHAS = [0.1, 0.3, 0.5, 1.0, 2.0]           # Dirichlet alpha values for non-IIDness
AGENT_COUNTS = list(range(5, 11))           # Number of agents/clients per experiment (5 to 10)
DATASETS = ["unsw", "cic", "ton"]           # Datasets to use
REPEATS = 1                                 # Number of experiment repeats per configuration

# Flag to enable/disable plot generation
ENABLE_PLOTS = False

# Attack model sets for parameter study
ATTACK_MODELS = [
    ['Normal', 'DoS', 'Reconnaissance'],
    ['Normal', 'Exploits'],
    ['Normal', 'Fuzzers']
]

# Cleans the DataFrame by removing NaNs and highly correlated features.
def clean_df(df):
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.select_dtypes(include=[np.number])
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    return df.drop(columns=to_drop)

# Automatically detects the label column for classification.
def find_label_column(df):
    stripped_cols = [col.strip() for col in df.columns]
    df.columns = stripped_cols
    candidates = [col for col in stripped_cols if "label" in col.lower() or "class" in col.lower() or "type" in col.lower()]
    if not candidates:
        for col in stripped_cols:
            if col.lower() == "label":
                return col
        # Extra: If "Label" exists, default to it
        if "Label" in df.columns:
            return "Label"
        raise ValueError(f"No label column found. Available columns: {stripped_cols}")
    return candidates[0]

# Splits data across agents using Dirichlet distribution for non-IID simulation.
def partition_dirichlet(X, y, agents, alpha, fallback_stratified=False):
    idxs, labels = [[] for _ in range(agents)], np.unique(y)
    for lbl in labels:
        idx = np.where(y == lbl)[0]
        props = np.random.dirichlet([alpha] * agents)
        splits = np.split(idx, (np.cumsum(props)*len(idx)).astype(int)[:-1])
        for i, part in enumerate(splits): idxs[i].extend(part)
    # If fallback_stratified is True, check if all agents have at least 2 classes. If not, fallback.
    if fallback_stratified:
        valid = [len(np.unique(y[i])) > 1 for i in idxs]
        if not all(valid):
            return None
    return idxs

# Stratified partitioning: ensures each agent has at least two classes (if possible)
def partition_stratified_min_classes(X, y, agents):
    class_indices = {cls: np.where(y == cls)[0].tolist() for cls in np.unique(y)}
    agent_bins = [[] for _ in range(agents)]
    for cls, idxs in class_indices.items():
        random.shuffle(idxs)
        for i, idx in enumerate(idxs):
            agent_bins[i % agents].append(idx)
    agent_data = [(X[bin], y[bin]) for bin in agent_bins]
    filtered = [(x, y_) for x, y_ in agent_data if len(np.unique(y_)) > 1]
    return filtered

# Loads and preprocesses data, splits into training/testing, applies SMOTE, and partitions across agents.
def load_from_split(X, y, test_ratio, alpha, agents):
    attempts = 0
    while attempts < 10:
        test_size = int(test_ratio * len(X))
        X_train, y_train = X[:-test_size], y[:-test_size]
        X_test, y_test = X[-test_size:], y[-test_size:]
        if len(np.unique(y_train)) < 2:
            attempts += 1
            continue
        # Check if training set is large enough to partition across agents
        if len(X_train) / agents < 200:
            raise ValueError("Training set too small to safely partition across agents.")
        try:
            # X_train, y_train = SMOTE().fit_resample(X_train, y_train)
            idxs = partition_dirichlet(X_train, y_train, agents, alpha)
            agents_data = [(X_train[i], y_train[i]) for i in idxs if len(np.unique(y_train[i])) > 1]
            return agents_data, X_test, y_test, X.shape[1]
        except:
            attempts += 1
    print("Class distribution in training set:", np.bincount(y_train))
    raise ValueError("SMOTE failed to generate two classes in training set after several attempts.")

# Loads and prepares the specified dataset (UNSW, CIC, or TON) for experiments.
def load_dataset(name="unsw", alpha=0.5, agents=20, attack_filter=None):
    if name == "unsw":
        df = pd.read_csv("UNSW_NB15_training-set.csv")
        df_t = pd.read_csv("UNSW_NB15_testing-set.csv")
        use = ['Normal', 'DoS', 'Exploits', 'Reconnaissance']
        df, df_t = df[df['attack_cat'].isin(use)], df_t[df_t['attack_cat'].isin(use)]
        if attack_filter:
            df = df[df['attack_cat'].isin(attack_filter)]
            df_t = df_t[df_t['attack_cat'].isin(attack_filter)]
        le = LabelEncoder()
        for col in df.select_dtypes('object'):
            df[col] = le.fit_transform(df[col].astype(str))
            df_t[col] = df_t[col].apply(lambda x: x if x in le.classes_ else 'unknown')
            le.classes_ = np.append(le.classes_, 'unknown')
            df_t[col] = le.transform(df_t[col].astype(str))
        X, y = StandardScaler().fit_transform(df.drop('label', axis=1)), df['label'].values
        X_t, y_t = StandardScaler().fit_transform(df_t.drop('label', axis=1)), df_t['label'].values
        # Filter classes with insufficient samples per agent
        y_counts = pd.Series(y).value_counts()
        min_required = max(agents * 10, 40)
        valid_classes = y_counts[y_counts >= min_required].index
        mask = np.isin(y, valid_classes)
        print(f"[INFO] {name.upper()} filtered to retain classes: {list(valid_classes)} with min_required={min_required}")
        X = X[mask]
        y = y[mask]
        if len(np.unique(y)) < 2:
            raise ValueError("Filtered dataset has fewer than 2 classes.")
        # X, y = SMOTE().fit_resample(X, y)
        max_attempts = 10
        alphas_to_try = [alpha, 1.0, 2.0]
        for alt_alpha in alphas_to_try:
            for attempt in range(max_attempts):
                try:
                    idxs = partition_dirichlet(X, y, agents, alt_alpha)
                    agents_data = [(X[i], y[i]) for i in idxs if len(np.unique(y[i])) > 1]
                    if len(agents_data) >= min(agents, 3):
                        return agents_data, X_t, y_t, X.shape[1]
                except Exception:
                    continue
        # Fallback to stratified allocation
        agents_data = partition_stratified_min_classes(X, y, agents)
        if len(agents_data) >= min(agents, 3):
            return agents_data, X_t, y_t, X.shape[1]
        # Fallback: apply SMOTE only if previous strategies failed
        try:
            X, y = SMOTE().fit_resample(X, y)
            idxs = partition_dirichlet(X, y, agents, alpha)
            agents_data = [(X[i], y[i]) for i in idxs if len(np.unique(y[i])) > 1]
            if len(agents_data) >= min(agents, 3):
                return agents_data, X_t, y_t, X.shape[1]
        except Exception:
            pass
        # Fallback: try BorderlineSMOTE if SMOTE fails
        try:
            from imblearn.over_sampling import BorderlineSMOTE
            X, y = BorderlineSMOTE().fit_resample(X, y)
            idxs = partition_dirichlet(X, y, agents, alpha)
            agents_data = [(X[i], y[i]) for i in idxs if len(np.unique(y[i])) > 1]
            if len(agents_data) >= min(agents, 3):
                return agents_data, X_t, y_t, X.shape[1]
        except Exception:
            pass
        # Fallback: try RandomUnderSampler if all oversampling fails
        try:
            from imblearn.under_sampling import RandomUnderSampler
            X, y = RandomUnderSampler().fit_resample(X, y)
            idxs = partition_dirichlet(X, y, agents, alpha)
            agents_data = [(X[i], y[i]) for i in idxs if len(np.unique(y[i])) > 1]
            if len(agents_data) >= min(agents, 3):
                return agents_data, X_t, y_t, X.shape[1]
        except Exception:
            pass
        # Fallback: try KMeansSMOTE if all above fail
        try:
            X, y = KMeansSMOTE().fit_resample(X, y)
            idxs = partition_dirichlet(X, y, agents, alpha)
            agents_data = [(X[i], y[i]) for i in idxs if len(np.unique(y[i])) > 1]
            if len(agents_data) >= min(agents, 3):
                return agents_data, X_t, y_t, X.shape[1]
        except Exception:
            pass
        print("❌ Failed to partition agents with valid class distributions after several strategies.")
        raise ValueError("Could not create valid agent partitions after Dirichlet and stratified attempts.")

    elif name == "cic":
        files = [
            "Monday-WorkingHours.pcap_ISCX.csv", "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
        ]
        df = pd.concat([pd.read_csv(f, low_memory=False) for f in files])
        df.columns = [col.strip() for col in df.columns]
        df = df.dropna(axis=1, how='all')  # Drop empty columns
        df = df.dropna(axis=0, how='any')  # Drop rows with NaNs

        # Normalize the labels BEFORE cleaning
        label_col = next((col for col in df.columns if col.strip().lower() in ["label", "class", "type"]), None)
        if not label_col:
            raise ValueError(f"CIC dataset missing label column. Columns: {df.columns.tolist()}")

        df[label_col] = df[label_col].astype(str).apply(lambda x: 0 if "benign" in x.lower() else 1)
        df = df[df[label_col].isin([0, 1])]

        # Filter classes with insufficient samples per agent
        y = df[label_col].values
        y_counts = pd.Series(y).value_counts()
        min_required = max(agents * 10, 40)
        valid_classes = y_counts[y_counts >= min_required].index
        mask = np.isin(y, valid_classes)
        print(f"[INFO] {name.upper()} filtered to retain classes: {list(valid_classes)} with min_required={min_required}")
        df = df.loc[mask]
        y = df[label_col].values
        if len(np.unique(y)) < 2:
            print("Class distribution in CIC:", pd.Series(y).value_counts().to_dict())
            raise ValueError("Filtered dataset has fewer than 2 classes.")

        if len(df[label_col].unique()) < 2:
            print("Class distribution in CIC:", df[label_col].value_counts().to_dict())
            raise ValueError("CIC dataset does not contain both classes after relabeling.")

        df = clean_df(df)
        try:
            return load_from_split(df.drop(columns=[label_col]).values, df[label_col].values, 0.2, alpha, agents)
        except Exception:
            agents_data = partition_stratified_min_classes(df.drop(columns=[label_col]).values, df[label_col].values, agents)
            if len(agents_data) >= min(agents, 3):
                test_size = int(0.2 * len(df))
                X_test = df.drop(columns=[label_col]).values[-test_size:]
                y_test = df[label_col].values[-test_size:]
                return agents_data, X_test, y_test, df.shape[1] - 1
            # Fallback: apply SMOTE only if previous strategies failed
            try:
                X_res, y_res = SMOTE().fit_resample(df.drop(columns=[label_col]).values, df[label_col].values)
                return load_from_split(X_res, y_res, 0.2, alpha, agents)
            except Exception:
                pass
            # Fallback: try BorderlineSMOTE if SMOTE fails
            try:
                from imblearn.over_sampling import BorderlineSMOTE
                X_res, y_res = BorderlineSMOTE().fit_resample(df.drop(columns=[label_col]).values, df[label_col].values)
                return load_from_split(X_res, y_res, 0.2, alpha, agents)
            except Exception:
                pass
            # Fallback: try RandomUnderSampler if all oversampling fails
            try:
                from imblearn.under_sampling import RandomUnderSampler
                X_res, y_res = RandomUnderSampler().fit_resample(df.drop(columns=[label_col]).values, df[label_col].values)
                return load_from_split(X_res, y_res, 0.2, alpha, agents)
            except Exception:
                pass
            # Fallback: try KMeansSMOTE if all above fail
            try:
                X_res, y_res = KMeansSMOTE().fit_resample(df.drop(columns=[label_col]).values, df[label_col].values)
                return load_from_split(X_res, y_res, 0.2, alpha, agents)
            except Exception:
                pass
            raise ValueError("❌ CIC fallback stratified partitioning failed.")

    elif name == "ton":
        df = pd.read_csv("ion_iot_train_test.csv")
        df = clean_df(df)
        label_col = find_label_column(df)
        df[label_col] = df[label_col].astype(str).str.lower()
        df = df[df[label_col].isin(["0", "1", "normal", "anomaly", "ddos", "scan", "dos", "backdoor", "injection"])]
        df[label_col] = df[label_col].apply(lambda x: 0 if str(x) in ["0", "normal"] else 1)
        # Filter classes with insufficient samples per agent
        y = df[label_col].values
        y_counts = pd.Series(y).value_counts()
        min_required = max(agents * 10, 40)
        valid_classes = y_counts[y_counts >= min_required].index
        mask = np.isin(y, valid_classes)
        print(f"[INFO] {name.upper()} filtered to retain classes: {list(valid_classes)} with min_required={min_required}")
        df = df.loc[mask]
        y = df[label_col].values
        if len(np.unique(y)) < 2:
            raise ValueError("Filtered dataset has fewer than 2 classes.")
        if len(df[label_col].unique()) < 2:
            raise ValueError("TON dataset does not contain both classes after relabeling.")
        try:
            return load_from_split(df.drop(columns=[label_col]).values, df[label_col].values, 0.2, alpha, agents)
        except Exception:
            agents_data = partition_stratified_min_classes(df.drop(columns=[label_col]).values, df[label_col].values, agents)
            if len(agents_data) >= min(agents, 3):
                test_size = int(0.2 * len(df))
                X_test = df.drop(columns=[label_col]).values[-test_size:]
                y_test = df[label_col].values[-test_size:]
                return agents_data, X_test, y_test, df.shape[1] - 1
            # Fallback: apply SMOTE only if previous strategies failed
            try:
                X_res, y_res = SMOTE().fit_resample(df.drop(columns=[label_col]).values, df[label_col].values)
                return load_from_split(X_res, y_res, 0.2, alpha, agents)
            except Exception:
                pass
            # Fallback: try BorderlineSMOTE if SMOTE fails
            try:
                from imblearn.over_sampling import BorderlineSMOTE
                X_res, y_res = BorderlineSMOTE().fit_resample(df.drop(columns=[label_col]).values, df[label_col].values)
                return load_from_split(X_res, y_res, 0.2, alpha, agents)
            except Exception:
                pass
            # Fallback: try RandomUnderSampler if all oversampling fails
            try:
                from imblearn.under_sampling import RandomUnderSampler
                X_res, y_res = RandomUnderSampler().fit_resample(df.drop(columns=[label_col]).values, df[label_col].values)
                return load_from_split(X_res, y_res, 0.2, alpha, agents)
            except Exception:
                pass
            # Fallback: try KMeansSMOTE if all above fail
            try:
                X_res, y_res = KMeansSMOTE().fit_resample(df.drop(columns=[label_col]).values, df[label_col].values)
                return load_from_split(X_res, y_res, 0.2, alpha, agents)
            except Exception:
                pass
            raise ValueError("❌ TON fallback stratified partitioning failed.")

# Returns a dictionary of different network topologies for agent communication.
def topologies(n):
    return {
        "Ring": {i: [(i-1)%n, (i+1)%n] if n > 1 else [] for i in range(n)},
        "Star": {i: [0] if i != 0 else list(range(1, n)) for i in range(n)},
        "FullyConnected": {i: [j for j in range(n) if j != i] for i in range(n)},
        "Random": {
            i: random.sample([j for j in range(n) if j != i], 
                             k=min(random.randint(1, 3), max(0, n - 1))) if n > 1 else []
            for i in range(n)
        }
    }

# Defines a MARL agent with training, trust updating, and decentralized aggregation capabilities.
class MARLAgent:
    def __init__(self, aid, X, y, fcount):
        self.id, self.X, self.y, self.fcount = aid, X, y, fcount
        self.epsilon, self.trust = 1.0, 1.0
        pos_weight = len(y) / (2 * sum(y))
        self.model = xgb.XGBClassifier(eval_metric="logloss", tree_method="hist", scale_pos_weight=pos_weight, n_jobs=1)
        self.energy_consumed = 0.0

    def train(self, feats):
        self.model.fit(self.X[:, feats], self.y, verbose=False)
        self.energy_consumed += 0.5  # simulate energy cost of training

    def compute_reward(self, acc, traffic):
        return 0.6 * acc + 0.2 * (1 - traffic.get('packet_loss', 0)) - 0.2 * traffic.get('latency', 0) / 100

    def aggregate(self, agents, neighbors, feats):
        if not neighbors:
            # No neighbors to aggregate from
            return
        if BASELINE_ONLY:
            # Simple unweighted average (baseline)
            Xn = np.vstack([agents[n].X[:, feats] for n in neighbors])
            yn = np.concatenate([agents[n].y for n in neighbors])
            full_X = np.vstack([self.X[:, feats], Xn])
            full_y = np.concatenate([self.y, yn])
            booster = xgb.train(self.model.get_xgb_params(), xgb.DMatrix(full_X, label=full_y), num_boost_round=1)
            self.model._Booster = booster
            self.energy_consumed += 0.2 * len(neighbors)  # simulate energy cost of communication
        else:
            # Trust-weighted MARL aggregation
            weights = np.array([agents[n].trust for n in neighbors])
            weights /= weights.sum() if weights.sum() > 0 else 1
            Xn = np.vstack([agents[n].X[:, feats] * weights[i] for i, n in enumerate(neighbors)])
            yn = np.concatenate([agents[n].y for n in neighbors])
            full_X = np.vstack([self.X[:, feats], Xn])
            full_y = np.concatenate([self.y, yn])
            booster = xgb.train(self.model.get_xgb_params(), xgb.DMatrix(full_X, label=full_y), num_boost_round=1)
            self.model._Booster = booster
            self.energy_consumed += 0.2 * len(neighbors)  # simulate energy cost of communication

# Simulates traffic metrics: bandwidth, latency, and packet loss.
def simulate(simulate_loss=False, loss_rate=0.0):
    return {
        "bandwidth": random.uniform(50, 100),
        "latency": random.uniform(10, 50),
        "packet_loss": random.uniform(0, 5) if not simulate_loss else loss_rate
    }

# Generates and saves boxplots for detection metrics.
def plot_box(df, name):
    for m in ['Accuracy', 'Precision', 'Recall', 'F1']:
        plt.figure(); sns.boxplot(x='Topology', y=m, data=df)
        plt.title(f"{m} ({name})"); plt.savefig(f"{name}_{m}.png"); plt.close()


# --- MARL Parameter Study with Progress Bar and Flexible Experimentation ---
from tqdm import tqdm

# This function runs MARL experiments across datasets, topologies, alphas, and agent counts.
# It saves detection and traffic metrics for each configuration in clearly named CSV files.
def run_parameter_study(alphas=[0.5], agent_counts=[20], datasets=["unsw", "cic", "ton"], repeats=3):
    """
    Runs MARL-XGB experiments for all combinations of dataset, topology, alpha, and agent count.
    Results are saved as CSV files for each configuration.
    """
    LOSS_RATES = [round(i * 0.1, 1) for i in range(11)]
    for dataset in datasets:
        for alpha in alphas:
            for agents in agent_counts:
                for attacks in ATTACK_MODELS:
                    label = "_".join([a.lower() for a in attacks])
                    print(f"🧪 Running MARL-XGB on {dataset.upper()} | alpha={alpha} | agents={agents} | attacks={label}")
                    # Load and partition the dataset for current configuration
                    try:
                        agents_data, X_t, y_t, fcount = load_dataset(dataset, alpha, agents, attack_filter=attacks)
                    except Exception as e:
                        print(f"❌ Failed to load {dataset} with alpha={alpha}, agents={agents}, attacks={label}: {e}")
                        print("🔁 Retrying with one less agent...")
                        try:
                            agents_data, X_t, y_t, fcount = load_dataset(dataset, alpha, agents - 1, attack_filter=attacks)
                            print(f"✅ Retry successful with {agents - 1} agents.")
                        except Exception as e2:
                            print(f"❌ Retry failed with alpha={alpha}, agents={agents - 1}, attacks={label}: {e2}")
                            failed_runs_log.append({
                                "Dataset": dataset, "Alpha": alpha, "Agents": agents, "Attacks": label, "Reason": str(e2)
                            })
                            continue
                    # For each packet loss scenario, and for each topology, run multiple repeats and collect results
                    for loss_rate in LOSS_RATES:
                        for topo_name, topo in topologies(len(agents_data)).items():
                            detection_records = []
                            traffic_records = []
                            print(f"🔄 Topology: {topo_name} | PacketLoss: {loss_rate}")
                            # Use tqdm to show progress for repeats
                            for _ in tqdm(range(repeats), desc=f"{dataset.upper()}-{topo_name}-α{alpha}-{label}-loss{int(loss_rate*100)}"):
                                current_agents_data = agents_data.copy()
                                # Simulate random joining/leaving of agents (dynamic mobile network)
                                drop_prob = 0.2  # 20% chance for a node to leave
                                join_prob = 0.2  # 20% chance for a node to join (if under max agents)
                                if len(current_agents_data) > 5 and random.random() < drop_prob:
                                    drop_idx = random.choice(range(len(current_agents_data)))
                                    del current_agents_data[drop_idx]
                                elif len(current_agents_data) < 20 and random.random() < join_prob:
                                    # Simulate joining by duplicating a random existing agent
                                    join_idx = random.choice(range(len(current_agents_data)))
                                    current_agents_data.append(current_agents_data[join_idx])

                                current_topo = topologies(len(current_agents_data))[topo_name]
                                agents_list = [MARLAgent(i, *current_agents_data[i], fcount) for i in range(len(current_agents_data))]
                                if not agents_list:
                                    print(f"⚠️ Skipping {dataset.upper()}-{topo_name}-α{alpha}-{label} due to empty agent list.")
                                    continue
                                # New check: skip if agent list is empty after retry
                                if not agents_list:
                                    print(f"⚠️ Skipping {dataset.upper()}-{topo_name}-α{alpha}-{label} due to empty agent list after retry.")
                                    continue
                                feats = random.sample(range(fcount), random.randint(10, fcount))
                                for agent in agents_list:
                                    agent.train(feats)
                                    agent.aggregate(agents_list, current_topo[agent.id], feats)
                                preds = agents_list[0].model.predict(X_t[:, feats])
                                acc = accuracy_score(y_t, preds)
                                p = precision_score(y_t, preds, zero_division=0)
                                r = recall_score(y_t, preds, zero_division=0)
                                f1 = f1_score(y_t, preds, zero_division=0)
                                avg_energy = np.mean([agent.energy_consumed for agent in agents_list])
                                traffic_sample = simulate(simulate_loss=True, loss_rate=loss_rate)
                                reward = agents_list[0].compute_reward(acc, traffic_sample)
                                # Trust update: skip if in baseline mode
                                if not BASELINE_ONLY:
                                    for agent in agents_list:
                                        agent.trust = 0.9 * agent.trust + 0.1 * reward
                                detection_records.append({
                                    "Dataset": dataset, "Topology": topo_name,
                                    "Alpha": alpha, "Agents": len(current_agents_data),
                                    "Accuracy": acc, "Precision": p, "Recall": r, "F1": f1,
                                    "Energy": avg_energy,
                                    "PacketLossSimulated": loss_rate,
                                    "PacketLossScenario": loss_rate
                                })
                                traffic_records.append({
                                    "Dataset": dataset, "Topology": topo_name,
                                    "Alpha": alpha, "Agents": len(current_agents_data),
                                    **traffic_sample,
                                    "PacketLossSimulated": loss_rate,
                                    "PacketLossScenario": loss_rate
                                })
                            # Save results for this topology/configuration and packet loss scenario
                            df_det = pd.DataFrame(detection_records)
                            df_tra = pd.DataFrame(traffic_records)
                            suffix = "_baseline" if BASELINE_ONLY else ""
                            det_filename = f"results/baseline/marl_xgb_{topo_name}_topology_{dataset}_detection_metrics_{alpha}_{label}_loss{int(loss_rate*100)}{suffix}.csv" if BASELINE_ONLY else f"results/full/marl_xgb_{topo_name}_topology_{dataset}_detection_metrics_{alpha}_{label}_loss{int(loss_rate*100)}.csv"
                            tra_filename = f"results/baseline/marl_xgb_{topo_name}_topology_{dataset}_traffic_metrics_{alpha}_{label}_loss{int(loss_rate*100)}{suffix}.csv" if BASELINE_ONLY else f"results/full/marl_xgb_{topo_name}_topology_{dataset}_traffic_metrics_{alpha}_{label}_loss{int(loss_rate*100)}.csv"
                            # Ensure directories exist
                            os.makedirs(os.path.dirname(det_filename), exist_ok=True)
                            os.makedirs(os.path.dirname(tra_filename), exist_ok=True)
                            df_det.to_csv(det_filename, index=False)
                            df_tra.to_csv(tra_filename, index=False)
                            print(f"✅ Saved {det_filename} and {tra_filename}")
                            if ENABLE_PLOTS:
                                plot_box(df_det, f"{topo_name}_{dataset}_{alpha}_{label}_loss{int(loss_rate*100)}{suffix}")
    # After all loops, log failed runs and plot summary
    if failed_runs_log:
        pd.DataFrame(failed_runs_log).to_csv("failed_runs_log.csv", index=False)
        success_count = sum(1 for _ in open("failed_runs_log.csv")) - 1
    else:
        success_count = len(ALPHAS) * len(AGENT_COUNTS) * len(DATASETS) * len(ATTACK_MODELS)

    total_runs = success_count + len(failed_runs_log)
    plt.figure()
    plt.pie([success_count, len(failed_runs_log)],
            labels=['Successful', 'Failed'],
            autopct='%1.1f%%', startangle=90)
    plt.title('Run Outcomes')
    suffix = "_baseline" if BASELINE_ONLY else ""
    plt.savefig(f'run_success_summary{suffix}.png')
    plt.close()




# --- HDFL Baseline Implementation ---


# Creates boxplots for HDFL performance metrics per dataset.

# --- Naive baseline temporarily disabled ---
# def run_naive_baseline(datasets=["unsw", "cic", "ton"], alphas=[0.5], agent_counts=[10], repeats=5):
#     """
#     Runs a naive baseline that always predicts the majority class from the training set.
#     Saves detection metrics for each configuration.
#     """
#     for dataset in datasets:
#         for alpha in alphas:
#             for agents in agent_counts:
#                 for attacks in ATTACK_MODELS:
#                     label = "_".join([a.lower() for a in attacks])
#                     print(f"🔍 Naive: {dataset.upper()} | α={alpha} | agents={agents} | attacks={label}")
#                     try:
#                         agents_data, X_test, y_test, _ = load_dataset(dataset, alpha=alpha, agents=agents, attack_filter=attacks)
#                     except Exception as e:
#                         print(f"❌ Failed to load {dataset} for naive baseline: {e}")
#                         continue
#
#                     results = []
#                     for rep in range(1, repeats + 1):
#                         y_train_all = np.concatenate([y for _, y in agents_data])
#                         majority_class = np.bincount(y_train_all).argmax()
#                         y_pred = np.full_like(y_test, majority_class)
#                         acc = accuracy_score(y_test, y_pred)
#                         prec = precision_score(y_test, y_pred, zero_division=0)
#                         rec = recall_score(y_test, y_pred, zero_division=0)
#                         f1 = f1_score(y_test, y_pred, zero_division=0)
#                         results.append({
#                             "Dataset": dataset, "Alpha": alpha, "Agents": agents,
#                             "Mode": "Naive", "Repeat": rep,
#                             "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1
#                         })
#
#                     filename = f"naive_baseline_{dataset}_detection_metrics_{alpha}_{label}.csv"
#                     pd.DataFrame(results).to_csv(filename, index=False)
#                     print(f"✅ Saved {filename}")


# --- Parallelized MARL and HDFL runners ---
def run_marl_parallel():
    with multiprocessing.Pool(processes=pool_size) as pool:
        configs = [(alpha, agents, dataset, REPEATS)
                   for dataset in DATASETS
                   for alpha in ALPHAS
                   for agents in AGENT_COUNTS]
        pool.starmap(run_parameter_study_wrapper, configs)

def run_parameter_study_wrapper(alpha, agents, dataset, repeats):
    try:
        run_parameter_study(alphas=[alpha], agent_counts=[agents], datasets=[dataset], repeats=repeats)
    except Exception as e:
        print(f"⚠️ MARL run failed for dataset={dataset}, alpha={alpha}, agents={agents}: {e}")


# --- Naive Baseline Implementation ---
def run_naive_baseline(agent_params, neighbors, eta=0.1, rounds=10):
    for _ in range(rounds):
        updated_params = agent_params.copy()
        for i, theta_i in agent_params.items():
            neighbor_indices = neighbors[i]
            delta_sum = sum(agent_params[j] - theta_i for j in neighbor_indices)
            updated_params[i] += eta * delta_sum / max(1, len(neighbor_indices))
        agent_params = updated_params.copy()
    return agent_params

def run_naive_baseline_parallel(agents=20, rounds=10):
    import multiprocessing
    import random

    agent_params = {i: random.uniform(0, 1) for i in range(agents)}
    neighbors = {i: [j for j in range(agents) if i != j and (i + j) % 4 == 0] for i in range(agents)}

    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            run_naive_baseline,
            [(agent_params.copy(), neighbors, 0.1, rounds) for _ in range(agents)]
        )

    averaged_results = {}
    for i, result in enumerate(results):
        for k, v in result.items():
            averaged_results[k] = averaged_results.get(k, 0) + v
    for k in averaged_results:
        averaged_results[k] /= agents

    print("Final Naive Baseline Model Parameters:")
    for agent, value in averaged_results.items():
        print(f"Agent {agent}: {value:.2f}")


# --- Additions: Save baseline results after evaluation loop ---
# Moved to baselines/naive_baseline.py for detailed metrics and CSV export

# --- HDFL Parameter Study -- mirror of run_parameter_study but with hdfl_ prefix and centralized logic placeholder
from sklearn.linear_model import LogisticRegression

# HDFL: Simulate hierarchical federated learning with MLP or Logistic Regression
def run_hdfl_parameter_study(alphas=[0.5], agent_counts=[20], datasets=["unsw", "cic", "ton"], repeats=3):
    """
    Runs HDFL experiments for all combinations of dataset, topology, alpha, and agent count.
    Results are saved as CSV files for each configuration.
    """
    LOSS_RATES = [round(i * 0.1, 1) for i in range(11)]
    # HDFL parameters
    num_local_epochs = 2
    num_global_rounds = 3
    num_aggregators = 2  # Number of local aggregators in hierarchy
    for dataset in datasets:
        for alpha in alphas:
            for agents in agent_counts:
                for attacks in ATTACK_MODELS:
                    label = "_".join([a.lower() for a in attacks])
                    print(f"🧪 Running HDFL on {dataset.upper()} | alpha={alpha} | agents={agents} | attacks={label}")
                    try:
                        agents_data, X_t, y_t, fcount = load_dataset(dataset, alpha, agents, attack_filter=attacks)
                    except Exception as e:
                        print(f"❌ Failed to load {dataset} with alpha={alpha}, agents={agents}, attacks={label}: {e}")
                        continue
                    for loss_rate in LOSS_RATES:
                        for topo_name, topo in topologies(len(agents_data)).items():
                            detection_records = []
                            traffic_records = []
                            print(f"🔄 [HDFL] Topology: {topo_name} | PacketLoss: {loss_rate}")
                            for _ in tqdm(range(repeats), desc=f"HDFL-{dataset.upper()}-{topo_name}-α{alpha}-{label}-loss{int(loss_rate*100)}"):
                                current_agents_data = agents_data.copy()
                                feats = random.sample(range(fcount), random.randint(10, fcount))
                                # Partition agents into groups for hierarchical aggregation
                                num_clients = len(current_agents_data)
                                if num_clients < num_aggregators:
                                    agg_groups = [list(range(num_clients))]
                                else:
                                    group_size = (num_clients + num_aggregators - 1) // num_aggregators
                                    agg_groups = [list(range(i*group_size, min((i+1)*group_size, num_clients)))
                                                  for i in range(num_aggregators)]
                                # Initialize global model (Logistic Regression)
                                global_model = LogisticRegression(max_iter=100, solver='lbfgs')
                                # Initialize model parameters (weights)
                                # Use first agent's data to initialize
                                global_model.fit(current_agents_data[0][0][:, feats], current_agents_data[0][1])
                                coef = global_model.coef_.copy()
                                intercept = global_model.intercept_.copy()
                                # HDFL rounds
                                for rnd in range(num_global_rounds):
                                    # Each aggregator collects updates from its group
                                    agg_coefs = []
                                    agg_intercepts = []
                                    for group in agg_groups:
                                        group_coefs = []
                                        group_intercepts = []
                                        for idx in group:
                                            X_local, y_local = current_agents_data[idx]
                                            local_model = LogisticRegression(max_iter=100, solver='lbfgs')
                                            local_model.coef_ = coef.copy()
                                            local_model.intercept_ = intercept.copy()
                                            # Fit local model for a few epochs (simulate by refitting)
                                            for ep in range(num_local_epochs):
                                                local_model.fit(X_local[:, feats], y_local)
                                            group_coefs.append(local_model.coef_)
                                            group_intercepts.append(local_model.intercept_)
                                        # Aggregate at aggregator (average)
                                        agg_coefs.append(np.mean(np.vstack(group_coefs), axis=0, keepdims=True))
                                        agg_intercepts.append(np.mean(np.vstack(group_intercepts), axis=0, keepdims=True))
                                    # Global aggregation (average across aggregators)
                                    coef = np.mean(np.vstack(agg_coefs), axis=0, keepdims=True)
                                    intercept = np.mean(np.vstack(agg_intercepts), axis=0, keepdims=True)
                                # Set global model weights
                                global_model.coef_ = coef
                                global_model.intercept_ = intercept
                                # Evaluate on test set
                                preds = global_model.predict(X_t[:, feats])
                                acc = accuracy_score(y_t, preds)
                                p = precision_score(y_t, preds, zero_division=0)
                                r = recall_score(y_t, preds, zero_division=0)
                                f1 = f1_score(y_t, preds, zero_division=0)
                                traffic_sample = simulate(simulate_loss=True, loss_rate=loss_rate)
                                detection_records.append({
                                    "Dataset": dataset, "Topology": topo_name,
                                    "Alpha": alpha, "Agents": len(current_agents_data),
                                    "Accuracy": acc, "Precision": p, "Recall": r, "F1": f1,
                                    "PacketLossSimulated": loss_rate,
                                    "PacketLossScenario": loss_rate
                                })
                                traffic_records.append({
                                    "Dataset": dataset, "Topology": topo_name,
                                    "Alpha": alpha, "Agents": len(current_agents_data),
                                    **traffic_sample,
                                    "PacketLossSimulated": loss_rate,
                                    "PacketLossScenario": loss_rate
                                })
                            df_det = pd.DataFrame(detection_records)
                            df_tra = pd.DataFrame(traffic_records)
                            det_filename = f"hdfl_{topo_name}_topology_{dataset}_detection_metrics_{alpha}_{label}_loss{int(loss_rate*100)}.csv"
                            tra_filename = f"hdfl_{topo_name}_topology_{dataset}_traffic_metrics_{alpha}_{label}_loss{int(loss_rate*100)}.csv"
                            df_det.to_csv(det_filename, index=False)
                            df_tra.to_csv(tra_filename, index=False)
                            print(f"✅ Saved {det_filename} and {tra_filename}")

# Wrapper for HDFL parameter study
def run_hdfl_parameter_study_wrapper(alpha, agents, dataset, repeats):
    try:
        run_hdfl_parameter_study(alphas=[alpha], agent_counts=[agents], datasets=[dataset], repeats=repeats)
    except Exception as e:
        print(f"⚠️ HDFL run failed for dataset={dataset}, alpha={alpha}, agents={agents}: {e}")

def run_hdfl_parallel():
    with multiprocessing.Pool(processes=pool_size) as pool:
        configs = [(alpha, agents, dataset, REPEATS)
                   for dataset in DATASETS
                   for alpha in ALPHAS
                   for agents in AGENT_COUNTS]
        pool.starmap(run_hdfl_parameter_study_wrapper, configs)




# --- Baseline parameter grid generation function ---
from multiprocessing import Pool
from tqdm import tqdm

def run_single_baseline(params):
    # Placeholder for the actual baseline experiment logic
    # Unpack the params tuple
    dataset, alpha, agents, attacks, loss_rate, topo_name = params
    # You should implement the baseline logic here and populate detection_records and traffic_records lists.
    # For demonstration, let's create dummy records if not present.
    detection_records = []
    traffic_records = []
    label = "_".join([str(a).lower() for a in attacks])
    # Example: simulate a single dummy record if no logic present
    detection_records.append({
        "Dataset": dataset, "Topology": topo_name,
        "Alpha": alpha, "Agents": agents,
        "Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
        "Energy": 0.0,
        "PacketLossSimulated": loss_rate,
        "PacketLossScenario": loss_rate
    })
    traffic_records.append({
        "Dataset": dataset, "Topology": topo_name,
        "Alpha": alpha, "Agents": agents,
        "bandwidth": 0.0, "latency": 0.0, "packet_loss": loss_rate,
        "PacketLossSimulated": loss_rate,
        "PacketLossScenario": loss_rate
    })
    # --- Save results to CSV after all records are appended ---
    os.makedirs("results/baseline", exist_ok=True)
    det_path = f"results/baseline/baseline_detection_{dataset}_{topo_name}_alpha{alpha}_agents{agents}_loss{int(loss_rate*100)}_{label}.csv"
    tra_path = f"results/baseline/baseline_traffic_{dataset}_{topo_name}_alpha{alpha}_agents{agents}_loss{int(loss_rate*100)}_{label}.csv"
    pd.DataFrame(detection_records).to_csv(det_path, index=False)
    pd.DataFrame(traffic_records).to_csv(tra_path, index=False)
    print(f"✅ Saved {det_path} and {tra_path}")

# Run a selected subset of naive baseline tests
def run_selected_naive_baseline(params_list=None):
    print("Running selected naive baseline tests...")
    if params_list is None:
        print("No parameter list provided to run_selected_naive_baseline.")
        return
    with Pool() as pool:
        list(tqdm(pool.imap_unordered(run_single_baseline, params_list), total=len(params_list)))
    print("Selected naive baseline tests completed.")


# Use the full baseline parameter sweep as main entry point
from baselines.naive_baseline import run_selected_naive_baseline

if __name__ == "__main__":
    # Full baseline parameter sweep
    datasets = ["cic", "ton", "unsw"]
    alphas = [0.1, 0.3, 0.5, 1.0, 2.0]
    agent_counts = [5, 6, 7, 8, 9, 10]
    topologies = ["Ring", "Star", "FullyConnected", "Random"]
    attack_models = [
        ["normal", "dos"],
        ["normal", "fuzzers"],
        ["normal", "reconnaissance"],
        ["normal", "dos", "reconnaissance"]
    ]
    LOSS_RATES = [round(i * 0.1, 1) for i in range(11)]

    full_param_list = []
    for dataset in datasets:
        for alpha in alphas:
            for agents in agent_counts:
                for attacks in attack_models:
                    for loss in LOSS_RATES:
                        for topo in topologies:
                            full_param_list.append((dataset, alpha, agents, attacks, loss, topo))

    run_selected_naive_baseline(full_param_list)