from collections import Counter
import numpy as np
import pandas as pd
from imblearn.under_sampling import TomekLinks, ClusterCentroids, RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler


def apply_sampling(
        X, 
        y, 
        use_tomek=True,
        use_cluster=True,
        use_random_under=False,      # 🔥 RandomUnderSampler control
        use_over="adasyn",           # 'adasyn', 'smote', 'random', None
        cluster_ratio=3,
        random_state=42,
        verbose=True
    ):
    """
    Versatile sampling pipeline:
    
    - TomekLinks (optional)
    - ClusterCentroids (optional)
    - RandomUnderSampler (optional: use_random_under=True)
    - Oversampling: ADASYN / SMOTE / RandomOverSampler
    """

    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values.ravel()
    else:
        y = np.array(y).ravel()

    if verbose:
        print("📌 Original distribution:", Counter(y))

    X_res, y_res = X, y

    if use_tomek:
        if verbose: print("\n🔍 Applying TomekLinks...")
        tl = TomekLinks(sampling_strategy="majority")
        X_res, y_res = tl.fit_resample(X_res, y_res)
        if verbose:
            print("➡️ After TomekLinks:", Counter(y_res))


    if use_cluster:
        if verbose: print("\n🔻 Applying ClusterCentroids...")

        counter = Counter(y_res)
        minority_class = min(counter, key=counter.get)
        majority_class = max(counter, key=counter.get)

        minority_count = counter[minority_class]
        desired_majority_count = minority_count * cluster_ratio

        sampling_strategy = {
            minority_class: minority_count,
            majority_class: desired_majority_count
        }

        cc = ClusterCentroids(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )

        X_res, y_res = cc.fit_resample(X_res, y_res)

        if verbose:
            print("➡️ After ClusterCentroids:", Counter(y_res))

    if use_random_under:
        if verbose: print("\n🔻 Applying RandomUnderSampler...")

        rus = RandomUnderSampler(random_state=random_state)
        X_res, y_res = rus.fit_resample(X_res, y_res)

        if verbose:
            print("➡️ After RandomUnderSampler:", Counter(y_res))


    if use_over is not None:

        if use_over.lower() == "adasyn":
            sampler = ADASYN(random_state=random_state)
            name = "ADASYN"

        elif use_over.lower() == "smote":
            sampler = SMOTE(random_state=random_state)
            name = "SMOTE"

        elif use_over.lower() == "random":
            sampler = RandomOverSampler(random_state=random_state)
            name = "RandomOverSampler"

        else:
            raise ValueError("use_over must be: 'adasyn', 'smote', 'random', None")

        if verbose: print(f"\n🔺 Applying {name}...")

        X_res, y_res = sampler.fit_resample(X_res, y_res)

        if verbose:
            print(f"➡️ After {name}:", Counter(y_res))

    if verbose:
        print("\n✅ Final distribution:", Counter(y_res))

    return X_res, y_res
