from typing import Any, Dict, List

import numpy as np


class FeatureVectoriser:
    def transform_info_to_features(self, tracks: List[Dict[str, Any]]) -> np.ndarray:
        features = []

        for track in tracks:
            audio_features = track["audio_features"]
            features.append(list(audio_features.values()))

        return np.asarray(features)
