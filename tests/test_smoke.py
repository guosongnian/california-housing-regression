"""不下载数据的快速单元测试。"""

import unittest

import pandas as pd

from housing_regression import add_features, build_search


class HousingSmokeTest(unittest.TestCase):
    def test_feature_engineering_and_search(self) -> None:
        frame = pd.DataFrame(
            {
                "longitude": [-122.0, -118.0],
                "latitude": [37.0, 34.0],
                "housing_median_age": [20.0, 30.0],
                "total_rooms": [100.0, 200.0],
                "total_bedrooms": [20.0, 50.0],
                "population": [80.0, 150.0],
                "households": [10.0, 25.0],
                "median_income": [4.0, 6.0],
                "ocean_proximity": ["NEAR BAY", "INLAND"],
            }
        )
        engineered = add_features(frame)
        self.assertIn("rooms_per_household", engineered)
        search = build_search(engineered, quick=True)
        self.assertEqual(search.scoring, "neg_root_mean_squared_error")


if __name__ == "__main__":
    unittest.main()
