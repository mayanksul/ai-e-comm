import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List

# Add project root to Python path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class RecommendationAgent:
    """
    An agent that recommends products from a catalog based on user preferences.
    """
    def __init__(self, data_path: str):
        """
        Initializes the RecommendationAgent and loads the product catalog.

        Args:
            data_path (str): The file path to the product catalog (Excel or CSV).
        """
        self.catalog = self._load_catalog(data_path)

    def _load_catalog(self, data_path: str) -> pd.DataFrame:
        """
        Reads product data from an Excel file and normalizes it for matching.
        
        Args:
            data_path (str): The path to the data file.

        Returns:
            pd.DataFrame: A DataFrame containing the normalized product catalog.
        """
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")
            
        # df = pd.read_excel(data_path)
        df = pd.read_csv(data_path)
        # Normalize text columns for case-insensitive matching
        for col in ["category", "fit", "fabric", "sleeve_length",
                    "color_or_print", "occasion", "neckline", "length", "pant_type"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        # Align column names with the mapping dictionary keys
        column_rename_map = {"length": "coverage_length"}
        df.rename(columns=column_rename_map, inplace=True)

        return df

    def _score_row(self, row: pd.Series, mp: Dict) -> int:
        """
        Scores a single product row against the user preference mapping.
        A score of -1 indicates a hard-filter rejection.

        Args:
            row (pd.Series): A row from the catalog DataFrame.
            mp (Dict): The user preference mapping dictionary.

        Returns:
            int: The match score for the product.
        """
        # Hard filters: reject if essential criteria don't match
        if "category" in mp and row.get("category") not in [c.lower() for c in mp.get("category", [])]:
            return -1
        if "price_max" in mp and row.get("price") > mp.get("price_max"):
            return -1

        # Soft-match scoring: award points for matching attributes
        score = 0
        def bump(condition, points=1):
            nonlocal score
            if condition:
                score += points
        
        if "size" in mp:
            available_sizes = {s.strip().upper() for s in str(row.get("available_sizes", "")).split(",")}
            bump(any(sz.upper() in available_sizes for sz in mp.get("size", [])), 2)

        bump("fit" in mp and mp.get("fit", "").lower() in row.get("fit", ""), 2)
        bump("fabric" in mp and any(f.lower() in row.get("fabric", "") for f in mp.get("fabric", [])), 2)
        bump("sleeve_length" in mp and mp.get("sleeve_length") in row.get("sleeve_length", ""), 1)
        bump("coverage_length" in mp and mp.get("coverage_length") in str(row.get("coverage_length", "")).lower(), 1)
        bump("color_or_print" in mp and mp.get("color_or_print") in row.get("color_or_print", ""), 1)
        bump("neckline" in mp and mp.get("neckline") in row.get("neckline", ""), 1)
        bump("pant_type" in mp and mp.get("pant_type") in row.get("pant_type", ""), 1)
        
        return score

    def get_recommendations(self, mp: Dict, n: int = 2) -> List[Dict]:
        """
        Filters and ranks products to find the top N recommendations.

        Args:
            mp (Dict): The user preference mapping dictionary.
            n (int): The number of top recommendations to return.

        Returns:
            List[Dict]: A list of recommended products, or an empty list if none match.
        """
        if self.catalog.empty:
            return []

        scored_df = self.catalog.copy()
        scored_df["match_score"] = scored_df.apply(self._score_row, axis=1, mp=mp)
        
        scored_df = scored_df[scored_df["match_score"] >= 0].sort_values(
            by=["match_score", "price"], ascending=[False, True]
        )
        
        if not scored_df.empty:
            # Handle multi-category requests by getting the top pick from each
            if "category" in mp and len(mp.get("category", [])) > 1:
                results = []
                for category in mp["category"]:
                    cat_items = scored_df[scored_df["category"] == category.lower()].head(1)
                    if not cat_items.empty:
                        results.append(cat_items)
                final_df = pd.concat(results) if results else scored_df.head(0)
            else:
                final_df = scored_df.head(n)

            # Define columns to return
            return_cols = ["id", "name", "category", "price", "fit", "fabric", "sleeve_length", "available_sizes"]
            # Filter to only existing columns to prevent errors
            existing_cols = [col for col in return_cols if col in final_df.columns]
            return final_df[existing_cols].to_dict(orient="records")
        
        return []

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Define project paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_file_path = os.path.join(project_root, 'data', 'raw', 'Apparels_shared.csv')
    
    # 2. Instantiate the agent
    recommender = RecommendationAgent(data_file_path)

    # 3. Provide a sample mapping dictionary
    mapping = {
        "category": ["dress", "top"],
        "sleeve_length": "sleeveless",
        "price_max": 100,
        "size": ["S", "M"],
        "fabric": ["linen", "cotton"],
        "fit": "relaxed"
    }

    # 4. Get recommendations
    picks = recommender.get_recommendations(mapping, n=2)

    # 5. Print the results
    import json
    print("--- Top Recommendations ---")
    print(json.dumps(picks, indent=2))
    print("---------------------------") 