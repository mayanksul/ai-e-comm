import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from loguru import logger

# Add project root to Python path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class VectorRecommendationAgent:
    """
    An enhanced agent that combines vector embeddings with rule-based scoring 
    for better product recommendations.
    """
    def __init__(self, data_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the agent with product catalog and embedding model.

        Args:
            data_path (str): Path to the product catalog CSV file
            model_name (str): Name of the sentence transformer model
        """
        self.model = SentenceTransformer(model_name)
        self.catalog = self._load_catalog(data_path)
        self.embeddings = None
        self.embeddings_path = data_path.replace('.csv', '_embeddings.pkl')

        logger.info(f"Embedding model path: {self.embeddings_path}")
        
        # Create embeddings for products
        self._create_or_load_embeddings()

    def _load_catalog(self, data_path: str) -> pd.DataFrame:
        """Load and normalize product catalog."""
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")
            
        df = pd.read_csv(data_path)
        
        # Normalize text columns for case-insensitive matching
        text_columns = ["category", "fit", "fabric", "sleeve_length",
                       "color_or_print", "occasion", "neckline", "length", "pant_type"]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        # Rename length to coverage_length for consistency
        if "length" in df.columns:
            df.rename(columns={"length": "coverage_length"}, inplace=True)

        # Create product descriptions for embedding
        df['description'] = self._create_product_descriptions(df)
        
        return df

    def _create_product_descriptions(self, df: pd.DataFrame) -> pd.Series:
        """Create rich text descriptions for products to embed."""
        descriptions = []
        
        for _, row in df.iterrows():
            desc_parts = [
                f"{row.get('name', '')}",
                f"{row.get('category', '')}",
                f"{row.get('fit', '')} fit" if pd.notna(row.get('fit')) else "",
                f"{row.get('fabric', '')} fabric" if pd.notna(row.get('fabric')) else "",
                f"{row.get('sleeve_length', '')}" if pd.notna(row.get('sleeve_length')) else "",
                f"{row.get('color_or_print', '')}" if pd.notna(row.get('color_or_print')) else "",
                f"{row.get('occasion', '')}" if pd.notna(row.get('occasion')) else "",
                f"{row.get('neckline', '')}" if pd.notna(row.get('neckline')) else "",
                f"{row.get('coverage_length', '')}" if pd.notna(row.get('coverage_length')) else "",
                f"{row.get('pant_type', '')}" if pd.notna(row.get('pant_type')) else "",
            ]
            
            # Filter out empty parts and join
            desc = " ".join([part for part in desc_parts if part and part.strip()])
            descriptions.append(desc)
        
        return pd.Series(descriptions)

    def _create_or_load_embeddings(self):
        """Create embeddings for products or load from cache."""
        if Path(self.embeddings_path).exists():
            print("Loading cached embeddings...")
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            print("Creating embeddings for products...")
            descriptions = self.catalog['description'].tolist()
            self.embeddings = self.model.encode(descriptions)
            
            # Cache embeddings
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            print(f"Embeddings cached to {self.embeddings_path}")

    def _score_row_rule_based(self, row: pd.Series, mp: Dict) -> int:
        """
        Rule-based scoring (same as original agent).
        Returns -1 for hard filter rejection, otherwise positive score.
        """
        # Hard filters
        if "category" in mp and row.get("category") not in [c.lower() for c in mp.get("category", [])]:
            return -1
        if "price_max" in mp and row.get("price") > mp.get("price_max"):
            return -1

        # Soft-match scoring
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

    def _create_query_text(self, mp: Dict) -> str:
        """Convert mapping dictionary to a search query for embedding."""
        query_parts = []
        
        # Add categories
        if "category" in mp:
            categories = mp["category"] if isinstance(mp["category"], list) else [mp["category"]]
            query_parts.extend(categories)
        
        # Add other attributes
        for key in ["fit", "fabric", "sleeve_length", "color_or_print", 
                   "occasion", "neckline", "coverage_length", "pant_type"]:
            if key in mp:
                value = mp[key]
                if isinstance(value, list):
                    query_parts.extend(value)
                else:
                    query_parts.append(str(value))
        
        return " ".join(query_parts)

    def get_recommendations(self, mp: Dict, n: int = 2, 
                          embedding_weight: float = 0.4, 
                          rule_weight: float = 0.6) -> List[Dict]:
        """
        Get recommendations using combined vector similarity and rule-based scoring.

        Args:
            mp (Dict): User preference mapping dictionary
            n (int): Number of recommendations to return
            embedding_weight (float): Weight for vector similarity score (0-1)
            rule_weight (float): Weight for rule-based score (0-1)

        Returns:
            List[Dict]: List of recommended products
        """
        if self.catalog.empty:
            return []

        # Create query embedding
        query_text = self._create_query_text(mp)
        if not query_text.strip():
            # Fallback to rule-based only if no text query
            return self._get_rule_based_recommendations(mp, n)
        
        query_embedding = self.model.encode([query_text])
        
        # Calculate vector similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Calculate rule-based scores
        scored_df = self.catalog.copy()
        scored_df["rule_score"] = scored_df.apply(self._score_row_rule_based, axis=1, mp=mp)
        
        # Filter out hard rejections
        scored_df = scored_df[scored_df["rule_score"] >= 0]
        
        if scored_df.empty:
            return []
        
        # Normalize similarities (0-1 range)
        max_sim = similarities.max() if similarities.max() > 0 else 1
        normalized_similarities = similarities / max_sim
        
        # Normalize rule scores (0-1 range)  
        valid_indices = scored_df.index
        valid_similarities = normalized_similarities[valid_indices]
        
        max_rule_score = scored_df["rule_score"].max() if scored_df["rule_score"].max() > 0 else 1
        normalized_rule_scores = scored_df["rule_score"] / max_rule_score
        
        # Combined score
        scored_df["vector_score"] = valid_similarities
        scored_df["combined_score"] = (
            embedding_weight * scored_df["vector_score"] + 
            rule_weight * normalized_rule_scores
        )
        
        # Sort by combined score and price
        scored_df = scored_df.sort_values(
            by=["combined_score", "price"], 
            ascending=[False, True]
        )
        
        # Handle multi-category requests
        if "category" in mp and len(mp.get("category", [])) > 1:
            results = []
            for category in mp["category"]:
                cat_items = scored_df[scored_df["category"] == category.lower()].head(1)
                if not cat_items.empty:
                    results.append(cat_items)
            final_df = pd.concat(results) if results else scored_df.head(0)
        else:
            final_df = scored_df.head(n)

        # Return formatted results
        return_cols = ["id", "name", "category", "price", "fit", "fabric", 
                      "sleeve_length", "available_sizes", "combined_score", "vector_score", "rule_score"]
        existing_cols = [col for col in return_cols if col in final_df.columns]
        
        return final_df[existing_cols].to_dict(orient="records")

    def _get_rule_based_recommendations(self, mp: Dict, n: int) -> List[Dict]:
        """Fallback to pure rule-based recommendations."""
        scored_df = self.catalog.copy()
        scored_df["rule_score"] = scored_df.apply(self._score_row_rule_based, axis=1, mp=mp)
        
        scored_df = scored_df[scored_df["rule_score"] >= 0].sort_values(
            by=["rule_score", "price"], ascending=[False, True]
        )
        
        if not scored_df.empty:
            final_df = scored_df.head(n)
            return_cols = ["id", "name", "category", "price", "fit", "fabric", "sleeve_length", "available_sizes"]
            existing_cols = [col for col in return_cols if col in final_df.columns]
            return final_df[existing_cols].to_dict(orient="records")
        
        return []


if __name__ == '__main__':
    # Example usage
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_file_path = os.path.join(project_root, 'data', 'raw', 'Apparels_shared.csv')
    
    # Initialize vector-based recommender
    recommender = VectorRecommendationAgent(data_file_path)

    # Test with sample mapping
    mapping = {
        "category": ["dress"],
        "sleeve_length": "sleeveless",
        "price_max": 200,
        "size": ["S", "M"],
        "fabric": ["linen", "cotton"],
        "fit": "relaxed"
    }

    # Get recommendations
    picks = recommender.get_recommendations(mapping, n=3)

    # Print results
    import json
    print("=== Vector + Rule-based Recommendations ===")
    for i, pick in enumerate(picks, 1):
        print(f"\n{i}. {pick['name']} ({pick['id']})")
        print(f"   Category: {pick['category']}")
        print(f"   Price: ${pick['price']}")
        print(f"   Combined Score: {pick.get('combined_score', 'N/A'):.3f}")
        print(f"   Vector Score: {pick.get('vector_score', 'N/A'):.3f}")
        print(f"   Rule Score: {pick.get('rule_score', 'N/A')}")
    print("=" * 45) 