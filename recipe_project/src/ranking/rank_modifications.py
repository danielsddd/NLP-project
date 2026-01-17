"""
Ranking Module
==============
Rank extracted modifications by helpfulness.

Combines model confidence with social signals (likes) as specified in the proposal.

Formula: Score = α * Confidence + β * log(1 + Likes) + γ * Frequency

Usage:
    from src.ranking.rank_modifications import ModificationRanker
    
    ranker = ModificationRanker()
    ranked = ranker.rank_for_video(video_id, modifications)
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict


@dataclass
class RankedModification:
    """A modification with its ranking score."""
    # Core modification data
    span: str
    aspect: str
    
    # Source info
    comment_id: str
    video_id: str
    video_title: str = ""
    
    # Signals for ranking
    confidence: float = 0.0  # Model's softmax probability
    likes: int = 0           # Comment like count
    frequency: int = 1       # How often this modification appears
    
    # Computed score
    score: float = 0.0
    
    # Optional: full comment text for context
    comment_text: str = ""


@dataclass
class RankingConfig:
    """Configuration for the ranking formula."""
    # Weights (should sum to 1.0)
    alpha: float = 0.4  # Weight for model confidence
    beta: float = 0.3   # Weight for social signal (likes)
    gamma: float = 0.3  # Weight for frequency
    
    # Normalization
    normalize_frequency: bool = True  # Divide frequency by total mods
    log_transform_likes: bool = True  # Use log(1 + likes)
    
    # Thresholds
    min_confidence: float = 0.5  # Minimum confidence to include
    min_likes: int = 0           # Minimum likes to include


class ModificationRanker:
    """
    Ranks extracted modifications by helpfulness.
    
    Implements the ranking formula from the proposal:
    Score = α * Confidence + β * log(1 + Likes) + γ * Frequency
    
    Usage:
        ranker = ModificationRanker()
        
        # Rank modifications for a single video
        ranked = ranker.rank_for_video(video_id, modifications)
        
        # Rank all modifications across videos
        all_ranked = ranker.rank_all(modifications)
    """
    
    def __init__(self, config: Optional[RankingConfig] = None):
        self.config = config or RankingConfig()
    
    def compute_score(
        self,
        confidence: float,
        likes: int,
        frequency: int,
        total_modifications: int = 1
    ) -> float:
        """
        Compute the ranking score for a modification.
        
        Args:
            confidence: Model's softmax probability (0-1)
            likes: Number of likes on the source comment
            frequency: How many times this modification type appears
            total_modifications: Total modifications for normalization
            
        Returns:
            Ranking score (higher is better)
        """
        cfg = self.config
        
        # Confidence component (already 0-1)
        conf_score = confidence
        
        # Likes component (log transform to reduce impact of outliers)
        if cfg.log_transform_likes:
            likes_score = math.log(1 + likes) / math.log(1 + 1000)  # Normalize to ~1 for 1000 likes
        else:
            likes_score = min(likes / 100, 1.0)  # Cap at 100 likes
        
        # Frequency component
        if cfg.normalize_frequency and total_modifications > 0:
            freq_score = frequency / total_modifications
        else:
            freq_score = min(frequency / 10, 1.0)  # Cap at 10 occurrences
        
        # Weighted sum
        score = (
            cfg.alpha * conf_score +
            cfg.beta * likes_score +
            cfg.gamma * freq_score
        )
        
        return score
    
    def _compute_frequencies(
        self, 
        modifications: List[Dict]
    ) -> Dict[Tuple[str, str], int]:
        """
        Compute frequency of each modification type.
        
        Groups by (aspect, normalized_span) to find similar modifications.
        """
        freq = Counter()
        
        for mod in modifications:
            aspect = mod.get("aspect", "UNKNOWN")
            span = mod.get("span", "").strip().lower()
            
            # Create a key for grouping similar modifications
            # You could make this more sophisticated with fuzzy matching
            key = (aspect, span)
            freq[key] += 1
        
        return freq
    
    def rank_for_video(
        self,
        video_id: str,
        modifications: List[Dict],
        top_k: Optional[int] = None
    ) -> List[RankedModification]:
        """
        Rank modifications for a single video.
        
        Args:
            video_id: The video ID to filter by
            modifications: List of modification dicts with keys:
                - span: The modification text
                - aspect: SUBSTITUTION, QUANTITY, TECHNIQUE, or ADDITION
                - confidence: Model confidence (optional)
                - comment_id: Source comment ID
                - likes: Comment like count (optional)
            top_k: Return only top K results (None for all)
            
        Returns:
            List of RankedModification sorted by score (descending)
        """
        # Filter to this video
        video_mods = [m for m in modifications if m.get("video_id") == video_id]
        
        if not video_mods:
            return []
        
        # Compute frequencies
        frequencies = self._compute_frequencies(video_mods)
        total_mods = len(video_mods)
        
        # Score each modification
        ranked = []
        for mod in video_mods:
            aspect = mod.get("aspect", "UNKNOWN")
            span = mod.get("span", "").strip()
            
            # Skip if below confidence threshold
            confidence = mod.get("confidence", 0.5)
            if confidence < self.config.min_confidence:
                continue
            
            # Get frequency for this modification type
            key = (aspect, span.lower())
            frequency = frequencies.get(key, 1)
            
            # Get likes
            likes = mod.get("likes", mod.get("like_count", 0))
            if likes < self.config.min_likes:
                continue
            
            # Compute score
            score = self.compute_score(
                confidence=confidence,
                likes=likes,
                frequency=frequency,
                total_modifications=total_mods
            )
            
            # Create ranked modification
            ranked_mod = RankedModification(
                span=span,
                aspect=aspect,
                comment_id=mod.get("comment_id", ""),
                video_id=video_id,
                video_title=mod.get("video_title", ""),
                confidence=confidence,
                likes=likes,
                frequency=frequency,
                score=score,
                comment_text=mod.get("text", mod.get("comment_text", "")),
            )
            ranked.append(ranked_mod)
        
        # Sort by score (descending)
        ranked.sort(key=lambda x: x.score, reverse=True)
        
        # Return top K if specified
        if top_k is not None:
            ranked = ranked[:top_k]
        
        return ranked
    
    def rank_all(
        self,
        modifications: List[Dict],
        group_by_video: bool = True,
        top_k_per_video: Optional[int] = 5
    ) -> Dict[str, List[RankedModification]]:
        """
        Rank all modifications, optionally grouped by video.
        
        Args:
            modifications: All modifications across videos
            group_by_video: If True, return dict grouped by video_id
            top_k_per_video: Return only top K per video (None for all)
            
        Returns:
            Dict mapping video_id to list of ranked modifications
        """
        if not group_by_video:
            # Treat all as one group
            all_ranked = self.rank_for_video("all", modifications, top_k_per_video)
            return {"all": all_ranked}
        
        # Group by video
        by_video = defaultdict(list)
        for mod in modifications:
            video_id = mod.get("video_id", "unknown")
            by_video[video_id].append(mod)
        
        # Rank each video's modifications
        results = {}
        for video_id, video_mods in by_video.items():
            # Add video_id to each mod if not present
            for m in video_mods:
                m["video_id"] = video_id
            
            ranked = self.rank_for_video(video_id, video_mods, top_k_per_video)
            if ranked:
                results[video_id] = ranked
        
        return results
    
    def format_for_display(
        self, 
        ranked: List[RankedModification],
        include_context: bool = True
    ) -> str:
        """Format ranked modifications for display."""
        lines = []
        lines.append("=" * 60)
        lines.append("TOP RECIPE MODIFICATIONS")
        lines.append("=" * 60)
        
        for i, mod in enumerate(ranked, 1):
            lines.append(f"\n{i}. [{mod.aspect}] {mod.span}")
            lines.append(f"   Score: {mod.score:.3f} (conf={mod.confidence:.2f}, likes={mod.likes}, freq={mod.frequency})")
            
            if include_context and mod.comment_text:
                # Truncate long comments
                text = mod.comment_text[:100] + "..." if len(mod.comment_text) > 100 else mod.comment_text
                lines.append(f"   Context: \"{text}\"")
        
        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


class VideoModificationSummarizer:
    """
    Summarize modifications for a video to show the "wisdom of the crowd".
    
    This aggregates similar modifications and presents the most impactful ones.
    """
    
    def __init__(self, ranker: Optional[ModificationRanker] = None):
        self.ranker = ranker or ModificationRanker()
    
    def summarize_video(
        self,
        video_id: str,
        video_title: str,
        modifications: List[Dict],
        top_k: int = 5
    ) -> Dict:
        """
        Create a summary of modifications for a video.
        
        Returns:
            Dict with video info and top modifications by aspect
        """
        ranked = self.ranker.rank_for_video(video_id, modifications)
        
        # Group by aspect
        by_aspect = defaultdict(list)
        for mod in ranked:
            by_aspect[mod.aspect].append(mod)
        
        summary = {
            "video_id": video_id,
            "video_title": video_title,
            "total_modifications": len(modifications),
            "unique_modifications": len(ranked),
            "by_aspect": {},
            "top_overall": [],
        }
        
        # Top by aspect
        for aspect, mods in by_aspect.items():
            summary["by_aspect"][aspect] = {
                "count": len(mods),
                "top": [
                    {
                        "span": m.span,
                        "score": m.score,
                        "likes": m.likes,
                        "frequency": m.frequency,
                    }
                    for m in mods[:3]  # Top 3 per aspect
                ]
            }
        
        # Top overall
        summary["top_overall"] = [
            {
                "span": m.span,
                "aspect": m.aspect,
                "score": m.score,
                "likes": m.likes,
                "frequency": m.frequency,
            }
            for m in ranked[:top_k]
        ]
        
        return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Rank recipe modifications")
    parser.add_argument("--input", "-i", required=True, help="Input predictions file (JSONL)")
    parser.add_argument("--output", "-o", help="Output file for ranked results")
    parser.add_argument("--top-k", type=int, default=10, help="Top K per video")
    parser.add_argument("--alpha", type=float, default=0.4, help="Weight for confidence")
    parser.add_argument("--beta", type=float, default=0.3, help="Weight for likes")
    parser.add_argument("--gamma", type=float, default=0.3, help="Weight for frequency")
    
    args = parser.parse_args()
    
    # Load predictions
    modifications = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Flatten if needed
                if "modifications" in data:
                    for mod in data["modifications"]:
                        mod["video_id"] = data.get("video_id", "unknown")
                        mod["comment_id"] = data.get("comment_id", "")
                        mod["likes"] = data.get("like_count", data.get("likes", 0))
                        mod["text"] = data.get("text", "")
                        modifications.append(mod)
                else:
                    modifications.append(data)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(modifications)} modifications")
    
    # Configure ranker
    config = RankingConfig(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    ranker = ModificationRanker(config)
    
    # Rank all
    results = ranker.rank_all(modifications, top_k_per_video=args.top_k)
    
    # Display results
    for video_id, ranked in results.items():
        print(f"\n{'='*60}")
        print(f"Video: {video_id}")
        print(ranker.format_for_display(ranked[:args.top_k]))
    
    # Save if output specified
    if args.output:
        output_data = {
            video_id: [
                {
                    "span": m.span,
                    "aspect": m.aspect,
                    "score": m.score,
                    "confidence": m.confidence,
                    "likes": m.likes,
                    "frequency": m.frequency,
                }
                for m in ranked
            ]
            for video_id, ranked in results.items()
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()