"""
Teacher Labeling Module
=======================
Uses Gemini 1.5 Pro (or GPT-4o) to generate silver labels for training data.

This implements the "Teacher" in our Teacher-Student distillation approach.

Usage:
    python -m src.teacher_labeling.generate_labels --input data/raw_youtube/comments.jsonl
"""

import json
import os
import time
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from tqdm import tqdm

# Try to import Google AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Run: pip install google-generativeai")

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Modification:
    """A single extracted modification."""
    span: str                    # The exact text span
    aspect: str                  # SUBSTITUTION, QUANTITY, TECHNIQUE, ADDITION
    sentiment: str = "constructive"  # constructive or positive
    start_char: Optional[int] = None  # Character start index (if found)
    end_char: Optional[int] = None    # Character end index (if found)


@dataclass
class TeacherOutput:
    """Output from the Teacher model for a single comment."""
    modifications: List[Modification] = field(default_factory=list)
    has_modification: bool = False
    overall_sentiment: str = "neutral"
    raw_response: Optional[str] = None
    error: Optional[str] = None


@dataclass
class LabeledComment:
    """A comment with its silver labels from the Teacher."""
    # Original comment data
    comment_id: str
    video_id: str
    text: str
    like_count: int
    video_title: str
    channel_title: str
    
    # Teacher output
    teacher_output: TeacherOutput = None
    
    # Metadata
    labeled_at: str = ""
    teacher_model: str = ""


# =============================================================================
# PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert culinary NLP assistant specializing in Hebrew text analysis.
Your task is to analyze user comments from cooking videos and extract recipe modifications.

For each comment, identify ALL modification suggestions and extract:
1. The EXACT text span containing the modification (preserve original Hebrew text exactly)
2. The aspect category: SUBSTITUTION, QUANTITY, TECHNIQUE, or ADDITION
3. Whether it's constructive (suggestion/correction) or positive (praise)

ASPECT DEFINITIONS:
- SUBSTITUTION: Replacing one ingredient with another (e.g., "במקום חמאה השתמשתי בשמן")
- QUANTITY: Modifying amounts (e.g., "הוספתי יותר סוכר", "כפול שום")
- TECHNIQUE: Changing cooking method, time, or temperature (e.g., "אפיתי 10 דקות יותר")
- ADDITION: Adding ingredients not in original recipe (e.g., "הוספתי גם קינמון")

CRITICAL RULES:
1. A comment may contain MULTIPLE modifications - extract ALL of them
2. Extract the MINIMAL text span that captures the modification
3. If spans are non-contiguous, list each separately
4. Return empty modifications list [] if no modifications found
5. Preserve Hebrew text EXACTLY as written (including typos)
6. Do NOT include general praise like "מעולה" unless it's about a modification

OUTPUT FORMAT (respond ONLY with valid JSON, no markdown):
{
  "modifications": [
    {
      "span": "<exact text from comment>",
      "aspect": "SUBSTITUTION|QUANTITY|TECHNIQUE|ADDITION",
      "sentiment": "constructive|positive"
    }
  ],
  "has_modification": true|false,
  "overall_sentiment": "constructive|positive|neutral"
}"""


FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Comment: "עשיתי את העוגה והיא יצאה מדהימה! רק הוספתי עוד כפית סוכר כי אני אוהבת מתוק"

Output:
{
  "modifications": [
    {
      "span": "הוספתי עוד כפית סוכר",
      "aspect": "QUANTITY",
      "sentiment": "constructive"
    }
  ],
  "has_modification": true,
  "overall_sentiment": "constructive"
}

EXAMPLE 2:
Comment: "במקום חמאה השתמשתי בשמן קוקוס והוספתי גם קינמון - יצא מושלם!"

Output:
{
  "modifications": [
    {
      "span": "במקום חמאה השתמשתי בשמן קוקוס",
      "aspect": "SUBSTITUTION",
      "sentiment": "constructive"
    },
    {
      "span": "הוספתי גם קינמון",
      "aspect": "ADDITION",
      "sentiment": "constructive"
    }
  ],
  "has_modification": true,
  "overall_sentiment": "constructive"
}

EXAMPLE 3:
Comment: "אפיתי 35 דקות במקום 25 כי התנור שלי חלש יותר"

Output:
{
  "modifications": [
    {
      "span": "אפיתי 35 דקות במקום 25",
      "aspect": "TECHNIQUE",
      "sentiment": "constructive"
    }
  ],
  "has_modification": true,
  "overall_sentiment": "constructive"
}

EXAMPLE 4:
Comment: "וואו מתכון מדהים! תודה רבה!"

Output:
{
  "modifications": [],
  "has_modification": false,
  "overall_sentiment": "positive"
}

EXAMPLE 5:
Comment: "עשיתי עם פחות מלח, החלפתי את הכוסברה בפטרוזיליה ואפיתי בתנור במקום על הכיריים"

Output:
{
  "modifications": [
    {
      "span": "עם פחות מלח",
      "aspect": "QUANTITY",
      "sentiment": "constructive"
    },
    {
      "span": "החלפתי את הכוסברה בפטרוזיליה",
      "aspect": "SUBSTITUTION",
      "sentiment": "constructive"
    },
    {
      "span": "אפיתי בתנור במקום על הכיריים",
      "aspect": "TECHNIQUE",
      "sentiment": "constructive"
    }
  ],
  "has_modification": true,
  "overall_sentiment": "constructive"
}
"""


# =============================================================================
# TEACHER MODEL CLASSES
# =============================================================================

class GeminiTeacher:
    """Teacher model using Google's Gemini 1.5 Pro."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        
        # Configure generation
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=1024,
        )
    
    def generate(self, comment_text: str) -> TeacherOutput:
        """Generate silver labels for a comment."""
        prompt = f"""{SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLES}

Now analyze this comment:
Comment: "{comment_text}"

Output (JSON only, no markdown):"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            raw_response = response.text
            return self._parse_response(raw_response, comment_text)
            
        except Exception as e:
            return TeacherOutput(error=str(e))
    
    def _parse_response(self, response: str, original_text: str) -> TeacherOutput:
        """Parse the JSON response from Gemini."""
        try:
            # Clean response (remove markdown if present)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```json?\n?', '', cleaned)
                cleaned = re.sub(r'\n?```$', '', cleaned)
            
            data = json.loads(cleaned)
            
            modifications = []
            for mod in data.get("modifications", []):
                span = mod.get("span", "")
                
                # Find character indices
                start_char = original_text.find(span) if span else None
                end_char = start_char + len(span) if start_char is not None and start_char >= 0 else None
                
                modifications.append(Modification(
                    span=span,
                    aspect=mod.get("aspect", "UNKNOWN"),
                    sentiment=mod.get("sentiment", "constructive"),
                    start_char=start_char if start_char >= 0 else None,
                    end_char=end_char
                ))
            
            return TeacherOutput(
                modifications=modifications,
                has_modification=data.get("has_modification", len(modifications) > 0),
                overall_sentiment=data.get("overall_sentiment", "neutral"),
                raw_response=response
            )
            
        except json.JSONDecodeError as e:
            return TeacherOutput(
                error=f"JSON parse error: {e}",
                raw_response=response
            )


class OpenAITeacher:
    """Teacher model using OpenAI's GPT-4o."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai not installed")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def generate(self, comment_text: str) -> TeacherOutput:
        """Generate silver labels for a comment."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + FEW_SHOT_EXAMPLES},
                    {"role": "user", "content": f'Analyze this comment:\nComment: "{comment_text}"\n\nOutput (JSON only):'}
                ],
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            raw_response = response.choices[0].message.content
            return self._parse_response(raw_response, comment_text)
            
        except Exception as e:
            return TeacherOutput(error=str(e))
    
    def _parse_response(self, response: str, original_text: str) -> TeacherOutput:
        """Parse the JSON response from OpenAI."""
        # Same logic as Gemini
        try:
            data = json.loads(response)
            
            modifications = []
            for mod in data.get("modifications", []):
                span = mod.get("span", "")
                start_char = original_text.find(span) if span else None
                end_char = start_char + len(span) if start_char is not None and start_char >= 0 else None
                
                modifications.append(Modification(
                    span=span,
                    aspect=mod.get("aspect", "UNKNOWN"),
                    sentiment=mod.get("sentiment", "constructive"),
                    start_char=start_char if start_char >= 0 else None,
                    end_char=end_char
                ))
            
            return TeacherOutput(
                modifications=modifications,
                has_modification=data.get("has_modification", len(modifications) > 0),
                overall_sentiment=data.get("overall_sentiment", "neutral"),
                raw_response=response
            )
            
        except json.JSONDecodeError as e:
            return TeacherOutput(error=f"JSON parse error: {e}", raw_response=response)


# =============================================================================
# MAIN LABELING PIPELINE
# =============================================================================

class SilverLabelGenerator:
    """
    Main pipeline for generating silver labels.
    
    Usage:
        generator = SilverLabelGenerator(api_key="...", provider="gemini")
        generator.process_file("data/raw_youtube/comments.jsonl")
    """
    
    def __init__(
        self,
        api_key: str,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        output_dir: str = "data/silver_labels",
        delay_between_calls: float = 0.5,
        max_retries: int = 3,
    ):
        self.provider = provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay_between_calls
        self.max_retries = max_retries
        
        # Initialize teacher model
        if provider == "gemini":
            self.teacher = GeminiTeacher(api_key, model_name or "gemini-1.5-pro")
        elif provider == "openai":
            self.teacher = OpenAITeacher(api_key, model_name or "gpt-4o")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'gemini' or 'openai'")
        
        self.model_name = self.teacher.model_name
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def process_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        limit: Optional[int] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a JSONL file of comments and generate silver labels.
        
        Args:
            input_file: Path to input comments.jsonl
            output_file: Path to output (default: silver_labels/teacher_output.jsonl)
            limit: Maximum comments to process (None for all)
            skip_existing: Skip comments already in output file
        
        Returns:
            Statistics dictionary
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        output_path = Path(output_file) if output_file else self.output_dir / "teacher_output.jsonl"
        
        # Load existing IDs if skipping
        existing_ids = set()
        if skip_existing and output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        existing_ids.add(data.get("comment_id"))
                    except:
                        pass
            self.logger.info(f"Found {len(existing_ids)} existing labels, will skip them")
        
        # Load comments
        comments = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    comment = json.loads(line)
                    if comment.get("comment_id") not in existing_ids:
                        comments.append(comment)
                except json.JSONDecodeError:
                    continue
        
        if limit:
            comments = comments[:limit]
        
        self.logger.info(f"Processing {len(comments)} comments with {self.model_name}")
        
        # Process comments
        stats = {
            "total_processed": 0,
            "successful": 0,
            "errors": 0,
            "with_modifications": 0,
            "total_modifications": 0,
            "aspect_counts": {"SUBSTITUTION": 0, "QUANTITY": 0, "TECHNIQUE": 0, "ADDITION": 0},
            "started_at": datetime.now().isoformat(),
        }
        
        with open(output_path, 'a', encoding='utf-8') as f:
            for comment in tqdm(comments, desc="Generating silver labels"):
                result = self._process_comment(comment)
                
                # Write result
                output_data = {
                    "comment_id": comment.get("comment_id"),
                    "video_id": comment.get("video_id"),
                    "text": comment.get("text"),
                    "like_count": comment.get("like_count", 0),
                    "video_title": comment.get("video_title", ""),
                    "channel_title": comment.get("channel_title", ""),
                    "teacher_output": self._serialize_teacher_output(result),
                    "teacher_model": self.model_name,
                    "labeled_at": datetime.now().isoformat(),
                }
                
                f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
                f.flush()
                
                # Update stats
                stats["total_processed"] += 1
                if result.error:
                    stats["errors"] += 1
                else:
                    stats["successful"] += 1
                    if result.has_modification:
                        stats["with_modifications"] += 1
                        stats["total_modifications"] += len(result.modifications)
                        for mod in result.modifications:
                            if mod.aspect in stats["aspect_counts"]:
                                stats["aspect_counts"][mod.aspect] += 1
                
                # Rate limiting
                time.sleep(self.delay)
        
        stats["ended_at"] = datetime.now().isoformat()
        
        # Save stats
        stats_path = self.output_dir / "generation_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self._print_summary(stats)
        return stats
    
    def _process_comment(self, comment: Dict) -> TeacherOutput:
        """Process a single comment with retries."""
        text = comment.get("text", "")
        
        for attempt in range(self.max_retries):
            result = self.teacher.generate(text)
            
            if not result.error:
                return result
            
            if attempt < self.max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return result
    
    def _serialize_teacher_output(self, output: TeacherOutput) -> Dict:
        """Convert TeacherOutput to dictionary."""
        return {
            "modifications": [
                {
                    "span": m.span,
                    "aspect": m.aspect,
                    "sentiment": m.sentiment,
                    "start_char": m.start_char,
                    "end_char": m.end_char,
                }
                for m in output.modifications
            ],
            "has_modification": output.has_modification,
            "overall_sentiment": output.overall_sentiment,
            "error": output.error,
        }
    
    def _print_summary(self, stats: Dict):
        """Print summary of generation."""
        print("\n" + "=" * 60)
        print("SILVER LABEL GENERATION SUMMARY")
        print("=" * 60)
        print(f"  Total processed:     {stats['total_processed']}")
        print(f"  Successful:          {stats['successful']}")
        print(f"  Errors:              {stats['errors']}")
        print(f"  With modifications:  {stats['with_modifications']}")
        print(f"  Total modifications: {stats['total_modifications']}")
        print("\n  Aspect breakdown:")
        for aspect, count in stats['aspect_counts'].items():
            print(f"    {aspect}: {count}")
        print("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    from dotenv import load_dotenv
    
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Generate silver labels using Teacher model")
    parser.add_argument("--input", "-i", required=True, help="Input comments.jsonl file")
    parser.add_argument("--output", "-o", help="Output file (default: data/silver_labels/teacher_output.jsonl)")
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini", help="Teacher model provider")
    parser.add_argument("--api-key", help="API key (or use GOOGLE_API_KEY/OPENAI_API_KEY env var)")
    parser.add_argument("--limit", type=int, help="Limit number of comments to process")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing labels")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key
    if not api_key:
        if args.provider == "gemini":
            api_key = os.environ.get("GOOGLE_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print(f"❌ No API key provided!")
        print(f"   Set {'GOOGLE_API_KEY' if args.provider == 'gemini' else 'OPENAI_API_KEY'} environment variable")
        print(f"   Or use --api-key argument")
        return 1
    
    # Run generator
    generator = SilverLabelGenerator(
        api_key=api_key,
        provider=args.provider,
        delay_between_calls=args.delay,
    )
    
    generator.process_file(
        input_file=args.input,
        output_file=args.output,
        limit=args.limit,
        skip_existing=not args.no_skip,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
