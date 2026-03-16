## In addition to splitting the processing of text and images
# we also add context to the images, add post processing to the units and resolve the material names
## Optimized how we process the nodes to prevent duplicate processing of text
## Also added feature for table and figure detection in the sections (so we can add the context to the images)

## V4->v5: add limits on the token output (it was 104k token output lool)
## V7->V8: Adapted prompts & schema for polymer-ceramic composite electrolyte systems
##   - Added material_class field (Polymer/Ceramic/Composite); provenance in source field
##   - Rewrote few-shot examples for polymer-ceramic composites
##   - Improved cited vs measured data distinction
##   - Better handling of wt% loading-based naming
import os
import re
import argparse
import base64
import json
import time
import asyncio
import math
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Set
from pydantic import BaseModel, Field, ValidationError
import random
from google import genai
from google.genai import types
from dotenv import load_dotenv
import uuid
from dataclasses import dataclass, field, asdict
from scifigure_parser import SciFigureParser
import numpy as np


load_dotenv()


VISION_MODEL = "gemini-3-flash-preview"
TEXT_MODEL = "gemini-flash-latest"
NUM_WORKERS = 5
FILE_DIR = ""

PROCESS_TEXT = True
PROCESS_IMAGE = True
PROCESS_TABLE = True

try:
    import spacy
    from spacy.symbols import ORTH
    nlp = spacy.load("en_core_web_sm")
    
    # Add special cases to prevent splitting on abbreviations
    special_cases = ["Fig.", "Figs.", "Eq.", "Eqs.", "Tab.", "Tabs.", "Ref.", "Refs.", "al.", "vs.", "i.e.", "e.g."]
    for case in special_cases:
        nlp.tokenizer.add_special_case(case, [{ORTH: case}])
        
    SPACY_AVAILABLE = True
    print("✅ SpaCy loaded with custom tokenization rules.")
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    print("⚠️ SpaCy not found. Using Regex fallback.")



## Utils
# ============================================================================
# UTILITIES for Robust LLM Execution + Cost Tracking
# ============================================================================

def sanitize_caption(caption: str, max_length: int = 500) -> str:
    """
    Sanitize figure/table captions by collapsing repetitive patterns
    that arise from corrupted OCR (e.g., '(lacktriangle)La;' repeated 80x).
    Also truncates to a reasonable length for LLM context windows.
    """
    if not caption:
        return caption
    
    # 1. Collapse repeating n-grams (3-50 chars repeated 3+ times)
    #    This catches patterns like "(lacktriangle)La; (lacktriangle)La; ..."
    collapsed = re.sub(
        r'((.{3,50}?)\s*)\2{2,}',  # match a phrase repeated 3+ times
        r'\1...',  # keep one instance + ellipsis
        caption
    )
    
    # 2. Collapse runs of semicolons/commas with similar content
    #    e.g., "(▲)La; (▲)La; (▲)La;" -> "(▲)La; ..."
    collapsed = re.sub(
        r'((?:[;,]\s*\([^)]*\)[^;,]{1,20})){3,}',
        lambda m: m.group(0)[:60] + '...',
        collapsed
    )
    
    # 3. Truncate to max_length, breaking at last space
    if len(collapsed) > max_length:
        truncated = collapsed[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.5:
            truncated = truncated[:last_space]
        collapsed = truncated + '...'
    
    return collapsed.strip()

def classify_section_type(section_title: str) -> str:
    """Classify a section into a standard type based on its title."""
    title = section_title.lower().strip()
    
    # Order matters — more specific patterns first
    if any(kw in title for kw in ['method', 'experimental', 'synthesis', 'preparation', 'procedure', 'fabrication']):
        return 'methods'
    if any(kw in title for kw in ['result', 'performance', 'characterization', 'measurement']):
        return 'results'
    if any(kw in title for kw in ['discussion', 'analysis']):
        return 'discussion'
    if any(kw in title for kw in ['abstract', 'introduction', 'background', 'overview']):
        return 'intro'
    if any(kw in title for kw in ['conclusion', 'summary', 'outlook']):
        return 'conclusion'
    if any(kw in title for kw in ['acknowledg', 'reference', 'supplementar', 'appendix']):
        return 'meta'
    return 'other'

def number_paragraphs(text: str) -> tuple:
    """
    Split text into numbered paragraphs for provenance tracking.
    Returns (numbered_text, paragraph_list) where numbered_text has [P1], [P2], etc. labels.
    """
    # Split on double newlines (paragraph boundaries)
    raw_paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip() and len(p.strip()) > 20]
    
    if not paragraphs:
        return text, []
    
    numbered_parts = []
    for i, para in enumerate(paragraphs):
        numbered_parts.append(f"[P{i+1}] {para}")
    
    return "\n\n".join(numbered_parts), paragraphs

class CostTracker:
    # Pricing Estimates (USD per 1M tokens) - Update as pricing changes
    # Logic: Defaults to standard Pro/Flash tiers if exact model string isn't found
    PRICING_TIERS = [
        # Model Substring          Input Cost    Output Cost
        ("2.5-flash-lite",       {"input": 0.10, "output": 0.40}),
        ("2.5-flash",            {"input": 0.30, "output": 2.50}),
        ("2.5-pro",              {"input": 1.25, "output": 10.00}),
        ("3-flash",              {"input": 0.50, "output": 3.00}), # Matches "gemini-3-flash-preview"
        ("3-pro",                {"input": 2.00, "output": 12.00}), # Matches "gemini-3-pro-preview"
    ]

    # Fallback for older models (Gemini 2.5 Flash, etc.) if needed
    DEFAULT_PRICING = {"input": 0.30, "output": 2.50}

    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.call_counts = {}

    def track(self, response, model_name: str):
        """Parses response metadata and accumulates cost."""
        if not response:
            return

        # Handle diverse input types (Gemini response object vs Dict from SciFigureParser)
        usage = None
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
        elif isinstance(response, dict) and '_usage_metadata' in response:
            usage = response['_usage_metadata']
        
        if not usage:
            return

        # Extract counts based on type
        if isinstance(usage, dict):
            in_tok = usage.get('prompt_token_count', 0)
            out_tok = usage.get('candidates_token_count', 0)
            # Some APIs might not return candidates_token_count if it failed?
            if out_tok == 0 and 'total_token_count' in usage:
                out_tok = usage['total_token_count'] - in_tok
        else:
            in_tok = usage.prompt_token_count or 0
            try:
                out_tok = usage.total_token_count - in_tok or 0
            except: 
                print("Failed to calculate output tokens")
                print(usage)
                out_tok = usage.candidates_token_count if hasattr(usage, 'candidates_token_count') else 0
        
        
        # 2. Determine Pricing Tier
        model_name_lower = model_name.lower()
        pricing = self.DEFAULT_PRICING
        
        for substring, prices in self.PRICING_TIERS:
            if substring in model_name_lower:
                pricing = prices
                break # Stop at first match (vital for lite vs flash)
            
        # 3. Calculate Cost (Price per 1M tokens)
        cost = (in_tok / 1_000_000 * pricing["input"]) + \
               (out_tok / 1_000_000 * pricing["output"])

        # 4. Update Totals
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self.total_cost_usd += cost
        
        # Track calls per model
        self.call_counts[model_name] = self.call_counts.get(model_name, 0) + 1

    def print_summary(self):
        print("\n" + "="*50)
        print("💰 PIPELINE COST SUMMARY")
        print("="*50)
        print(f"{'Total Calls:':<20} {sum(self.call_counts.values())}")
        print(f"{'Total Input:':<20} {self.total_input_tokens:,} tokens")
        print(f"{'Total Output:':<20} {self.total_output_tokens:,} tokens")
        print("-" * 50)
        print(f"{'TOTAL COST:':<20} ${self.total_cost_usd:.4f}")
        print("-" * 50)
        print("Breakdown by Model:")
        for model, count in self.call_counts.items():
            print(f"  - {model:<30}: {count} calls")
        print("="*50 + "\n")

# Global singleton instance
tracker = CostTracker()

async def safe_text_call_with_retry(sec, client, model_name, sem, timeout=150, max_retries=3, paper_context=None):
    async with sem:
        for attempt in range(max_retries):
            try:
                # Add a timeout to the specific model call
                result, raw_response, success = await asyncio.wait_for(
                    process_text(client, model_name, sec.content, sec.title, paper_context=paper_context),
                    timeout=timeout
                )
                
                if raw_response:
                    tracker.track(raw_response, model_name)
                return result, success

            except asyncio.TimeoutError:
                print(f"   \033[93m[Timeout]\033[0m {sec.title} (Attempt {attempt+1})")
                if attempt == max_retries - 1: return None, False
                
            except Exception as e:
                err_str = str(e).lower()
                if "503" in err_str or "overloaded" in err_str:
                    wait_time = (2 ** attempt) + random.random()
                    print(f"   [Retry] {sec.title} - Model overloaded. Waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"   \033[91m[Text Error]\033[0m {sec.title}: {e}")
                    break 
        return None, False

async def safe_image_call_with_retry(img_path, context, client, model_name, sem, sf_parser=None, timeout=180, max_retries=3):
    """
    This function now only retries on ACTUAL failures, not empty results.
    """
    async with sem:
        try:
            result, raw_response, success = await asyncio.wait_for(
                process_image(client, model_name, img_path, context, sf_parser=sf_parser), 
                timeout=timeout
            )
            
            if raw_response:
                tracker.track(raw_response, model_name)
            
            # Return regardless of success - we'll handle failures upstream
            return result, success
            
        except asyncio.TimeoutError:
            print(f"   ⏱️  {img_path.name}: Timeout")
            return [], False
            
async def safe_table_call_with_retry(table_data, client, model_name, sem, timeout=180, max_retries=3, paper_context=None):
    async with sem:
        for attempt in range(max_retries):
            try:
                result, raw_response, success = await asyncio.wait_for(
                    process_table_node(client, model_name, table_data, paper_context=paper_context), 
                    timeout=timeout
                )
                
                if raw_response:
                    tracker.track(raw_response, model_name)
                return result, success

            except asyncio.TimeoutError:
                print(f"   \033[93m[Timeout]\033[0m {table_data['caption']} (Attempt {attempt+1})")
                if attempt == max_retries - 1: return None, False
                
            except Exception as e:
                err_str = str(e).lower()
                if "503" in err_str or "overloaded" in err_str:
                    wait_time = (2 ** (attempt + 1)) + random.random()
                    print(f"   [Retry] {table_data['caption']} - Model overloaded. Waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"   \033[91m[Table Error]\033[0m {table_data['caption']}: {e}")
                    break 
    return ExtractionResult(measurements=[]), False
# ==============================================================================
# 1. Data Schema (Enhanced)
# ==============================================================================
@dataclass(frozen=True)
class ImageInfo:
    filename: str
    id: str          # e.g. "Fig 1"
    caption: str
    line_index: int

@dataclass(frozen=True)
class TableInfo:
    id: str          # e.g. "Table 1"
    content: str     # The raw markdown table text
    caption: str     # Found above/below table
    line_index: int

@dataclass
class SectionInfo:
    title: str
    content: str
    line_num: int
    end_line_num: int
    id: str
    # The section "owns" the assets discussed within it
    images: List[ImageInfo] = field(default_factory=list)
    tables: List[TableInfo] = field(default_factory=list)


class MeasuredPoint(BaseModel):
    # We capture the "Raw" string for provenance, and "Normalized" for database
    raw_composition: str = Field(..., description="The name as it appears in the source (e.g. 'PEO-LiTFSI/LLZO (50 wt%)', 'x=0.1', 'Sample A').")
    
    # We will fill these in via Post-Processing
    canonical_formula: Optional[str] = Field(None, description="Normalized chemical formula (e.g. 'PEO-LiTFSI/Li6.4Al0.2La3Zr2O12 (50 wt%)', 'Li3.8Mg0.1Ti1.63O4').")
    reduced_formula: Optional[str] = Field(None, description="GCD-reduced formula for matching (e.g. 'Al3Li13O120P30Ti17').")
    
    # V8: Material classification
    material_class: Optional[str] = Field(
        None, 
        description="Classification of the material system: "
                    "'Polymer' (neat polymer electrolyte, e.g. PEO-LiTFSI), "
                    "'Ceramic' (neat ceramic electrolyte, e.g. LLZO, LAGP), "
                    "'Composite' (polymer+ceramic composite, e.g. PEO-LiTFSI/LLZO). "
                    "Use the material's actual type even if the data was cited from another reference — "
                    "provenance is tracked separately in the 'source' field (cited_text / cited_table)."
    )

    material_definitions: List[str] = Field(
        default_factory=list, 
        description="Brief 4-5 concise sentences describing the material (e.g. 'Composite polymer electrolyte with 50 wt% LLZO nanofibers in PEO-LiTFSI matrix')."
    )

    raw_conductivity: str = Field(..., description="Ionic conductivity value as extracted (e.g. '1.24e-4', '5.2').")
    raw_conductivity_unit: str = Field(..., description="Corresponding ionic conductivity unit as extracted (e.g. 'mS/cm', 'S cm-1').")
    normalized_conductivity: Optional[float] = Field(None, description="Normalized ionic conductivity value in S/cm.")
    
    raw_temperature: str = Field(
        ..., 
        description="The numeric value ONLY. Example: if axis says '1000/T = 3.35', extract '3.35'."
    )
    raw_temperature_unit: str = Field(
        ..., 
        description="The unit or axis label exactly as shown. Example: 'C', 'K', '1000/T (K-1)', '10^3/T'."
    )

    normalized_temperature_c: Optional[float] = Field(None, description="Temperature in Celsius.")
    
    source_figure_id: Optional[str] = Field(None, description="The real Figure ID (e.g. 'Fig. 5') if known.")
    source_caption: Optional[str] = Field(None, description="The context from the figure caption.")
    source: str = Field(..., description="The source of the data. Use 'text', 'table', 'figure' for original data, or 'cited_text', 'cited_table', 'cited_figure' for data cited from other works.")
    source_image_filename: Optional[str] = Field(None, description="The actual image filename (e.g. 'image_003.png') for figure-sourced measurements.")
    processing_method: Optional[str] = Field(None, description="The processing method used to obtain the sample (e.g. 'slot-die coating', 'solution casting', 'sol-gel', 'electrospinning').")
    processing_method_detail: Optional[str] = Field(None, description="Detailed processing description with parameters (temps, atmospheres, durations).")
    
    # V6 Provenance Fields
    source_section: Optional[str] = Field(None, description="Section title where this measurement was extracted (e.g. 'Introduction', 'Results').")
    source_section_type: Optional[str] = Field(None, description="Classification of source section: intro, methods, results, discussion, conclusion, meta, other.")
    source_paragraph_indices: List[int] = Field(default_factory=list, description="Paragraph indices [P1, P2, ...] within the section that sourced this measurement.")
    source_paragraph_id: Optional[str] = Field(None, description="Human-readable paragraph reference (e.g., 'P3', 'P1,P2') for quick reviewer lookup.")
    source_paragraph_text: List[str] = Field(default_factory=list, description="The actual text content of the paragraphs listed in source_paragraph_indices.")
    
    # V8.1: Temporal/condition fields
    aging_time: Optional[str] = Field(None, description="Storage/aging duration if applicable (e.g., '5 days', '2 weeks', 'freshly made').")
    measurement_condition: Optional[str] = Field(None, description="Special measurement condition (e.g., 'freshly made', 'after 100 cycles', 'in air', 'after aging').")

    # V8.2: ML exclusion flag (Fix #6)
    exclude_from_ml: bool = Field(
        False,
        description="If True, this measurement should be excluded from ML/tier screening. "
                    "Set for cited data from introductions/reviews lacking processing metadata."
    )

    confidence: str = Field(..., description="high/medium/low")
    warnings: List[str] = Field(default_factory=list, description="Warnings about the data.")

class ExtractionResult(BaseModel):
    measurements: List[MeasuredPoint]
    # We also extract "Material Definitions" from text to help us resolve "x=0.1" later
    

# ==============================================================================
# 2. Context Parsing (Solves Problem 1: Context Loss)
# ==============================================================================
class MarkdownContextParser:
    # --- IMPROVED REGEX ---
    # Matches: Fig 1, Fig. 1, Fig 1a, Fig 1(a), Figure 1(a), etc.
    # Logic:
    # 1. Prefix: Fig/Figure/Tab/Table (case insensitive)
    # 2. Separator: Optional dot + optional space
    # 3. Number: \d+
    # 4. Suffix: Optional letter (a) OR parenthesized letter (a)
    REF_PATTERN = re.compile(r'\b(Fig(?:\.|ure)?|Tab(?:\.|le)?)\s*(\d+)(?:[\s-]?(\(?[a-zA-Z]\)?))?', re.IGNORECASE)

    def parse_structure(self, md_text: str) -> Tuple[str, List[SectionInfo]]:
        """Parses headers to build document sections."""
        lines = md_text.split('\n')
        headers = []
        in_code_block = False
        header_pattern = re.compile(r'^(#{1,6})\s+(.*?)(?:\s+#+)?$')

        for i, line in enumerate(lines):
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block: continue
            
            match = header_pattern.match(line.strip())
            if match:
                headers.append({'level': len(match.group(1)), 'title': match.group(2).strip(), 'line_num': i})

        doc_title = "Untitled Document"
        sections: List[SectionInfo] = []

        # Helper to create section
        def create_section(title, start, end):
            content = "\n".join(lines[start:end]).strip()
            if content:
                return SectionInfo(title=title, content=content, line_num=start, end_line_num=end, id=str(uuid.uuid4()))
            return None

        if not headers:
            if not md_text.strip(): return doc_title, []
            return doc_title, [create_section("Full Text", 0, len(lines))]

        # Get Doc Title
        h1 = next((h for h in headers if h['level'] == 1), headers[0])
        doc_title = h1['title']
        
        slice_points = [h['line_num'] for h in headers] + [len(lines)]

        # Pre-header (Intro)
        if headers[0]['line_num'] > 0:
            sections.append(create_section("Introduction", 0, headers[0]['line_num']))

        # Main Sections
        for i, header in enumerate(headers):
            start = header['line_num'] + 1
            end = slice_points[i+1]
            title = "Abstract / Introduction" if (header['level'] == 1 and header['title'] == doc_title) else header['title']
            sec = create_section(title, start, end)
            if sec: sections.append(sec)

        return doc_title, sections


    def _normalize_id(self, prefix: str, number: str, suffix: str = None) -> str:
        """
        Standardizes IDs so "Fig. 4(a)" and "Figure 4a" both become "Fig 4a".
        """
        # Normalize Prefix
        clean_prefix = "Table" if "tab" in prefix.lower() else "Fig"
        
        # Normalize Suffix (remove parens, lower case)
        clean_suffix = ""
        if suffix:
            clean_suffix = suffix.replace('(', '').replace(')', '').strip().lower()
            
        return f"{clean_prefix} {number}{clean_suffix}"

    def _extract_nearby_text(self, img: ImageInfo, section_text: str, window_lines=5, max_chars=1000) -> str:
        """
        Extract text near the image reference for context.
        
        Args:
            img: ImageInfo object with line_index
            section_text: Full section content
            window_lines: How many lines above/below to include
            max_chars: Maximum characters to return
            
        Returns:
            Truncated nearby text
        """
        lines = section_text.split('\n')
        
        # Calculate window bounds
        start_idx = max(0, img.line_index - window_lines)
        end_idx = min(len(lines), img.line_index + window_lines + 1)
        
        # Extract nearby lines
        nearby_lines = lines[start_idx:end_idx]
        nearby_text = '\n'.join(nearby_lines).strip()
        
        # Truncate to max_chars
        if len(nearby_text) > max_chars:
            nearby_text = nearby_text[:max_chars] + "..."
        
        return nearby_text
    

    def parse_images(self, text: str) -> List[ImageInfo]:
        """Extracts images and nearby captions."""
        images = []
        lines = text.split('\n')
        img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

        for i, line in enumerate(lines):
            img_match = img_pattern.search(line)
            if img_match:
                filename = Path(img_match.group(1)).name
                caption, fig_id = "No caption found", "Unknown"
                
                # Context Search (Look Ahead 5, Behind 5)
                for direction in [1, -1]:
                    for j in range(1, 6):
                        idx = i + (j * direction)
                        if 0 <= idx < len(lines):
                            # Search line for "Fig X" pattern
                            # We use search() instead of match() to find it anywhere in the line
                            cap_match = self.REF_PATTERN.search(lines[idx])
                            if cap_match:
                                # Standardize: "Fig. 4(a)" -> "Fig 4a"
                                fig_id = self._normalize_id(cap_match.group(1), cap_match.group(2), cap_match.group(3))
                                caption = sanitize_caption(lines[idx].strip()) # Sanitize caption
                                break
                    if fig_id != "Unknown": break

                images.append(ImageInfo(filename, fig_id, caption, i))
        return images

    def parse_tables(self, text: str) -> List[TableInfo]:
        """Extracts Markdown tables and nearby captions."""
        tables = []
        lines = text.split('\n')
        
        # Regex for table caption: "Table 1: Results"
        # Updated Regex:
        # 1. Removed ^ anchor and added support for leading HTML tags
        # 2. Made the capture of the description more robust
        tab_cap_pattern = re.compile(
            r'(?:<[^>]+>)?\s*(?:\*\*|#+)?\s*(Table|Tab\.?)\s*(\d+[a-z]?)\s*[:\.]?\s*(.*)', 
            re.IGNORECASE
        )

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Detect start of table (must start with |)
            if line.startswith('|'):
                start_line = i
                # Consume table lines
                table_lines = []
                while i < len(lines) and lines[i].strip().startswith('|'):
                    table_lines.append(lines[i])
                    i += 1
                
                # Search for caption (Look Behind 5, Ahead 5)
                caption, tab_id = "No caption found", "Unknown"
                search_indices = list(range(start_line - 5, start_line)) + list(range(i, i + 5))
                
                for idx in search_indices:
                    if 0 <= idx < len(lines):
                        cap_match = tab_cap_pattern.search(lines[idx])
                        if cap_match:
                            tab_id = f"Table {cap_match.group(2)}" # Normalize
                            caption = cap_match.group(3).strip()
                            break
                            
                tables.append(TableInfo(tab_id, "\n".join(table_lines), caption, start_line))
            else:
                i += 1
        return tables
    def _find_references_robust(self, text: str) -> Set[str]:
        """
        Unified extraction that works for both Regex and SpaCy modes.
        Directly scanning text is often more robust for IDs than token-walking.
        """
        refs = set()
        matches = self.REF_PATTERN.findall(text)
        
        for prefix, number, suffix in matches:
            # Normalize found ref to match image IDs
            # e.g. Found "Fig. 4(a)" -> Normalizes to "Fig 4a"
            norm_id = self._normalize_id(prefix, number, suffix)
            refs.add(norm_id)
            
            # OPTIONAL: Also add the "parent" ID. 
            # If text says "Fig 4a", also link "Fig 4" just in case the image is labeled "Fig 4"
            if suffix:
                 refs.add(self._normalize_id(prefix, number, None))
                 
        return refs

    def _find_references_spacy(self, text: str) -> Set[str]:
        """Uses SpaCy to find 'Fig 1', 'Table 2' references in text."""
        doc = nlp(text)
        refs = set()
        
        # Iterate through tokens to find patterns
        for i, token in enumerate(doc):
            # Check for "Fig", "Figure", "Table", "Tab"
            t_text = token.text.replace('.', '') # Handle "Fig." -> "Fig"
            
            if t_text in ["Fig", "Figure", "Table", "Tab"] and i + 1 < len(doc):
                next_token = doc[i+1]
                # Check if next token is a number (e.g., "1", "2a")
                # We use a simple regex check on the token text to allow "1a"
                if re.match(r'^\d+[a-z]?$', next_token.text):
                    # Standardize output
                    prefix = "Table" if t_text.startswith("Tab") else "Fig"
                    refs.add(f"{prefix} {next_token.text}")
                    
        return refs

    def _find_references_regex(self, text: str) -> Set[str]:
        """Fallback Regex if SpaCy is missing."""
        refs = set()
        # Pattern: (Fig|Table) [dot?] (Number)
        pattern = re.compile(r'\b(Fig|Figure|Tab|Table)\.?\s+(\d+[a-z]?)', re.IGNORECASE)
        matches = pattern.findall(text)
        for type_str, num in matches:
            prefix = "Table" if type_str.lower().startswith("tab") else "Fig"
            refs.add(f"{prefix} {num}")
        return refs

    def build_figure_id_map(self, images: List[ImageInfo], sections: List[SectionInfo]) -> Dict[str, str]:
        """
        Fix #4: Build a mapping from image filename → manuscript figure number.

        Strategy:
        1. Use caption-based matching: each ImageInfo has a fig_id parsed from its caption.
        2. For images with fig_id == "Unknown", attempt order-based heuristic within sections.

        Returns:
            Dict mapping lowercase filename → manuscript figure ID (e.g., "image_003.png" → "Fig 1d")
        """
        figure_id_map = {}

        # Pass 1: Caption-based mapping (high confidence)
        for img in images:
            if img.id and img.id != "Unknown":
                figure_id_map[img.filename.lower()] = img.id

        # Pass 2: For unknowns, try to find figure references in section text near the image
        unknown_images = [img for img in images if img.id == "Unknown"]
        for img in unknown_images:
            # Search the section that contains this image for figure references
            for sec in sections:
                if sec.line_num <= img.line_index < sec.end_line_num:
                    # Look for figure references in the nearby text
                    sec_lines = sec.content.split('\n')
                    relative_line = img.line_index - sec.line_num
                    start = max(0, relative_line - 3)
                    end = min(len(sec_lines), relative_line + 3)
                    nearby = '\n'.join(sec_lines[start:end])

                    ref_match = self.REF_PATTERN.search(nearby)
                    if ref_match and 'fig' in ref_match.group(1).lower():
                        fig_id = self._normalize_id(ref_match.group(1), ref_match.group(2), ref_match.group(3))
                        figure_id_map[img.filename.lower()] = fig_id
                    break

        # Pass 3: Order-based fallback for remaining unknowns
        # Only figures (not tables), sorted by document order
        fig_images = sorted(
            [img for img in images if img.filename.lower() not in figure_id_map],
            key=lambda x: x.line_index
        )
        # Find the highest figure number already assigned
        assigned_nums = set()
        for fid in figure_id_map.values():
            num_match = re.search(r'(\d+)', fid)
            if num_match:
                assigned_nums.add(int(num_match.group(1)))

        # Assign sequentially for remaining (low confidence)
        next_num = max(assigned_nums) + 1 if assigned_nums else 1
        for img in fig_images:
            figure_id_map[img.filename.lower()] = f"Fig {next_num} (inferred)"
            next_num += 1

        return figure_id_map

    def link_assets_to_sections(self, sections: List[SectionInfo], images: List[ImageInfo], tables: List[TableInfo]):
        """
        Links Images and Tables to Sections based on text references.
        """
        # Create lookups
        img_lookup = {img.id.lower(): img for img in images}
        tab_lookup = {tab.id.lower(): tab for tab in tables}

        for section in sections:
            # Use the robust regex scanner on the full section text
            # This bypasses tokenization issues with "(a)" completely
            refs = self._find_references_robust(section.content)
            
            # 2. Link Assets
            for ref in refs:
                ref_lower = ref.lower() # e.g. "fig 1" or "table 2"
                
                if ref_lower in img_lookup:
                    img = img_lookup[ref_lower]
                    if img not in section.images:
                        section.images.append(img)
                        
                elif ref_lower in tab_lookup:
                    tab = tab_lookup[ref_lower]
                    if tab not in section.tables:
                        section.tables.append(tab)

        # 3. Handle Orphans (Assets never mentioned in text)
        # Fallback to physical location
        linked_imgs = {img for sec in sections for img in sec.images}
        linked_tabs = {tab for sec in sections for tab in sec.tables}
        
        for img in images:
            if img not in linked_imgs:
                for sec in sections:
                    if sec.line_num <= img.line_index < sec.end_line_num:
                        sec.images.append(img); break
                        
        for tab in tables:
            if tab not in linked_tabs:
                for sec in sections:
                    if sec.line_num <= tab.line_index < sec.end_line_num:
                        sec.tables.append(tab); break

# ==============================================================================
# 3. Normalizer Logic (Solves Problem 2: Normalization)
# ==============================================================================
def calculate_standard_units(cond_val: str, cond_unit: str, temp_val: str, temp_unit: str) -> dict:
    """
    Robust normalizer using the split Value/Unit fields.
    Handles non-numeric values like "room temperature" or "RT".
    """
    def safe_float(val: str) -> float:
        if not val:
            raise ValueError("Empty value")
        
        # Clean string
        clean = str(val).lower().strip().replace(',', '')
        
        if clean in ['null', 'none', 'n/a', 'unknown', 'not specified']:
             raise ValueError(f"Null or N/A value: {val}")

        # Handle "room temperature" and variations
        if clean in ["room temperature", "rt", "room temp", "room-temperature", "ambient", "ambient temperature", "ambient temp"]:
            return 25.0
            
        # Try direct conversion
        try:
            return float(clean)
        except ValueError:
            # Try to extract the first number
            import re
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", clean)
            if match:
                return float(match.group())
            raise ValueError(f"Could not convert '{val}' to float")

    try:
        # --- 1. Temperature Normalization ---
        # Get raw value, handling room temperature and known non-numeric strings
        temp_val_clean = str(temp_val).lower().strip()
        if temp_val_clean in ["n/a", "none", "unknown", "arrhenius plot", "not specified", "null", "ambient"]:
             if temp_val_clean in ["ambient"]:
                  # Pass it down to be handled as 25.0
                  pass
             else:
                  return {"cond": None, "temp": None}

        raw_t = safe_float(temp_val)
        unit_clean = temp_unit.lower().strip()
        
        # If it was "RT", force Celsius unit if not already set specifically
        if temp_val_clean in ["room temperature", "rt", "room temp", "room-temperature"]:
            if not unit_clean or unit_clean == "celsius":
                unit_clean = "c" # Force Celsius path

        temp_k = None      # Kelvin (needed for conductivity calc)
        norm_temp_c = None # Celsius (for DB)
        
        # Logic Branch: Composition vs Arrhenius
        # If the extracted "temperature" is actually a small number (x < 1.0) and unit is ambiguous,
        # it is likely a Composition value (x in Li...x...), NOT temperature.
        # In this case, we assume Room Temperature (25 C).
        # ADDED: Check for common stoichiometry labels in unit string
        if (raw_t < 1.0 or any(m in unit_clean for m in ['x=', 'z=', 'y='])) and "k" not in unit_clean and "c" not in unit_clean:
             norm_temp_c = 25.0
             temp_k = 298.15
        
        # CHECK 1: Is this an Arrhenius inverse scale?
        # CHECK 1: Is this an Arrhenius inverse scale?
        if ("1000" in unit_clean or "10^3" in unit_clean) and "t" in unit_clean:
            if raw_t > 0:
                temp_k = 1000.0 / raw_t
                norm_temp_c = temp_k - 273.15
        
        # CHECK 1b: Implicit Arrhenius (Unit is just K-1 but values are 1000/T range)
        elif ("k-1" in unit_clean or "1/k" in unit_clean) and 0.2 < raw_t < 10.0:
             # Heuristic: 1000/T usually falls between 0.5 (2000K) and 5.0 (200K)
             # If it were really 1/T, values would be ~0.001 - 0.005
             temp_k = 1000.0 / raw_t
             norm_temp_c = temp_k - 273.15

        # CHECK 2: Standard Kelvin
        elif "k" in unit_clean and "c" not in unit_clean: 
             temp_k = raw_t
             norm_temp_c = temp_k - 273.15
             
        # CHECK 3: Standard Celsius
        elif "c" in unit_clean:
            norm_temp_c = raw_t
            temp_k = raw_t + 273.15
            
        # Fallback: Guess based on magnitude
        else:
            if raw_t > 200: # Likely Kelvin
                temp_k = raw_t
                norm_temp_c = raw_t - 273.15
            else: # Likely Celsius
                norm_temp_c = raw_t
                temp_k = raw_t + 273.15

        # --- 2. Conductivity Normalization ---
        cond_val_clean = str(cond_val).lower().strip()
        if cond_val_clean in ["n/a", "none", "unknown", "not specified", "null"]:
            return {"cond": None, "temp": norm_temp_c}

        raw_c = safe_float(cond_val)
        cond_u_clean = cond_unit.lower().strip()
        norm_cond = None

        if "log" in cond_u_clean:
            # Case A: log(Sigma * T)
            if ("t" in cond_u_clean) and temp_k:
                sigma_times_t = 10 ** raw_c
                norm_cond = sigma_times_t / temp_k
            # Case B: just log(Sigma)
            else:
                norm_cond = 10 ** raw_c
            
        elif "ln" in cond_u_clean:
            import math
            # Case A: ln(Sigma * T)
            if ("t" in cond_u_clean) and temp_k:
                sigma_times_t = math.exp(raw_c)
                norm_cond = sigma_times_t / temp_k
            # Case B: ln(Sigma)
            else:
                 norm_cond = math.exp(raw_c)
            
        else:
            # Standard Linear Units
            multiplier = 1.0
            if "ms" in cond_u_clean: multiplier = 1e-3
            elif "us" in cond_u_clean: multiplier = 1e-6
            elif "ns" in cond_u_clean: multiplier = 1e-9
            
            # Geometry fix (S/m -> S/cm)
            if "m" in cond_u_clean and "cm" not in cond_u_clean:
                if "m-1" in cond_u_clean or "/m" in cond_u_clean:
                    multiplier *= 0.01

            norm_cond = raw_c * multiplier

        return {"cond": norm_cond, "temp": round(norm_temp_c, 2) if norm_temp_c is not None else None}

    except Exception as e:
        # Only log if it's not a known non-numeric string that somehow got through
        if "could not convert" in str(e).lower() and any(x in str(e) for x in ["'N/A'", "'Arrhenius plot'"]):
            pass
        else:
            print(f"Norm Error: {e}")
        return {"cond": None, "temp": None}

    # --- 3. Activation Energy Filter ---
    # Convert extracted unit to lower case for check
    cond_u_clean_final = cond_unit.lower().strip()
    if any(x in cond_u_clean_final for x in ['ev', 'kj', 'joule', 'mol']):
        return {"cond": None, "temp": None} # discard activation energy

    return {"cond": norm_cond, "temp": round(norm_temp_c, 2) if norm_temp_c is not None else None}

# ==============================================================================
# 4. Canonicalizer (Solves Problem 3: Useless Names)
# ==============================================================================

def validate_formula_stoichiometry(formula: str) -> bool:
    """
    Validate that a chemical formula has physically plausible stoichiometry.
    
    Checks:
    1. Sum of all element coefficients is reasonable
    2. Individual element coefficients are in realistic ranges
    
    Returns True if plausible, False otherwise.
    """
    if not formula or formula == "null":
        return False
    
    # Extract all element-coefficient pairs
    # Pattern: Element (uppercase + optional lowercase) followed by optional number
    pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    
    if not matches:
        return False
    
    total_atoms = 0
    element_counts = {}
    
    for element, coeff_str in matches:
        coeff = float(coeff_str) if coeff_str else 1.0
        element_counts[element] = element_counts.get(element, 0) + coeff
        total_atoms += coeff
    
    # Physical plausibility checks
    # 1. Total atoms should be reasonable (typically 5-150 for solid electrolytes)
    if total_atoms < 3 or total_atoms > 200:
        return False
    
    # 2. Individual element checks
    for element, count in element_counts.items():
        # Li: typically 3-30
        if element == 'Li' and (count < 0.1 or count > 50):
            return False
        # O: typically 10-100
        if element == 'O' and (count < 1 or count > 150):
            return False
        # Other elements: typically 0.1-20
        if element not in ['Li', 'O'] and (count < 0.05 or count > 30):
            return False
    
    return True


def normalize_formula_to_reduced(formula: str) -> str:
    """
    Reduce a chemical formula to its simplest integer-ratio representation.
    
    Handles both flat formulas and formulas with parenthesized groups:
    - Li1.3Al0.3Ti1.7(PO4)3 -> Li13Al3Ti17P30O120 (×10 to clear decimals)
    - Li7.86Sc1.86Ti10.14P18O72 -> same reduced form
    
    This enables matching between per-formula-unit and per-unit-cell representations.
    Returns a string like "Al3Li13O120P30Ti17" (alphabetically sorted, reduced).
    """
    if not formula or formula.lower() == 'null':
        return formula
    
    from math import gcd
    from functools import reduce
    
    # Step 1: Expand parenthesized groups
    #   e.g., (PO4)3 -> P3O12
    def expand_parens(f: str) -> str:
        """Recursively expand (group)N patterns."""
        while '(' in f:
            # Match innermost parenthesized group with optional multiplier
            match = re.search(r'\(([^()]+)\)(\d*\.?\d*)', f)
            if not match:
                break
            inner = match.group(1)
            mult = float(match.group(2)) if match.group(2) else 1.0
            
            # Parse elements inside the group
            elem_pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
            expanded_parts = []
            for elem, coeff_str in re.findall(elem_pattern, inner):
                coeff = float(coeff_str) if coeff_str else 1.0
                new_coeff = coeff * mult
                # Format: avoid unnecessary decimals
                if new_coeff == int(new_coeff):
                    expanded_parts.append(f"{elem}{int(new_coeff)}" if int(new_coeff) != 1 else elem)
                else:
                    expanded_parts.append(f"{elem}{new_coeff}")
            
            f = f[:match.start()] + ''.join(expanded_parts) + f[match.end():]
        return f
    
    try:
        flat = expand_parens(formula)
        
        # Step 2: Parse all element-coefficient pairs from the flat formula
        elem_pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
        element_counts = {}
        for elem, coeff_str in re.findall(elem_pattern, flat):
            coeff = float(coeff_str) if coeff_str else 1.0
            element_counts[elem] = element_counts.get(elem, 0) + coeff
        
        if not element_counts:
            return formula
        
        # Step 3: Find multiplier to make all coefficients integers
        # Multiply all by 10^N where N clears the most decimal places
        coeffs = list(element_counts.values())
        
        # Find the precision needed (max decimal places)
        max_decimals = 0
        for c in coeffs:
            s = f"{c:.10f}".rstrip('0')
            if '.' in s:
                dec_part = s.split('.')[1]
                max_decimals = max(max_decimals, len(dec_part))
        
        multiplier = 10 ** max_decimals
        int_coeffs = {elem: round(coeff * multiplier) for elem, coeff in element_counts.items()}
        
        # Step 4: Reduce by GCD
        all_values = [v for v in int_coeffs.values() if v > 0]
        if all_values:
            common = reduce(gcd, all_values)
            int_coeffs = {elem: v // common for elem, v in int_coeffs.items()}
        
        # Step 5: Build sorted formula string
        parts = []
        for elem in sorted(int_coeffs.keys()):
            count = int_coeffs[elem]
            if count == 1:
                parts.append(elem)
            elif count > 0:
                parts.append(f"{elem}{count}")
        
        return ''.join(parts)
    
    except Exception:
        return formula


def deduplicate_text_measurements(measurements: List[MeasuredPoint]) -> List[MeasuredPoint]:
    """
    Remove duplicate text extractions based on conductivity + temperature (primary),
    then composition (secondary filter if needed).
    Keep the most specific/confident extraction.
    
    Groups by (conductivity, temperature) first since these are more reliable than formulas.
    Within each group, keeps the most specific/confident measurement.
    """
    from collections import defaultdict
    
    # Group by (conductivity, temperature) - rounded to avoid floating point issues
    groups = defaultdict(list)
    non_text = []
    
    for idx, m in enumerate(measurements):
        # Only deduplicate text extractions
        if m.source != 'text':
            non_text.append(m)
            continue
            
        cond = m.normalized_conductivity
        temp = m.normalized_temperature_c
        
        if cond is None or temp is None:
            non_text.append(m)
            continue
        
        # Round to 6 decimal places for conductivity, 1 for temperature
        key = (round(cond, 6), round(temp, 1))
        groups[key].append((idx, m))
    
    # Process each group
    deduplicated = []
    
    for key, group in groups.items():
        if len(group) == 1:
            deduplicated.append(group[0][1])
            continue
        
        # Sort by preference:
        # 1. Has canonical_formula (not null)
        # 2. Higher confidence
        # 3. More specific composition (longer canonical_formula)
        # 4. Earlier in document (lower index)
        def sort_key(item):
            idx, m = item
            has_canon = 1 if m.canonical_formula else 0
            conf_score = {'high': 3, 'medium': 2, 'low': 1}.get(m.confidence or 'low', 0)
            canon_len = len(m.canonical_formula) if m.canonical_formula else 0
            return (-has_canon, -conf_score, -canon_len, idx)
        
        sorted_group = sorted(group, key=sort_key)
        
        # Keep the first (best) one
        best = sorted_group[0][1]
        deduplicated.append(best)
        
        # Log what we removed
        for idx, m in sorted_group[1:]:
            print(f"   [Dedup] Removed duplicate: {m.raw_composition[:50]} (σ={m.normalized_conductivity:.2e}, T={m.normalized_temperature_c:.1f}°C)")
    
    # Combine with non-text measurements
    return non_text + deduplicated


def _resolve_nasicon_acronym(raw_comp: str) -> str:
    """
    Deterministically resolve NASICON-type acronyms to canonical formulas.
    LATP03 -> Li1.3Al0.3Ti1.7(PO4)3
    LCTP02 -> Li1.2Cr0.2Ti1.8(PO4)3
    LFTP01 -> Li1.1Fe0.1Ti1.9(PO4)3
    LTP    -> LiTi2(PO4)3
    Also strips (Bulk)/(GB) suffixes and returns them as measurement_type.
    """
    # Extract measurement type (bulk vs grain boundary)
    measurement_type = None
    clean = raw_comp.strip()
    gb_match = re.search(r'\s*\((?:GB|G\.?B\.?|grain\s*boundar(?:y|ies))\)', clean, re.I)
    bulk_match = re.search(r'\s*\(Bulk\)', clean, re.I)
    if gb_match:
        measurement_type = 'grain_boundary'
        clean = clean[:gb_match.start()].strip()
    elif bulk_match:
        measurement_type = 'bulk'
        clean = clean[:bulk_match.start()].strip()
    
    # Map element letter to element name
    element_map = {'A': 'Al', 'C': 'Cr', 'F': 'Fe', 'G': 'Ga', 'I': 'In', 'S': 'Sc'}
    
    # Match LATP03 pattern (L + element_letter + TP + digits)
    m = re.match(r'^L([A-Z])TP\s*0*(\d+)$', clean, re.I)
    if m:
        elem_letter = m.group(1).upper()
        x_digits = m.group(2)
        elem = element_map.get(elem_letter)
        if elem:
            x = int(x_digits) / 10.0  # "03" -> 0.3, "005" -> handled by 0*
            # Handle LATP005 -> x=0.05 edge case
            if len(x_digits) >= 2 and x_digits.startswith('0') and int(x_digits) < 10:
                x = int(x_digits) / 100.0
            elif len(x_digits) == 1:
                x = int(x_digits) / 10.0
            else:
                x = int(x_digits) / 10.0
            # More careful: "005" -> 0.5 or 0.05? In paper, LATP005 = x=0.05
            # Re-parse: remove leading zeros, then divide
            raw_num = x_digits.lstrip('0') or '0'
            if x_digits.startswith('0') and len(x_digits) > 1:
                # "005" → 0.05, "01" → 0.1, "03" → 0.3
                x = float(f"0.{x_digits}")
            else:
                x = float(raw_num) / 10.0
            
            li = round(1 + x, 4)
            ti = round(2 - x, 4)
            formula = f"Li{li}{elem}{x}Ti{ti}(PO4)3"
            return formula, measurement_type
    
    # Match LTP (base compound, x=0)
    if re.match(r'^LTP$', clean, re.I):
        return 'LiTi2(PO4)3', measurement_type
    
    return None, measurement_type


async def extract_abbreviation_map(client, doc_title: str, sections: list, model_name: str = "gemini-2.5-flash") -> dict:
    """
    Extract abbreviation → canonical formula mappings from paper text.
    Runs ONCE per paper before any extraction tasks.
    
    Uses few-shot prompts covering diverse material systems:
    NASICON, argyrodite, perovskite, LISICON, garnet.
    
    Returns:
        dict: {"LATP03": "Li1.3Al0.3Ti1.7(PO4)3", "x=0.25": "Li6.25P0.75Ge0.25S5I", ...}
    """
    # Gather text from relevant sections (abstract, introduction, experimental)
    relevant_text = ""
    for sec in sections:
        title_lower = sec.title.lower()
        if any(kw in title_lower for kw in ['abstract', 'introduction', 'experimental', 'method', 'synthesis', 'preparation', 'sample']):
            relevant_text += f"\n--- {sec.title} ---\n{sec.content[:2000]}\n"
    
    if not relevant_text.strip():
        # Fallback: use first 3000 chars of the document
        for sec in sections[:3]:
            relevant_text += f"\n{sec.content[:1000]}\n"
    
    prompt = f"""You are a scientific abbreviation extractor for solid-state ionic conductor papers.
This paper may studies polymer-ceramic composite electrolytes.

Given a paper's title and relevant sections, extract ALL abbreviation → canonical material mappings.
Also extract the general series formula if one exists.

**IMPORTANT for polymer-ceramic composites:**
- Many papers use abbreviations like "CPE", "SPE", "CSPE" for composite/solid polymer electrolytes
- Compositions are often described by weight percent (wt%) or volume percent (vol%) of ceramic filler
- The polymer matrix (e.g., PEO-LiTFSI) and ceramic filler (e.g., LLZO) should BOTH be captured
- For composite samples, use the format: "PolymerMatrix/CeramicFiller (loading)"
- Tag each entry with a material_class: "Polymer", "Ceramic", or "Composite"

=== FEW-SHOT EXAMPLES ===

EXAMPLE 1 — Polymer-ceramic composite electrolyte (CPE with wt% loading):
Paper title: "PVDF-LiTFSI Composite Electrolytes with Li1.5Al0.5Ge1.5(PO4)3 Nanoparticles"
Text excerpt: "...PVDF-HFP was dissolved with LiTFSI (PVDF-LiTFSI)...Li1.5Al0.5Ge1.5(PO4)3 (LAGP) nanoparticles were added at various loadings...samples labeled as 0, 5, 10, and 20 wt% LAGP..."

Expected output:
{{
  "series_formula": null,
  "abbreviations": {{
    "PVDF-LiTFSI": {{"formula": "PVDF-HFP/LiTFSI", "material_class": "Polymer"}},
    "0 wt% LAGP": {{"formula": "PVDF-HFP/LiTFSI", "material_class": "Polymer"}},
    "5 wt% LAGP": {{"formula": "PVDF-HFP/LiTFSI/Li1.5Al0.5Ge1.5(PO4)3 (5 wt%)", "material_class": "Composite"}},
    "10 wt% LAGP": {{"formula": "PVDF-HFP/LiTFSI/Li1.5Al0.5Ge1.5(PO4)3 (10 wt%)", "material_class": "Composite"}},
    "20 wt% LAGP": {{"formula": "PVDF-HFP/LiTFSI/Li1.5Al0.5Ge1.5(PO4)3 (20 wt%)", "material_class": "Composite"}},
    "LAGP": {{"formula": "Li1.5Al0.5Ge1.5(PO4)3", "material_class": "Ceramic"}},
    "CPE": {{"formula": "PVDF-HFP/LiTFSI/Li1.5Al0.5Ge1.5(PO4)3 composite", "material_class": "Composite"}}
  }}
}}

EXAMPLE 2 — PAN-based polymer-ceramic composite:
Paper title: "Enhanced Ionic Conductivity of PAN-Li0.33La0.56TiO3 Composite Electrolytes"
Text excerpt: "...polyacrylonitrile (PAN) dissolved with LiClO4...Li0.33La0.56TiO3 (LLTO) nanowires added at 0, 5, 10, 15 wt%...composites denoted PAN-xLLTO..."

Expected output:
{{
  "series_formula": null,
  "abbreviations": {{
    "PAN-LiClO4": {{"formula": "PAN-LiClO4", "material_class": "Polymer"}},
    "PAN-0LLTO": {{"formula": "PAN-LiClO4", "material_class": "Polymer"}},
    "PAN-5LLTO": {{"formula": "PAN-LiClO4/Li0.33La0.56TiO3 (5 wt%)", "material_class": "Composite"}},
    "PAN-10LLTO": {{"formula": "PAN-LiClO4/Li0.33La0.56TiO3 (10 wt%)", "material_class": "Composite"}},
    "PAN-15LLTO": {{"formula": "PAN-LiClO4/Li0.33La0.56TiO3 (15 wt%)", "material_class": "Composite"}},
    "LLTO NWs": {{"formula": "Li0.33La0.56TiO3 nanowires", "material_class": "Ceramic"}}
  }}
}}

EXAMPLE 3 — NASICON-type crystalline electrolyte:
Paper title: "A systematic study of NASICON-type Li1+xMxTi2−x(PO4)3 (M = Cr, Al, Fe)"
Text excerpt: "...samples termed LATP for Al, LCTP for Cr..."

Expected output:
{{
  "series_formula": "Li1+xMxTi2-x(PO4)3",
  "abbreviations": {{
    "LTP": {{"formula": "LiTi2(PO4)3", "material_class": "Ceramic"}},
    "LATP03": {{"formula": "Li1.3Al0.3Ti1.7(PO4)3", "material_class": "Ceramic"}},
    "LCTP02": {{"formula": "Li1.2Cr0.2Ti1.8(PO4)3", "material_class": "Ceramic"}}
  }}
}}

EXAMPLE 4 — Garnet-type:
Paper title: "Lattice Instabilities and Phase Diagram of Li7La3Zr2O12"
Text excerpt: "...LLZO denotes Li7La3Zr2O12...Ta-doped samples Li7-xLa3Zr2-xTaxO12..."

Expected output:
{{
  "series_formula": "Li7-xLa3Zr2-xTaxO12",
  "abbreviations": {{
    "LLZO": {{"formula": "Li7La3Zr2O12", "material_class": "Ceramic"}},
    "x=0.25": {{"formula": "Li6.75La3Zr1.75Ta0.25O12", "material_class": "Ceramic"}}
  }}
}}

EXAMPLE 5 — Argyrodite sulfides:
Paper title: "Inducing High Ionic Conductivity in Li6+xP1−xGexS5I"
Text excerpt: "...Li6+xP1-xGexS5I...x = 0, 0.25..."

Expected output:
{{
  "series_formula": "Li6+xP1-xGexS5I",
  "abbreviations": {{
    "x=0": {{"formula": "Li6PS5I", "material_class": "Ceramic"}},
    "x=0.25": {{"formula": "Li6.25P0.75Ge0.25S5I", "material_class": "Ceramic"}}
  }}
}}

=== END OF EXAMPLES ===

=== YOUR TASK ===

Paper title: "{doc_title}"

Relevant text from paper:
{relevant_text[:4000]}

**Instructions:**
1. Determine if this is a polymer-ceramic composite paper or a crystalline electrolyte paper
2. For COMPOSITE papers: identify the polymer matrix, salt, ceramic filler, and all compositions tested
3. For CRYSTALLINE papers: identify the series formula and abbreviations with x-values
4. For each entry, include "material_class": "Polymer" | "Ceramic" | "Composite"
5. Use wt% or vol% loading in the formula for composites (e.g., "PEO-LiTFSI/LLZO (50 wt%)")
6. Include baseline/control samples (e.g., "0 wt% LLZO" = neat polymer)
7. Return ONLY chemically plausible formulas

Return JSON with "series_formula" (null for composites) and "abbreviations" dict.
"""
    
    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )
        result = json.loads(response.text)
        raw_abbrevs = result.get("abbreviations", {})
        series_formula = result.get("series_formula", None)
        
        # V8: Normalize abbreviation map — entries may be dicts or strings
        abbrev_map = {}
        material_classes = {}  # Store material_class info separately
        for k, v in raw_abbrevs.items():
            if isinstance(v, dict):
                # New nested format: {"formula": "...", "material_class": "..."}
                abbrev_map[k] = v.get("formula", str(v))
                if v.get("material_class"):
                    material_classes[k] = v["material_class"]
            else:
                # Old string format (backward compatible)
                abbrev_map[k] = v
        
        if abbrev_map:
            print(f"   📖 Extracted {len(abbrev_map)} abbreviation mappings (series: {series_formula})")
            for k, v in list(abbrev_map.items())[:5]:
                mc = material_classes.get(k, "")
                mc_str = f" [{mc}]" if mc else ""
                print(f"      {k} → {v}{mc_str}")
            if len(abbrev_map) > 5:
                print(f"      ... and {len(abbrev_map) - 5} more")
        else:
            print(f"   📖 No abbreviation mappings found in paper text")
        
        # Store metadata for downstream use
        if series_formula:
            abbrev_map['__series_formula__'] = series_formula
        if material_classes:
            abbrev_map['__material_classes__'] = material_classes
        
        if response:
            tracker.track(response, model_name)
        
        return abbrev_map
        
    except Exception as e:
        print(f"   ⚠️ Abbreviation extraction failed: {e}")
        return {}


async def extract_processing_info(client, doc_title: str, sections: list, model_name: str = "gemini-2.5-flash") -> dict:
    """
    Extract sample label -> processing method mappings.
    Runs ONCE per paper using text sections.
    
    Returns:
        dict: {"LATP03": "Solid state reaction, sintered at 900C", ...}
    """
    # Gather text from experimental/methods sections
    relevant_text = ""
    for sec in sections:
        title_lower = sec.title.lower()
        if any(kw in title_lower for kw in ['experimental', 'method', 'synthesis', 'preparation', 'sample', 'procedure']):
            relevant_text += f"\n--- {sec.title} ---\n{sec.content[:3000]}\n"
    
    if len(relevant_text) < 50:
        # Fallback to intro + abstract if no method section found
        relevant_text = ""
        for sec in sections[:3]:
            relevant_text += f"\n{sec.content[:1500]}\n"

    prompt = f"""You are a materials science information extractor.
    
    Goal: Identify the PROCESSING METHOD used for each sample/material described in the text.
    
    Paper Title: "{doc_title}"
    
    Text Sections:
    {relevant_text[:8000]}
    
    Instructions:
    1. Identify all sample labels or material compositions mentioned (e.g., "LATP", "x=0.1", "Sample A").
    2. Extract the synthesis/processing method for each (e.g., "solid state reaction", "sol-gel", "hot pressing", "sintered at 1000°C").
    3. If a general method applies to ALL samples, assign it to "ALL_SAMPLES".
    4. Be specific about temperatures and atmosphere if mentioned (e.g. "sintered at 900C in air").
    
    Return JSON: {{ "processing_map": {{ "Sample Label": "Method Description" }} }}
    """
    
    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )
        result = json.loads(response.text)
        proc_map = result.get("processing_map", {})
        
        if proc_map:
            print(f"   ⚗️ Extracted {len(proc_map)} processing method descriptions")
        
        if response:
            tracker.track(response, model_name)
            
        return proc_map

    except Exception as e:
        print(f"   ⚠️ Processing info extraction failed: {e}")
        return {}


async def extract_paper_context(client, doc_title: str, sections: list, model_name: str = "gemini-2.5-flash") -> dict:
    """
    Extract a structured paper-level context summary for cross-section context injection.
    Runs ONCE per paper using all text sections.
    
    The returned context dict is injected into image and table extraction calls
    so those LLMs can fill in processing_method even when the info is in another section.
    
    Returns:
        dict with keys: experimental_procedure_summary, nomenclature_key,
                        material_systems_overview, measurement_and_testing_setup,
                        baseline_and_champion_samples
    """
    # Gather text from ALL relevant sections (not just methods)
    relevant_text = ""
    for sec in sections:
        title_lower = sec.title.lower()
        # Skip references and acknowledgements
        if any(skip in title_lower for skip in ['acknowledgements', 'references', 'supporting info']):
            continue
        # Include methods/experimental sections in full, others truncated
        if any(kw in title_lower for kw in ['experimental', 'method', 'synthesis', 'preparation', 'sample', 'procedure']):
            relevant_text += f"\n--- {sec.title} ---\n{sec.content[:4000]}\n"
        else:
            relevant_text += f"\n--- {sec.title} ---\n{sec.content[:1500]}\n"
    
    if len(relevant_text) < 100:
        print(f"   ⚠️  Not enough text for paper context extraction")
        return {}
    
    prompt = f"""You are a materials science information extractor.

    Paper Title: "{doc_title}"

    Text Sections:
    {relevant_text[:12000]}

    Extract the following structured context from this paper. Return JSON with these keys:

    1. "experimental_procedure_summary": A chronological narrative of how each specific material system 
       was synthesized, cast, dried, and annealed. Explicitly capture the relationships between 
       processing methods and specific formulations. Include any mentions of component storage or 
       pre-treatment conditions (e.g., 'LiTFSI was vacuum-dried at 80°C overnight', 'LLZO calcined 
       at 900°C', 'all components handled in Ar glovebox').

    2. "nomenclature_key": Extract every custom abbreviation the authors use for their specific 
       material samples and map it to its full formulation in a clear sentence (e.g., 'CPE-10 refers 
       to Composite Polymer Electrolyte with 10% TiO2; PLL represents PEO+LLZO').

    3. "material_systems_overview": A narrative summary of the distinct material combinations tested 
       in this paper. Describe how the different components relate to each other.

    4. "measurement_and_testing_setup": Summarize the equipment, cell configurations, and environmental 
       conditions used to measure ionic conductivity. Include cell assembly atmosphere, applied pressure, 
       and electrode materials if mentioned.

    5. "baseline_and_champion_samples": Identify the baseline/control sample and the 'champion' sample 
       that achieved the highest performance. Note the peak conductivity value and temperature.

    Return JSON: {{ "experimental_procedure_summary": "...", "nomenclature_key": "...", 
                    "material_systems_overview": "...", "measurement_and_testing_setup": "...",
                    "baseline_and_champion_samples": "..." }}
    """
    
    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )
        result = json.loads(response.text)
        
        if result:
            ctx_keys = [k for k in result if result[k] and len(str(result[k])) > 10]
            print(f"   📋 Extracted paper context: {len(ctx_keys)} fields populated")
        
        if response:
            tracker.track(response, model_name)
            
        return result

    except Exception as e:
        print(f"   ⚠️ Paper context extraction failed: {e}")
        return {}


def _lookup_abbreviation_map(raw_comp: str, abbreviation_map: dict) -> tuple:
    """
    Look up a raw composition in the abbreviation map.
    Handles exact matches, suffix stripping (Bulk/GB), and fuzzy matching.
    
    Returns:
        (canonical_formula, measurement_type) or (None, measurement_type)
    """
    if not abbreviation_map:
        return None, None
    
    # Get material_classes metadata if available
    material_classes_meta = abbreviation_map.get('__material_classes__', {})
    
    # Extract measurement type suffix
    measurement_type = None
    clean = raw_comp.strip()
    gb_match = re.search(r'\s*\((?:GB|G\.?B\.?|grain\s*boundar(?:y|ies))\)', clean, re.I)
    bulk_match = re.search(r'\s*\(Bulk\)', clean, re.I)
    if gb_match:
        measurement_type = 'grain_boundary'
        clean = clean[:gb_match.start()].strip()
    elif bulk_match:
        measurement_type = 'bulk'
        clean = clean[:bulk_match.start()].strip()
    
    # 1. Exact match
    if clean in abbreviation_map:
        return abbreviation_map[clean], measurement_type
    
    # 2. Case-insensitive match
    clean_lower = clean.lower()
    for key, val in abbreviation_map.items():
        if key.startswith('__'):  # Skip metadata keys (including __material_classes__)
            continue
        if isinstance(val, dict):  # Skip non-string values
            continue
        if key.lower() == clean_lower:
            return val, measurement_type
    
    # 3. Match with whitespace normalization (e.g., "LATP 03" vs "LATP03")
    clean_nospace = re.sub(r'\s+', '', clean)
    for key, val in abbreviation_map.items():
        if key.startswith('__'):
            continue
        if isinstance(val, dict):
            continue
        if re.sub(r'\s+', '', key) == clean_nospace:
            return val, measurement_type
    
    # 4. Check if raw_comp is an x=value pattern (common for figure series labels)
    x_match = re.match(r'^x\s*=\s*([\d.]+)$', clean, re.I)
    if x_match:
        x_val = x_match.group(1)
        # Try matching "x=0.25" style keys
        for key, val in abbreviation_map.items():
            key_x_match = re.match(r'^x\s*=\s*([\d.]+)$', key, re.I)
            if key_x_match and key_x_match.group(1) == x_val:
                return val, measurement_type
    
    return None, measurement_type


async def canonicalize_materials(client, measurements: List[MeasuredPoint], definitions: List[str], model_name: str = None, doc_title: str = None, doc_abstract: str = None, abbreviation_map: dict = None, processing_map: dict = None):
    """
    Uses Gemini to resolve "x=0.1" -> "Li3.8Mg0.1..." using the extracted text definitions.
    Enhanced to handle series formulas like "Li6+xP1-xGexS5I" and validate physical plausibility.
    Now also uses document title and abstract for better context on base formulas.
    
    Also applies PROCESSING METHOD from the extracted processing_map.
    
    Resolution order:
    1. Abbreviation map lookup (from extract_abbreviation_map, most reliable)
    2. Deterministic NASICON acronym resolver (fast fallback)
    3. LLM-based canonicalization (for remaining cases)
    """
    if not measurements: return measurements
    
    # --- PROCESSING METHOD APPLICATION ---
    if processing_map:
        general_method = processing_map.get("ALL_SAMPLES")
        
        for m in measurements:
            # 1. Try exact match of raw_composition
            if m.raw_composition in processing_map:
                m.processing_method = processing_map[m.raw_composition]
            # 2. Try match in processing map keys — V6: tighter word-boundary matching
            else:
                found = False
                for k, v in processing_map.items():
                    k_lower = k.lower().strip()
                    comp_lower = m.raw_composition.lower().strip()
                    # Exact match (case-insensitive)
                    if k_lower == comp_lower:
                        m.processing_method = v
                        found = True
                        break
                    # Word-boundary match: key must be >= 3 chars and appear as a distinct word
                    # e.g., "LATP" matches composition "LATP series" but not unrelated materials
                    if len(k_lower) >= 3 and re.search(r'\b' + re.escape(k_lower) + r'\b', comp_lower):
                        m.processing_method = v
                        found = True
                        break
                
                # 3. Fallback to general method
                if not found and general_method:
                    m.processing_method = general_method

    # --- PRE-PASS 1: Abbreviation map lookup (LLM-extracted, paper-specific) ---
    map_resolved = 0
    material_classes_meta = abbreviation_map.get('__material_classes__', {}) if abbreviation_map else {}
    if abbreviation_map:
        for m in measurements:
            if m.canonical_formula:  # Already resolved
                continue
            resolved_formula, meas_type = _lookup_abbreviation_map(m.raw_composition, abbreviation_map)
            if resolved_formula and resolved_formula != 'null':
                # V8: For polymer-ceramic composites, skip stoichiometry validation
                # ("PEO-LiTFSI/LLZO (50 wt%)" is not a chemical formula)
                is_composite_name = any(kw in resolved_formula.lower() for kw in ['peo', 'pan', 'pvdf', 'wt%', 'vol%', 'composite', 'polymer'])
                if is_composite_name or validate_formula_stoichiometry(resolved_formula):
                    m.canonical_formula = resolved_formula
                    if not is_composite_name:
                        m.reduced_formula = normalize_formula_to_reduced(resolved_formula)
                    if meas_type:
                        if not m.warnings:
                            m.warnings = []
                        m.warnings.append(f"measurement_type:{meas_type}")
                    # V8: Apply material_class from abbreviation map
                    # Use the abbreviation map key that matched the formula, not raw_composition
                    # This avoids "LLZO" substring matching "0 wt% LLZO" incorrectly
                    matched_key = m.raw_composition.strip()
                    # Find which abbreviation map key resolved this formula
                    resolved_map_key = None
                    for k, v in abbreviation_map.items():
                        if k.startswith('__'):
                            continue
                        if isinstance(v, dict):
                            continue
                        if v == resolved_formula:
                            # Check if this key matches the raw_composition
                            if k.lower() == matched_key.lower() or re.sub(r'\s+', '', k) == re.sub(r'\s+', '', matched_key):
                                resolved_map_key = k
                                break
                    # If no exact key match, try the raw_composition against material_classes
                    # Prioritize exact match over substring
                    best_mc_key = resolved_map_key
                    if not best_mc_key:
                        # Exact match first
                        for k in material_classes_meta:
                            if k.lower() == matched_key.lower():
                                best_mc_key = k
                                break
                    if not best_mc_key:
                        # Longest substring match as last resort
                        best_mc_len = 0
                        for k in material_classes_meta:
                            if k.lower() in matched_key.lower() and len(k) > best_mc_len:
                                best_mc_key = k
                                best_mc_len = len(k)
                    if best_mc_key and best_mc_key in material_classes_meta:
                        m.material_class = material_classes_meta[best_mc_key]
                    map_resolved += 1
                    mc_str = f" [{m.material_class}]" if m.material_class else ""
                    print(f"   [AbbrevMap] {m.raw_composition} → {resolved_formula}{mc_str}")
                else:
                    print(f"   [AbbrevMap] Rejected (invalid stoichiometry): {resolved_formula} for {m.raw_composition}")
    
    if map_resolved > 0:
        print(f"   ... Resolved {map_resolved} names from abbreviation map")
    
    # --- PRE-PASS 2: Deterministic NASICON acronym resolution (fallback) ---
    nasicon_resolved = 0
    for m in measurements:
        if m.canonical_formula:  # Already resolved
            continue
        resolved_formula, meas_type = _resolve_nasicon_acronym(m.raw_composition)
        if resolved_formula:
            m.canonical_formula = resolved_formula
            m.reduced_formula = normalize_formula_to_reduced(resolved_formula)
            if meas_type:
                if not m.warnings:
                    m.warnings = []
                m.warnings.append(f"measurement_type:{meas_type}")
            nasicon_resolved += 1
            print(f"   [NASICON] {m.raw_composition} → {resolved_formula} ({meas_type or 'unknown'})")
    
    if nasicon_resolved > 0:
        print(f"   ... Resolved {nasicon_resolved} NASICON acronyms deterministically")
    
    # Filter points that still need resolution
    to_resolve = []
    for i, m in enumerate(measurements):
        if m.canonical_formula:  # Already resolved
            continue
        is_variable_axis = "x" in m.raw_temperature_unit.lower() or "=" in m.raw_temperature_unit
        is_series_formula = bool(re.search(r'[a-z]\s*[+\-]\s*x|Li\d+\+x|state|pressed|sintered', m.raw_composition, re.I))
        # V8: Also flag polymer-composite names (e.g., "PEO-LiTFSI", "50 wt% LLZO", "CPE")
        is_composite_name = bool(re.search(r'wt\s*%|vol\s*%|PEO|PAN|PVDF|CPE|SPE|composite|polymer', m.raw_composition, re.I))
        # Also resolve if it looks like an acronym with (Bulk)/(GB) suffix
        has_meas_type = bool(re.search(r'\(Bulk\)|\(GB\)|\(G\.?B\.?\)', m.raw_composition, re.I))
        
        # Resolve if: short name, has "=", variable axis, series formula, meas type suffix, or composite name
        if len(m.raw_composition) < 15 or "=" in m.raw_composition or is_variable_axis or is_series_formula or has_meas_type or is_composite_name:
            to_resolve.append(i)
    
    if not to_resolve: return measurements

    print(f"   ... Resolving {len(to_resolve)} remaining ambiguous material names via LLM...")

    # Build Context — document-level + text definitions + abbreviation map
    doc_context = ""
    if doc_title:
        doc_context += f"PAPER TITLE: {doc_title}\n"
    if doc_abstract:
        doc_context += f"ABSTRACT (excerpt): {doc_abstract[:800]}\n"
    
    # Include abbreviation map as context for the LLM canonicalizer
    abbrev_context = ""
    if abbreviation_map:
        series = abbreviation_map.get('__series_formula__', 'unknown')
        known_mappings = {k: v for k, v in abbreviation_map.items() if not k.startswith('__')}
        if known_mappings:
            abbrev_context = f"\nKNOWN ABBREVIATION MAPPINGS (from paper text):\n"
            abbrev_context += f"Series formula: {series}\n"
            for k, v in list(known_mappings.items())[:10]:
                abbrev_context += f"  {k} → {v}\n"
    
    # Deduplicate definitions
    unique_defs = list(dict.fromkeys(definitions))  # preserves order
    context_str = "\n".join([f"- {d}" for d in unique_defs])
    items_str = "\n".join([
        f"ID {i}: Label='{measurements[i].raw_composition}', "
        f"Conductivity='{measurements[i].raw_conductivity} {measurements[i].raw_conductivity_unit}', "
        f"Temperature='{measurements[i].raw_temperature} {measurements[i].raw_temperature_unit}', "
        f"Context='{(measurements[i].source_caption or '')[:200]}'" 
        for i in to_resolve
    ])
    
    prompt = f"""
    You are a Chemical Context Resolver for solid-state ionic conductors.
    I have a list of abbreviated/generic material names extracted from scientific papers.
    I have the paper title, abstract, and Material Definitions found in the text.

    Your Task: Map the abbreviated names to their Full Canonical Chemical Formulas.

    {doc_context}
    {abbrev_context}
    DEFINITIONS FOUND IN TEXT:
    {context_str}

    ITEMS TO RESOLVE:
    {items_str}

    === FEW-SHOT EXAMPLES ===
    
    EXAMPLE 1 — Polymer-ceramic composite by wt% loading:
    Input: Label="50 wt% LLZO", Context="CPE membranes of PEO-LiTFSI with LLZO nanofibers, Li6.4Al0.2La3Zr2O12"
    Output: "PEO-LiTFSI/Li6.4Al0.2La3Zr2O12 (50 wt%)"
    Reasoning: Composite material — capture polymer matrix AND ceramic filler with loading

    EXAMPLE 2 — Neat polymer electrolyte (baseline):
    Input: Label="0 wt% LLZO", Context="PEO-LiTFSI without LLZO, EO:Li 15:1"
    Output: "PEO-LiTFSI (EO:Li 15:1)"
    Reasoning: No ceramic filler — this is the neat polymer baseline
    
    EXAMPLE 3 — Series formula with x value:
    Input: Label="x=0.25", Series="Li6+xP1-xGexS5I"
    Output: "Li6.25P0.75Ge0.25S5I"
    Reasoning: x=0.25 → Li(6+0.25)=Li6.25, P(1-0.25)=P0.75, Ge(0.25)
    
    EXAMPLE 4 — Acronym with numeric suffix:
    Input: Label="LATP03", Context="LATP for Al in Li1+xAlxTi2-x(PO4)3"
    Output: "Li1.3Al0.3Ti1.7(PO4)3"
    Reasoning: "03" means x=0.3 → Li(1+0.3), Al(0.3), Ti(2-0.3)
    
    EXAMPLE 5 — Processing condition label (NO specific x):
    Input: Label="Li6+xP1-xGexS5I (cold-pressed)", no x value in context
    Output: null
    Reasoning: No specific x value → cannot resolve to a specific formula
    
    EXAMPLE 6 — Legend symbol mapping:
    Input: Label="Filled circles", Context="Filled circles represent 50 wt% LLZO in PEO-LiTFSI"
    Output: "PEO-LiTFSI/LLZO (50 wt%)"

    EXAMPLE 7 — Garnet abbreviation:
    Input: Label="LLZO", Context="LLZO denotes Li7La3Zr2O12"
    Output: "Li7La3Zr2O12"
    
    EXAMPLE 8 — Composite electrolyte abbreviation:
    Input: Label="CPE", Context="CPE = composite polymer electrolyte of PEO-LiTFSI with LLZO nanofibers"
    Output: null
    Reasoning: CPE is a general category, not a specific composition. Need wt% loading to resolve.
    
    EXAMPLE 9 — Ambiguous without context:
    Input: Label="Sample A", no context available
    Output: null
    Reasoning: Cannot determine formula without a mapping definition
    
    === END EXAMPLES ===

    Logic:
    1. **Polymer-Ceramic Composite Resolution**: 
       - If you see wt% or vol% loadings, construct: "PolymerMatrix/CeramicFiller (loading)"
       - Include the specific ceramic formula if known from context (e.g., Li6.4Al0.2La3Zr2O12)
       - For "0 wt%" or "neat polymer", return just the polymer formula
    
    2. **Series Formula Resolution**: 
       - If you see a series formula like "Li6+xP1-xGexS5I" with "x=0.25", substitute and calculate
       - Compute each stoichiometric coefficient by substituting the x value
       - Round coefficients to reasonable precision (2-3 decimal places)
    
    3. **Measurement Type Disambiguation**:
       - Distinguish between "bulk" vs "grain boundary" conductivity
       - If context mentions "NMR", mark it clearly or return null
    
    4. **Legend/Symbol Mapping**: 
       - If item is "Square", "Triangle", etc., look in Context for mappings
    
    5. **Validation**:
       - Only return formulas/names that are chemically plausible or clearly defined composites
       - If you cannot determine a specific formula with confidence, return null
       - Do NOT guess - better to return null than an incorrect formula

    Return JSON: {{ "mappings": {{ "ID": "Canonical Formula or null" }} }}
    """
    
    try:
        response = await client.aio.models.generate_content(
            model=model_name or "gemini-3-flash-preview", 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        mapping = json.loads(response.text).get("mappings", {})
        
        # Apply updates with physical validation
        for i_str, formula in mapping.items():
            # Extract just the digits from keys like "ID 2" or "2"
            digits = re.search(r'\d+', str(i_str))
            if digits:
                idx = int(digits.group())
                # Ensure index is safe
                if 0 <= idx < len(measurements):
                    # Validate physical plausibility
                    if formula and formula.lower() != "null":
                        # V8 fix: Skip stoichiometry check for composite names (same logic as abbrev map path)
                        is_composite_name = any(kw in formula.lower() for kw in ['peo', 'pan', 'pvdf', 'wt%', 'vol%', 'composite', 'polymer', '/'])
                        if is_composite_name or validate_formula_stoichiometry(formula):
                            measurements[idx].canonical_formula = formula
                            if not is_composite_name:
                                measurements[idx].reduced_formula = normalize_formula_to_reduced(formula)
                            print(f"   [Canon] Resolved: {measurements[idx].raw_composition[:40]} → {formula}")
                        else:
                            print(f"   [Canon] Rejected (invalid stoichiometry): {formula} for {measurements[idx].raw_composition[:40]}")
                            measurements[idx].confidence = "low"
                    else:
                        # LLM returned null - keep original but mark as low confidence
                        measurements[idx].confidence = "low"
            
    except Exception as e:
        print(f"   [Canonicalizer Error]: {e}")
    
    # Post-pass: compute reduced_formula for any measurements that already had canonical_formula
    for m in measurements:
        if m.canonical_formula and not m.reduced_formula:
            m.reduced_formula = normalize_formula_to_reduced(m.canonical_formula)
    
    return measurements

# ==============================================================================
# 5. Gemini Pipelines (Text & Vision)
# ==============================================================================
async def process_text(client, model, text_content, text_title, max_retries: int = 3, paper_context: dict = None):
    # V6: Number paragraphs for provenance tracking
    numbered_content, paragraph_list = number_paragraphs(text_content)

    # V8.1: Build methods context preamble if available (mirrors process_table_node)
    methods_preamble = ""
    if paper_context:
        proc_summary = paper_context.get('experimental_procedure_summary', '')
        nomenclature = paper_context.get('nomenclature_key', '')
        if proc_summary or nomenclature:
            methods_preamble = f"""
        **PAPER CONTEXT (for processing method and sample identification):**
        Experimental Procedures: {proc_summary[:1500]}
        Sample Nomenclature: {nomenclature[:500]}
"""

    prompt = f"""{methods_preamble}Extract ionic conductivity data points from this text. Return Format: JSON only.
        
        **CRITICAL: Distinguish MEASURED vs CITED data.**
        - Use source="text" ONLY for conductivity values measured/reported by THIS paper's authors.
        - Use source="cited_text" for values attributed to other works (e.g., "Ref [12] reported...", 
          "LLZO has conductivity ~1 mS/cm [4,5]", "compared to [Y]...", or any value with reference numbers).
        - Conductivity values mentioned alongside reference numbers [X] are almost always cited data.
        
        For each measurement:
        - raw_composition: Full material name as described (e.g., "PEO-LiTFSI/LLZO (50 wt%)", "Li7La3Zr2O12")
          * For polymer composites, include the polymer matrix, salt, ceramic filler, and loading
          * Do NOT abbreviate beyond what the paper uses (if they say "50 wt% LLZO", use that)
        - raw_conductivity: Numeric value (e.g. "1.2e-4") without any ~ or < or > symbols
        - raw_conductivity_unit: Unit (e.g. "S/cm", "mS/cm")
        - raw_temperature: Only the temperature value (e.g. "25", "298", "2.0", "2.4")
        - raw_temperature_unit: Unit (e.g. "Celsius", "Kelvin", "1000/T (K-1)", "10^3/T / K-1")
        - material_class: "Polymer" (neat polymer, e.g. PEO-LiTFSI), "Ceramic" (neat ceramic, e.g. LLZO),
          or "Composite" (polymer+ceramic, e.g. PEO-LiTFSI/LLZO). Classify by what the material IS,
          not where the data came from — cited data is tracked via source="cited_text"
        - source: "text" OR "cited_text" (see rules above)
        - source_paragraph_indices: List of paragraph numbers [1, 2, ...] that contain this measurement data. Each paragraph is labeled [P1], [P2], etc. in the text below. THIS IS CRITICAL for reviewer verification — always include the specific paragraph number(s).
        - confidence: "high" if it is explicitly stated / "low" if it is inferred or calculated or was cited from another source
        - aging_time: If the measurement is after storage/aging, record the duration (e.g., "5 days", "2 weeks", "freshly made"). Leave null if not applicable.
        - measurement_condition: Special condition under which this measurement was taken (e.g., "freshly made", "after 100 cycles", "in air", "after aging 10 days"). Leave null if not applicable.

        **TEMPORAL / AGING DATA:**
        - If the same sample is measured at DIFFERENT TIME POINTS (e.g., freshly prepared, after 2 days, after 5 days, after 10 days), extract EACH time point as a SEPARATE measurement.
        - Include the aging condition in both `aging_time` (e.g., "5 days") and `measurement_condition` (e.g., "after aging 5 days").
        - Also reflect the time point in `raw_composition` by appending a suffix (e.g., "PEO-LLZO (50 wt%) [day 5]").

        **RANGES:**
        - If a conductivity RANGE is reported (e.g., "10⁻² to 10⁻⁴ S cm⁻¹"), extract BOTH the upper and lower bounds as separate measurements.
        - Add a warning "range_upper_bound" or "range_lower_bound" to each respectively.

        **MULTI-CONDITION SAMPLES:**
        - If the same sample is measured under different conditions (humidity levels, atmospheres, pressures, cycling states), extract EACH condition as a separate measurement.
        - Record the condition in `measurement_condition`.

        Optional: Extract up to 1-2 material definition sentences that summarizes chemical formula, processing method, and any other information about the samples mentioned in the text. Keep this EXTREMELY BRIEF to save tokens!"""

    last_exception = None

    # Store the results to a file
    with open(f"{FILE_DIR}/results_log_v6.json", "a") as f:
        

        for attempt in range(1, max_retries+1):
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[prompt, text_title + "\n\n" + numbered_content],
                    config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=ExtractionResult.model_json_schema(),
                    temperature=0.7 if '2.5' in model else 1.0, 
                    max_output_tokens=8192,
                    # thinking_config=types.ThinkingConfig(thinking_level="low") if '2.5' in model else None
                    )
                )


                if not response.candidates or not response.candidates[0].content.parts:
                    finish_reason = response.candidates[0].finish_reason if response.candidates else "No candidates"
                    print(f"   [Warning] No content generated. Reason: {finish_reason}")

                if not response.text:
                    print('>> DEBUG TEXT', response)
                    print(f"   [Text Warning] Empty response for {text_title}")
                    f.write(f"\n\n--- DEBUG INFO FOR {text_title} ---\n\n")
                    f.write(f"\n\n--- [Text Warning] Empty response for {text_title}")
                    f.write(f"\n\n--- DEBUG INFO FOR {response} ---\n\n")

                    return ExtractionResult(measurements=[]), response, False

                # if response:
                #     print(f"\n   [DEBUG] {text_title}:")
                #     print(f"   - Response length: {len(response.text) if response.text else 0} chars")
                #     print(f"   - Usage metadata: {response.usage_metadata}")
                #     print(f"   - First 500 chars: {response.text}")
            
                # Write detailed debug info to file
                f.write(f"\n\n[DEBUG] {text_title}:\n")
                f.write(f"- Response length: {len(response.text) if response.text else 0} chars\n")
                f.write(f"- Usage metadata: {response.usage_metadata}\n")
                f.write(f"- chars: {response.text}\n")

                result = ExtractionResult.model_validate_json(response.text)
                
                # Fix #5: Populate source_paragraph_text from paragraph_list
                for m in result.measurements:
                    if m.source_paragraph_indices:
                        para_texts = []
                        for idx in m.source_paragraph_indices:
                            # [P1] is paragraph_list[0]
                            if 1 <= idx <= len(paragraph_list):
                                para_texts.append(paragraph_list[idx-1])
                        m.source_paragraph_text = para_texts

                return result, response, True
            except (ValidationError, json.JSONDecodeError) as e:
                # Capture specific JSON errors (Truncated JSON, Invalid JSON)
                last_exception = e
                print(f"   [Retry {attempt}/{max_retries}] JSON Error for {text_title}: {str(e)[:100]}...")
                
                # Optional: Exponential backoff
                await asyncio.sleep(1 * attempt)
            except Exception as e:
                # Capture other API errors (503, etc)
                last_exception = e
                print(f"   [Retry {attempt}/{max_retries}] API Error for {text_title}: {e}")
                await asyncio.sleep(1 * attempt)
        
        # FINAL FAILURE HANDLER
        print(f"   \033[91m[Text Error]\033[0m {text_title}: Failed after {max_retries} attempts. Last error: {last_exception}")
        return ExtractionResult(measurements=[]), None, False


from dataclasses import dataclass, asdict
from typing import List, Optional
import numpy as np

@dataclass
class MeasurementSeries:
    """
    Represents a full curve (e.g., 'x=0.1') extracted from a plot.
    Keeps data grouped for cleaner logging and analysis.
    """
    series_label: str
    
    # Vectorized Data (Standard Python floats for JSON compatibility)
    temperature_c: List[float]
    conductivity_s_cm: List[float]
    
    # Metadata
    confidence: str
    warnings: List[str]
    source_figure: str
    
    def to_dict(self):
        """Helper to ensure safe serialization (handles numpy types)"""
        return asdict(self)

class MeasurementProcessor:
    def __init__(self):
        # Physical Bounds
        self.MAX_REALISTIC_COND_RT = 0.5  # S/cm (Liquid electrolytes ~0.01-0.1, Solids rarely >0.05)
        self.MIN_REALISTIC_COND = 1e-12   # S/cm
    
    def process_extraction(self, sf_result: Dict[str, Any], fig_id: str, context: str, axis_metadata: Optional[Dict[str, Any]] = None) -> List[MeasuredPoint]:
        """
        Main entry point: Flattens LLM output, normalizes units, and applies guardrails.

        Args:
            sf_result: Extraction result from SciFigureParser
            fig_id: Figure identifier
            context: Caption or context string
            axis_metadata: Dict containing x_axis, left_y_axis, right_y_axis metadata
        """
        # Extract annotated_temperature from SciFigureParser result (Fix #2)
        self._annotated_temperature = sf_result.get('annotated_temperature', None)
        measurements = []
        
        # 1. Unpack axis metadata
        axis_meta = axis_metadata or {}
        x_axis_def = axis_meta.get('x_axis', {})
        left_y_axis_def = axis_meta.get('left_y_axis', {})
        right_y_axis_def = axis_meta.get('right_y_axis', {})
        
        x_quantity = x_axis_def.get('quantity_type', 'other')
        
        # 2. Get data series from extraction result
        data_series = sf_result.get('data_series', [])
        
        if not data_series:
            print(f"   ℹ️ No data series found in extraction result for {fig_id}")
            return measurements
        
        # 3. Process each series
        for raw_series in data_series:
            series_label = raw_series.get('series_label', 'Unknown')
            x_vals = raw_series.get('x_values', [])
            y_vals = raw_series.get('y_values', [])
            
            # Validate data
            if not x_vals or not y_vals:
                print(f"   ⚠️ Empty data for series '{series_label}', skipping")
                continue
            
            if len(x_vals) != len(y_vals):
                print(f"   ⚠️ Mismatched x/y lengths for '{series_label}': {len(x_vals)} vs {len(y_vals)}")
                continue
            
            # 4. Determine which Y-axis this series uses
            axis_key = raw_series.get('mapped_y_axis', 'left')
            if axis_key == 'right' and right_y_axis_def:
                y_axis_def = right_y_axis_def
            else:
                y_axis_def = left_y_axis_def
            
            # 5. CRITICAL FILTER: Skip frequency plots
            if x_quantity == 'frequency':
                print(f"   \033[93m[Skip]\033[0m Series '{series_label}' has frequency X-axis (should have been filtered earlier)")
                continue
            
            # 6. Extract axis labels for context
            raw_x_label = x_axis_def.get('title_text') or x_axis_def.get('label', 'X-axis')
            raw_y_label = y_axis_def.get('title_text') or y_axis_def.get('label', 'Y-axis')
            
            # Build enhanced caption with axis info
            enhanced_caption = f"{context} [Series: {series_label}] [X: {raw_x_label}] [Y: {raw_y_label}]"
            
            # 7. Physics check: Arrhenius slope validation
            slope_warning = self._check_arrhenius_slope(x_vals, y_vals, x_axis_def)
            
            # 8. Process each data point
            for x_val, y_val in zip(x_vals, y_vals):
                warnings = []
                if slope_warning:
                    warnings.append(slope_warning)
                
                # --- TEMPERATURE EXTRACTION ---
                temp_c = None
                raw_temp_value = None
                raw_temp_unit = None
                
                # Priority 1: Check if series label contains temperature (e.g., "223K", "25°C")
                temp_from_label = self._extract_temp_from_label(series_label)
                if temp_from_label:
                    temp_c = temp_from_label['celsius']
                    raw_temp_value = temp_from_label['raw_value']
                    raw_temp_unit = temp_from_label['raw_unit']
                    # print(f"   [Temp from Label] '{series_label}' -> {temp_c}°C")
                
                # Priority 2: Use X-axis if it's temperature-related
                elif x_quantity in ['temperature_inverse', 'temperature_absolute']:
                    temp_c = self._normalize_temperature(x_val, x_axis_def)
                    raw_temp_value = str(x_val)
                    raw_temp_unit = x_axis_def.get('unit') or raw_x_label
                
                # Priority 3: Extract from caption/context or figure annotation
                elif x_quantity == 'stoichiometry' or x_quantity not in ['temperature_inverse', 'temperature_absolute']:
                    # Priority 3a: Check annotated_temperature from SciFigureParser (Fix #2)
                    temp_resolved = False
                    if hasattr(self, '_annotated_temperature') and self._annotated_temperature:
                        temp_from_annotation = self._extract_temp_from_label(self._annotated_temperature)
                        if temp_from_annotation:
                            temp_c = temp_from_annotation['celsius']
                            raw_temp_value = temp_from_annotation['raw_value']
                            raw_temp_unit = temp_from_annotation['raw_unit']
                            temp_resolved = True

                    # Priority 3b: Extract from caption text
                    if not temp_resolved:
                        temp_from_caption = self._extract_temp_from_caption(context)
                        if temp_from_caption:
                            temp_c = temp_from_caption['celsius']
                            raw_temp_value = temp_from_caption['raw_value']
                            raw_temp_unit = temp_from_caption['raw_unit']
                            temp_resolved = True

                    # Priority 3c: No temperature found — report null (let post-processing assume 25°C)
                    if not temp_resolved:
                        temp_c = None
                        raw_temp_value = None
                        raw_temp_unit = None
                        warnings.append("Temperature not found in caption or figure annotation")
                
                # Validate temperature
                if temp_c is None:
                    warnings.append("Temperature normalization failed")
                    temp_c = 25.0
                
                # --- CONDUCTIVITY EXTRACTION ---
                try:
                    # Ensure y_axis_def has unit populated (fix for NoneType crash)
                    if not y_axis_def.get('unit') and y_axis_def.get('title_text'):
                        # Infer unit from title text (e.g., "log(σ / S cm⁻¹)" → "log(S/cm)")
                        title = y_axis_def['title_text']
                        if 'log' in title.lower() and ('s' in title.lower() and 'cm' in title.lower()):
                            y_axis_def['unit'] = 'log(S/cm)'
                        elif 's/cm' in title.lower() or 's cm' in title.lower():
                            y_axis_def['unit'] = 'S/cm'

                    # Convert temp_c to Kelvin for σT normalization
                    temp_k = (temp_c + 273.15) if temp_c is not None else None
                    cond_s_cm = self._normalize_conductivity(y_val, y_axis_def, temperature_k=temp_k)
                except Exception as e:
                    # Fallback: if value is negative, likely log-scale — attempt 10^value
                    if y_val < 0:
                        cond_s_cm = 10 ** y_val
                        warnings.append(f"Conductivity normalization failed ({str(e)}); applied log-scale fallback: 10^{y_val} = {cond_s_cm:.2e}")
                        print(f"   ⚠️ Conductivity normalization failed for {y_val}, applied log fallback: {cond_s_cm:.2e}")
                    else:
                        cond_s_cm = y_val
                        warnings.append(f"Conductivity normalization failed: {str(e)}")
                        print(f"   ⚠️ Conductivity normalization failed for {y_val}: {e}")
                
                # --- FILTER: Skip if σ₀ detected (returns None) ---
                if cond_s_cm is None:
                    continue
                
                # --- COMPOSITION HANDLING ---
                # Fix #3: Blocklist generic/axis-derived labels
                GENERIC_LABELS = {
                    "σ", "logσ", "log(σ)", "log σ", "log(s)", "conductivity",
                    "σ (s/cm)", "log(σ/s cm⁻¹)", "σ (s cm⁻¹)", "s/cm",
                    "none", "data", "unknown", "series 1", "series 2",
                    "data 1", "data 2", "y-axis", "y axis",
                }
                label_lower = series_label.strip().lower()
                is_generic = label_lower in GENERIC_LABELS or label_lower.startswith("log(σ")

                if is_generic:
                    # Try to resolve from caption context
                    resolved_label = f"Series from {fig_id}"
                    if context and context.strip():
                        resolved_label = f"[{fig_id}] {context[:120]}"
                    warnings.append(f"generic_series_label_resolved: original='{series_label}'")
                    series_label = resolved_label

                # For stoichiometry plots, append x-value to composition
                if x_quantity == 'stoichiometry':
                    composition = f"{series_label} (x={x_val})"
                else:
                    composition = series_label
                
                # --- PHYSICAL VALIDATION ---
                # HARD FILTER: Discard obviously wrong conductivities (>10 S/cm at RT)
                if cond_s_cm > 10.0 and temp_c is not None and temp_c < 100:
                    print(f"   \033[93m[Filter]\033[0m Discarding unrealistic conductivity {cond_s_cm:.2e} S/cm for '{series_label}' at {temp_c}°C")
                    continue
                
                if cond_s_cm > self.MAX_REALISTIC_COND_RT and temp_c is not None and temp_c < 100:
                    warnings.append(f"Suspiciously high conductivity ({cond_s_cm:.4e} S/cm) for solid-state material at {temp_c}°C")
                
                if cond_s_cm < self.MIN_REALISTIC_COND:
                    warnings.append(f"Value below realistic detection limit ({cond_s_cm:.2e} S/cm)")
                
                # Check for negative conductivity
                if cond_s_cm < 0:
                    warnings.append(f"Negative conductivity detected ({cond_s_cm:.2e}), likely extraction error")
                
                # --- CREATE MEASUREMENT RECORD ---
                meas = MeasuredPoint(
                    raw_composition=composition,
                    canonical_formula=None,  # Will be filled by canonicalizer
                    material_definitions=[],
                    
                    # Normalized values
                    normalized_conductivity=cond_s_cm,
                    normalized_temperature_c=temp_c,
                    
                    # Raw extracted values
                    raw_conductivity=str(y_val),
                    raw_conductivity_unit=y_axis_def.get('unit') or raw_y_label,
                    raw_temperature=str(raw_temp_value),
                    raw_temperature_unit=raw_temp_unit,
                    
                    # Metadata
                    source_figure_id=fig_id,
                    source_caption=enhanced_caption,
                    source="figure",
                    confidence="low" if warnings else "high",
                    warnings=warnings
                )
                
                measurements.append(meas)
        
        return measurements

    def _normalize_temperature(self, value: float, x_axis_def: Dict[str, Any]) -> Optional[float]:
        """
        Convert temperature to Celsius based on axis definition
        """
        x_quantity = x_axis_def.get('quantity_type', 'other')
        unit = x_axis_def.get('unit', '').lower()
        
        try:
            if x_quantity == 'temperature_inverse':
                # X-axis is 1000/T (K^-1)
                # Convert to Kelvin first: T = 1000 / x
                kelvin = 1000.0 / value
                celsius = kelvin - 273.15
                return celsius
            
            elif x_quantity == 'temperature_absolute':
                # Direct temperature
                if 'k' in unit or 'kelvin' in unit:
                    return value - 273.15
                elif 'c' in unit or 'celsius' in unit or '°c' in unit:
                    return value
                else:
                    # Assume Kelvin if unclear
                    return value - 273.15
            
            else:
                return 25.0
                
        except (ValueError, ZeroDivisionError):
            return 25.0

    def _normalize_conductivity(self, value: float, y_axis_def: Dict[str, Any],
                                temperature_k: Optional[float] = None) -> float:
        """
        Convert conductivity to S/cm.
        Handles log scales, σT product units, and different unit systems.
        
        IMPORTANT: The SciFigureParser LLM returns LINEAR values even for
        log-scale axes (e.g., it reads 10⁻⁵ and returns 1e-05, not -5).
        Only truly log-space values (negative numbers like -4.8) need 10^x.
        
        Returns None if the measurement is σ₀ (pre-exponential factor) or
        another non-conductivity quantity.
        """
        unit = y_axis_def.get('unit', '').lower()
        title = y_axis_def.get('title_text', '').lower()
        scale = y_axis_def.get('scale_type', 'linear').lower()
        combined = f"{unit} {title}"
        
        # --- FILTER: Detect σ₀ / pre-exponential factor (NOT ionic conductivity) ---
        is_sigma_0 = bool(re.search(
            r'σ[_\s]?0|σ₀|sigma[_\s]?0|pre[\-\s]?exponential|prefactor|σ\s*0|sigma\s*naught',
            combined, re.IGNORECASE
        ))
        if is_sigma_0:
            return None  # Discard: σ₀ is NOT ionic conductivity
        
        # Detect σT product units (e.g., "σT (SK/cm)", "σT / S·K·cm⁻¹", "log(Tσ/K Scm⁻¹)")
        is_sigma_t = bool(re.search(r'σt|tσ|sigma\s*t|t\s*sigma|s\s*k\s*/?\s*cm|sk\s*cm|sk/cm|k\s*s\s*cm', combined, re.IGNORECASE))
        
        # HEURISTIC: Determine if value is in log-space or already linear.
        # SciFigureParser returns LINEAR values even for log-scale axes.
        # Clue: If value is NEGATIVE on a log-scale → it's log₁₀(σ) (e.g., -4.8)
        # If value is POSITIVE (even very small) → LLM already extracted the linear value.
        if scale == 'log' or 'log' in unit:
            if value < 0:
                # Truly in log₁₀ space, e.g., -4.8 means σ = 10⁻⁴·⁸
                sigma = 10 ** value
            else:
                # Positive value → LLM already converted from log-scale axis
                sigma = value
        elif 'ln' in unit:
            if value < 0:
                sigma = np.exp(value)
            else:
                sigma = value
        else:
            sigma = value
        
        # Handle σT product → divide by T(K) to get σ
        if is_sigma_t:
            if temperature_k and temperature_k > 0:
                sigma = sigma / temperature_k
            else:
                # No temperature available — cannot convert, add implicit warning
                # The value will remain as σT, which is wrong but better than 1.0
                pass
        
        # Convert units to S/cm
        # IMPORTANT: Check more-specific prefixed units FIRST (ms, μs) before generic s/cm
        if 'μs/cm' in unit or 'us/cm' in unit or 'μscm' in unit:
            return sigma / 1_000_000.0  # μS/cm to S/cm
        elif 'ms/cm' in unit or 'ms cm' in unit or 'mscm' in unit:
            return sigma / 1000.0  # mS/cm to S/cm
        elif 's/cm' in unit or 's cm' in unit or 'scm' in unit or 'sk/cm' in unit or 'sk cm' in unit:
            return sigma
        elif 's/m' in unit or 's m' in unit:
            return sigma / 100.0  # S/m to S/cm
        else:
            # Assume S/cm if unclear
            return sigma
    
    def _check_arrhenius_slope(self, x_vals: List[float], y_vals: List[float], x_axis_def: Dict[str, Any]) -> Optional[str]:
        """
        Validate Arrhenius plot slope direction.
        For 1000/T plots, log(sigma) should DECREASE as 1000/T INCREASES (cooling).
        """
        x_quantity = x_axis_def.get('quantity_type', 'other')
        
        if x_quantity != 'temperature_inverse':
            return None  # Not an Arrhenius plot
        
        if len(x_vals) < 2:
            return None  # Not enough points
        
        # Calculate approximate slope
        try:
            # Simple linear fit
            n = len(x_vals)
            sum_x = sum(x_vals)
            sum_y = sum(y_vals)
            sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
            sum_x2 = sum(x * x for x in x_vals)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # For Arrhenius: slope should be NEGATIVE
            # (higher 1000/T = lower T = lower conductivity = lower log(sigma))
            if slope > 0:
                return "⚠️ Positive Arrhenius slope detected (log(σ) increases with 1000/T). This is physically unusual and may indicate extraction error."
        
        except (ValueError, ZeroDivisionError):
            pass
        
        return None

    def _extract_temp_from_label(self, label: str) -> Optional[Dict[str, Any]]:
        """
        Extract temperature from series label like "223K", "25°C", "300 K"
        Returns dict with 'celsius', 'raw_value', 'raw_unit' or None
        """
        import re
        
        # Match patterns like: 223K, 25C, 25°C, 300 K, 25 deg C
        pattern = r'(\d+(?:\.\d+)?)\s*°?\s*(K|C|deg\s*C|°C|Celsius|Kelvin)\b'
        match = re.search(pattern, label, re.IGNORECASE)
        
        if not match:
            return None
        
        val_str, unit_str = match.groups()
        
        try:
            val = float(val_str)
            unit_upper = unit_str.upper()
            
            # Convert to Celsius
            if 'K' in unit_upper or 'KELVIN' in unit_upper:
                celsius = val - 273.15
                unit = "K"
            else:
                celsius = val
                unit = "°C"
            
            return {
                'celsius': celsius,
                'raw_value': val_str,
                'raw_unit': unit
            }
        except ValueError:
            return None


    def _extract_temp_from_caption(self, caption: str) -> Optional[Dict[str, Any]]:
        """
        Extract temperature from caption text
        Returns dict with 'celsius', 'raw_value', 'raw_unit' or None
        """
        import re
        
        # Common patterns in captions
        patterns = [
            r'at\s+(\d+(?:\.\d+)?)\s*°?\s*(K|C|°C|Celsius|Kelvin)',
            r'(\d+(?:\.\d+)?)\s*°?\s*(K|C|°C)\s+measurement',
            r'temperature[:\s]+(\d+(?:\.\d+)?)\s*°?\s*(K|C|°C)',
            r'T\s*=\s*(\d+(?:\.\d+)?)\s*°?\s*(K|C|°C)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, caption, re.IGNORECASE)
            if match:
                val_str, unit_str = match.groups()
                try:
                    val = float(val_str)
                    unit_upper = unit_str.upper()
                    
                    if 'K' in unit_upper or 'KELVIN' in unit_upper:
                        celsius = val - 273.15
                        unit = "K"
                    else:
                        celsius = val
                        unit = "°C"
                    
                    return {
                        'celsius': celsius,
                        'raw_value': val_str,
                        'raw_unit': unit
                    }
                except ValueError:
                    continue
        
        return None



processor = MeasurementProcessor()

async def process_image(client, model, img_path, context_dict: dict, max_retries: int = 3, sf_parser: Optional[SciFigureParser] = None):
    if "logo" in img_path.name.lower(): 
        return [], None, True
    
    try:
        fig_id = context_dict.get("id", "Unknown Figure")
        caption = sanitize_caption(context_dict.get("caption", "No caption found."))
        section_title = context_dict.get("section_title", "Unknown Section")
        section_content = context_dict.get("section", "No section content found.")
        # V7: Cross-section methods context
        methods_context = context_dict.get("methods_context", "")
    except Exception as e:
        print(f"   [Context Error] {img_path.name}: {e}")
        return [], None, False

    # --- SCI-FIGURE PARSER INTEGRATION ---
    # Note: sf_parser should be initialized with save_debug=False for production speed.
    if sf_parser:
        try:
            print(f"   🔍 {img_path.name} ({fig_id}): Detecting subplot...")
            # [OPTIMIZED] Using async detection
            detection_result = await sf_parser.detect_subplot_async(str(img_path), "ionic conductivity measurement")
            tracker.track(detection_result, VISION_MODEL)
            
            # print(f"   🔍 {img_path.name} ({fig_id}): Detection result: {detection_result}")

            is_multi = detection_result.get("is_multi_panel", False)
            all_subplots = detection_result.get("subplots", [])
            
            valid_subplots = []
            for subplot in all_subplots:
                if not subplot.get('contains_conductivity_data', False):
                    label = subplot.get('label', 'Unknown')
                    print(f"   \033[93m[Skip]\033[0m {img_path.name} ({fig_id}): Subplot '{label}' does not contain ionic conductivity measurements.")
                    continue
                

                ## Check x-axis type (critical filter)
                x_axis = subplot.get('x_axis', {})
                x_quantity = x_axis.get('quantity_type', '')

                # Reject frequency plot
                if x_quantity == 'frequency':
                    label = subplot.get('label', 'Unknown')
                    print(f"   \033[93m[Skip]\033[0m {img_path.name} ({fig_id}) - {label}: X-axis is frequency (not temperature)")
                    continue

                ## Accept temperature or stoichiometry plots
                if x_quantity in ['temperature_inverse', 'temperature_absolute', 'stoichiometry']:
                    valid_subplots.append(subplot)
                    label = subplot.get('label', 'Unknown')
                    print(f"   ✓ {img_path.name} ({fig_id}) - {label}: Valid plot (X={x_quantity})")
                else:
                    # Uncertain - include with warning
                    subplot['_warning'] = f"Uncertain X-axis type: {x_quantity}"
                    valid_subplots.append(subplot)
            
            if not valid_subplots:
                print(f"   \033[93m[Skip]\033[0m {img_path.name} ({fig_id}): No valid subplots found.")
                return ExtractionResult(measurements=[]), None, True

            if len(valid_subplots) == 1:
                is_multi = False

            all_measurements = []

            async def process_subplot(subplot_data, idx):
                label = subplot_data.get('label', f'Panel {idx+1}')
                print(f"      → Processing {label}...")
                
                # Prepare axis hints
                def clean_axis(ax_data):
                    if not ax_data: 
                        return None
                    return {
                        "title_text": ax_data.get('title', ''),
                        "unit": ax_data.get('unit'),
                        "quantity_type": ax_data.get('quantity_type'),
                        "scale_type": ax_data.get('scale_type', 'linear')
                    }

                axis_hints = {
                    "x_axis": clean_axis(subplot_data.get('x_axis')),
                    "left_y_axis": clean_axis(subplot_data.get('left_y_axis')),
                    "right_y_axis": clean_axis(subplot_data.get('right_y_axis')),
                    "subplot_label": label  # Pass this through for context
                }
                
                # Crop the subplot
                safe_label = re.sub(r'[^a-zA-Z0-9]', '_', label)
                unique_suffix = f"_cropped_{safe_label}"
                
                box_list = subplot_data.get('box_2d')
                box_dict = {
                    'ymin': box_list[0],
                    'xmin': box_list[1],
                    'ymax': box_list[2],
                    'xmax': box_list[3]
                }
                cropped_path = sf_parser.crop_image(
                    str(img_path), 
                    box_dict, 
                    padding=80, 
                    suffix=unique_suffix
                )
                
                # Extract with axis hints — use higher resolution grid for log-scale plots
                y_scale = subplot_data.get('left_y_axis', {}).get('scale_type', 'linear')
                grid_rows = 3 if y_scale == 'log' else 2
                grid_cols = 3 if y_scale == 'log' else 2
                
                sf_result = await sf_parser.extract_data_async(
                    cropped_path, 
                    grid_config={"enabled": True, "rows": grid_rows, "cols": grid_cols}, 
                    context=f"{caption} [Panel: {label}]",
                    axis_hints=axis_hints
                )
                tracker.track(sf_result, VISION_MODEL)
                
                # Process extraction with validation
                clean_measurements = processor.process_extraction(
                    sf_result, 
                    fig_id=f"{fig_id}-{label}", 
                    context=caption,
                    axis_metadata=axis_hints  # NEW: Pass axis info to processor
                )
                
                # Add subplot-specific warnings
                if subplot_data.get('_warning'):
                    for m in clean_measurements:
                        if not m.warnings:
                            m.warnings = []
                        m.warnings.append(subplot_data['_warning'])
                
                return clean_measurements

            # Process all valid subplots in parallel
            subplot_tasks = [process_subplot(sp, i) for i, sp in enumerate(valid_subplots)]
            subplot_results = await asyncio.gather(*subplot_tasks)
            
            # Flatten results
            for res in subplot_results:
                if isinstance(res, list):
                    all_measurements.extend(res)
                elif hasattr(res, 'measurements'):
                    all_measurements.extend(res.measurements)
            
            result = ExtractionResult(measurements=all_measurements)
            
            # V6: Tag all measurements with image filename
            for m in result.measurements:
                m.source_image_filename = img_path.name
            
            # Write to debug log
            log_dir = FILE_DIR if FILE_DIR else img_path.parent
            with open(f"{log_dir}/results_log_v6.json", "a") as f:
                f.write(f"\n\n[SCI-FIGURE DEBUG] {img_path.name}:\n")
                f.write(f"- Total subplots detected: {len(all_subplots)}\n")
                f.write(f"- Valid subplots: {len(valid_subplots)}\n")
                f.write(f"- Measurements extracted: {len(all_measurements)}\n")
                for i, sp in enumerate(valid_subplots):
                    f.write(f"  • {sp.get('label')}: X={sp['x_axis'].get('quantity_type')}, "
                           f"Y={sp['left_y_axis'].get('quantity_type')}\n")
            if len(result.measurements) > 0:
                print(f"   ✓ {img_path.name} ({fig_id}): Found {len(result.measurements)} points total via SciFigureParser")
            
            return result, None, True 
            
        except Exception as e:
            print(f"   ⚠️ {img_path.name}: SciFigureParser failed: {e}. Falling back to standard processing...")
            # Fall through to standard processing

    # Standard processing as fallback
    try:
        img_bytes = img_path.read_bytes()
    except Exception as e:
        print(f"   [Image Read Error] {img_path.name}: {e}")
        return [], None, False
        
    prompt = f"""
    Analyze this scientific image and determine if it contains ionic conductivity measurements.

    **Metadata:**
    - Figure ID: {fig_id}
    - Caption: {caption}
    - Found in Section: {section_title}
    - Section Content: {section_content}

    *** EXAMPLES OF TEMPERATURE EXTRACTION ***
    Case 1: Standard Celsius
    Input: "Temperature was maintained at 25 °C"
    Output:
    {{
    "raw_temperature": "25",
    "raw_temperature_unit": "Celsius"
    }}

    Case 2: Standard Kelvin
    Input: "Measured at 298 K"
    Output:
    {{
    "raw_temperature": "298",
    "raw_temperature_unit": "K"
    }}

    Case 3: Arrhenius Plot (Inverse Temperature)
    Input: "The x-axis shows 1000/T (K⁻¹) ranging from 2.0 to 3.5"
    Output:
    {{
    "raw_temperature": "2.0",
    "raw_temperature_unit": "1000/T (K-1)"
    }}

    Case 4: Stoichiometry Plot (Composition vs Conductivity)
    Input: "The x-axis represents the variable x in Li1+xAlxTi2-x(PO4)3, and caption states 298 K."
    Output:
    {{
    "raw_temperature": "298",
    "raw_temperature_unit": "K",
    "raw_composition": "[Material Name] (x=0.2)"
    }}

    **Task**:
    **Step 1: Classify the image**
    Is this a:
    - [ ] Data plot with conductivity values (Arrhenius plot, stoichiometry plot, bar chart, etc.)
    - [ ] Stability/aging plot (conductivity vs. time/days/cycles/hours)
    - [ ] Table with conductivity measurements
    - [ ] Structural diagram / schematic / photo (NO DATA)

    **Step 2: Extract (ONLY if you checked one of the first three options)**
    If this contains conductivity data, extract measurements.
    - CRITICAL: Detect if the X-axis is stoichiometry (e.g., 'x', 'z', 'composition').
    - If it is stoichiometry, extract the x-value and append it to 'raw_composition' (e.g. "Al (x=0.2)").
    - If the caption specifies a fixed temperature for the whole plot, use it for 'raw_temperature'.
    - If the figure explicitly CITES another paper (e.g. "Data from Ref. [12]", "Comparison with [15]"), use source="cited_figure". Otherwise source="figure".

    **For Stability/Aging Plots:**
    - Extract EACH data point (each time step) as a SEPARATE measurement.
    - The X-axis is TIME (days, hours, cycles) — do NOT use it as temperature.
    - Use the time value as `aging_time` (e.g., "5 days") and append it to `raw_composition` as a suffix (e.g., "PEO-LLZO [day 5]").
    - Set `measurement_condition` to describe the aging state (e.g., "after aging 5 days").
    - Get temperature from the caption or surrounding text context, NOT from the x-axis.

    Otherwise return empty.

    **Step 3: Extract Material Definitions**
    - If the image contains a description of the material composition, extract it.
    - Return the definition in a concise format.

    Return JSON with measurements array (can be empty). Do not convert units yet.
    """

    # print('>>>> prompt >>>>\n', prompt)
    # print('\n\n')

    content = types.Content(
        parts=[
            types.Part(text=prompt),
            types.Part(
                inline_data=types.Blob(
                    mime_type="image/png" if img_path.suffix.lower() == '.png' else "image/jpeg",
                    data=base64.b64encode(img_bytes).decode('utf-8')
                ),
                media_resolution={"level": "media_resolution_high"}
            )
        ]
    )


    # Re-try logic
    last_exception = None
    log_dir = FILE_DIR if FILE_DIR else img_path.parent
    for attempt in range(1, max_retries+1):
        with open(f"{log_dir}/results_log_v6.json", "a") as f:
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[content],
                    config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=ExtractionResult.model_json_schema(),
                    temperature=1.0,
                    max_output_tokens=16384,
                    # thinking_config=types.ThinkingConfig(thinking_level="medium")
                )
            )
                # response = types.Content(text="")
                # response.text = ""
            
                if not response.text:
                    print(f"   ⚠️  {img_path.name}: Empty response (likely safety filter)")
                    f.write(f"\n\n--- [Image Warning] Empty response for {img_path.name}")
                    last_exception = "Empty response"
                    await asyncio.sleep(1 * attempt)
                    continue

                # Write extensive debugging info to the file
                f.write(f"\n\n[DEBUG] {img_path.name}:\n")
                f.write(f"- Response length: {len(response.text) if response.text else 0} chars\n")
                f.write(f"- Usage metadata: {response.usage_metadata}\n")
                f.write(f"- chars: {response.text}\n")

                result = ExtractionResult.model_validate_json(response.text)
            
                # Tag the source and also check the source if it's other than figure we should skip those measurements
                for m in result.measurements:
                    m.source_figure_id = fig_id
                    m.source_caption = caption
                    m.source_image_filename = img_path.name  # V6: track image filename
                    if m.source != "figure":
                        print(f"   \033[91m[Image Warning]\033[0m {img_path.name}: {m.raw_composition} not from figure, skipping measurement")
                        m.raw_composition = "Not Specified"
                        m.raw_temperature = "Not Specified"
                        m.normalized_temperature_c = None
                        m.confidence = "low"
                    if m.raw_composition == "Not Specified" and caption:
                        # Temporary fallback: put caption in composition so canonicalizer sees it
                        m.raw_composition = f"Series from {fig_id}" 
                
                if len(result.measurements) > 0:
                    print(f"   ✓ {img_path.name} ({fig_id}): Found {len(result.measurements)} points")
                return result, response, True
            except (ValidationError, json.JSONDecodeError) as e:
                # These are REAL failures - malformed JSON
                last_exception = e
                print(f"   [Retry {attempt}/{max_retries}] {img_path.name}: {str(e)[:100]}...")
                await asyncio.sleep(1 * attempt)
                
            except Exception as e:
                # API errors
                last_exception = e
                print(f"   [Retry {attempt}/{max_retries}] {img_path.name}: {e}")
                await asyncio.sleep(1 * attempt)
        # All retries exhausted
    print(f"   ❌ {img_path.name}: Failed after {max_retries} attempts - {last_exception}")
    return ExtractionResult(measurements=[]), None, False


async def process_table_node(client, model, table_data: dict, max_retries: int = 3, paper_context: dict = None):
    """
    Extract data from a Markdown table found via regex.
    Enhanced prompt handles multi-level headers and NASICON-type acronyms.
    V7: Accepts optional paper_context for cross-section methods context.
    """
    # V7: Build methods context preamble if available
    methods_preamble = ""
    if paper_context:
        proc_summary = paper_context.get('experimental_procedure_summary', '')
        nomenclature = paper_context.get('nomenclature_key', '')
        if proc_summary or nomenclature:
            methods_preamble = f"""\n    **PAPER CONTEXT (for processing method identification):**
    Experimental Procedures: {proc_summary[:1500]}
    Sample Nomenclature: {nomenclature[:500]}\n"""

    prompt = f"""
    Extract ionic conductivity data points from this Markdown table.
    
    **Table Caption:** {table_data['caption']}
    {methods_preamble}
    **Table Content:**
    ```markdown
    {table_data['content']}
    ```

    **IMPORTANT INSTRUCTIONS:**
    1. This table may have MULTI-LEVEL HEADERS (e.g., "BULK" and "G.B." spanning sub-columns 
       like Ea, σ0, σRT). Look for these hierarchical groupings.
    2. PRIORITIZE extracting **σRT** (room-temperature ionic conductivity) over σ0 (pre-exponential factor) 
       or Ea (activation energy). σ0 values are typically very large (10³-10⁵) and are NOT conductivity.
    3. For EACH row, extract BOTH bulk AND grain boundary σRT values as separate measurements.
    4. DO NOT resolve acronyms or expand formulas — just use the EXACT sample name from the table.
    5. For multi-row tables with Unicode superscripts (e.g., "10⁻⁵"), parse them as scientific notation.
    6. Tag each measurement with whether it's "bulk" or "grain_boundary" in the material_definitions field.
    7. If paper context is provided above, use it to fill in the processing_method field.
    8. For polymer-ceramic composite tables, capture the polymer matrix, filler type, and loading amount.
       Example: if a table lists various wt% of LLZO in PEO-LiTFSI, extract the full composition.
    9. Set material_class to "Polymer", "Ceramic", or "Composite" based on what the material IS.
       Do NOT use "Cited" — provenance is tracked via source="cited_table" instead.
    10. If the table compares with literature data or cites references, use source="cited_table".
    11. If a conductivity RANGE is given (e.g., "10⁻² – 10⁻⁴ S cm⁻¹"), extract BOTH bounds as separate
        measurements with warnings "range_upper_bound" / "range_lower_bound".
    12. If the table contains time-series data (e.g., conductivity at different aging times or cycle counts),
        extract EACH time point as a separate measurement. Set `aging_time` and `measurement_condition` accordingly.

    **Task**:
    Extract all Ionic Conductivity measurements (σRT preferred, σ0 if σRT not available).
    For each:
    - raw_composition: EXACT sample name from the table (e.g., "PEO-LiTFSI/LLZO (50 wt%)")
    - raw_conductivity: Numeric value (the σRT column value)
    - raw_conductivity_unit: Unit (e.g., "S cm-1")
    - raw_temperature: "room temperature" (since σRT is at room temperature) or explicit temperature
    - raw_temperature_unit: "Celsius"
    - material_class: "Polymer" / "Ceramic" / "Composite" (not "Cited" — use source field for provenance)
    - material_definitions: ["bulk"] or ["grain_boundary"] to indicate which region
    - source: "markdown_table" or "cited_table"
    - aging_time: If applicable (e.g., "5 days", "freshly made"). Null otherwise.
    - measurement_condition: If applicable (e.g., "after 100 cycles"). Null otherwise.
    - confidence: "high"

    Return JSON with measurements array.
    """

    last_exception = None
    log_dir = FILE_DIR 
    for attempt in range(1, max_retries+1):
        with open(f"{log_dir}/results_log_v6.json", "a") as f:
            try:
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=ExtractionResult.model_json_schema(),
                        temperature=0.0,
                        max_output_tokens=16384,
                    )
                )
            
                if not response.text:
                    print(f"   ⚠️  {table_data['caption']}: Empty response")
                    last_exception = "Empty response"
                    continue

                f.write(f"\n\n[TABLE DEBUG] {table_data['caption']}:\n")
                f.write(f"- Response: {response.text}\n")

                result = ExtractionResult.model_validate_json(response.text)
                
                # Tag metadata
                for m in result.measurements:
                    m.source_caption = table_data['caption']
                    m.source_figure_id = table_data['caption'].split(':')[0] if ':' in table_data['caption'] else "Table"

                if len(result.measurements) > 0:
                    print(f"   ✓ {table_data['caption']}: Found {len(result.measurements)} points")
                return result, response, True
            except Exception as e:
                last_exception = e
                await asyncio.sleep(1 * attempt)
                
    print(f"   ❌ {table_data['caption']}: Failed - {last_exception}")
    return ExtractionResult(measurements=[]), None, False


# ==============================================================================
# 6. Main Orchestrator
# ==============================================================================
async def classify_material_classes_llm(client, measurements, paper_context, model_name="gemini-flash-latest"):
    """
    Use LLM to classify measurements that don't have material_class set.
    Replaces brittle keyword-based classification with a generalizable LLM call.
    """
    unclassified = [(i, m) for i, m in enumerate(measurements) if not m.material_class]
    if not unclassified:
        return measurements

    items = "\n".join([
        f"ID {i}: composition='{m.raw_composition}', canonical='{m.canonical_formula or 'N/A'}', source='{m.source or 'N/A'}'"
        for i, m in unclassified
    ])

    context = paper_context.get('material_systems_overview', '') if paper_context else ''

    prompt = f"""Classify each material measurement into exactly one of these categories based on what the material IS:
- "Polymer": Neat polymer electrolyte with NO ceramic filler (e.g., PEO-LiTFSI, PAN-LiClO4, "0 wt% filler" samples)
- "Ceramic": Neat ceramic electrolyte (e.g., LLZO, LAGP, LATP, Li7La3Zr2O12)
- "Composite": Polymer+ceramic composite electrolyte (e.g., PEO-LiTFSI with LLZO filler, any sample with nonzero wt%/vol% ceramic loading)

Classify by the material type, NOT by provenance. Cited data is already tracked in the source field.
A cited LLZO measurement is still "Ceramic". A cited PEO-LLZO composite is still "Composite".

Paper context: {context}

Items to classify:
{items}

Return JSON: {{"classifications": {{"<ID>": "<class>"}} }}
Only use the three classes above."""

    try:
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            )
        )
        result = json.loads(response.text)
        classifications = result.get("classifications", {})

        applied = 0
        valid_classes = {'Polymer', 'Ceramic', 'Composite'}
        for id_str, cls in classifications.items():
            digits = re.search(r'\d+', str(id_str))
            if digits:
                idx = int(digits.group())
                if 0 <= idx < len(measurements) and cls in valid_classes:
                    measurements[idx].material_class = cls
                    applied += 1

        if response:
            tracker.track(response, model_name)

        print(f"      LLM classified {applied}/{len(unclassified)} measurements")

    except Exception as e:
        print(f"   ⚠️ LLM material_class classification failed: {e}")

    return measurements


async def run_pipeline(markdown_file, asset_dir, model):
    global FILE_DIR
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    try:
        sem = asyncio.Semaphore(NUM_WORKERS)

        # Create a log file
        FILE_DIR = markdown_file.parent
        with open(f"{FILE_DIR}/results_log_v6.json", "w") as f:
            f.write(f"\n\n--- [Document] {markdown_file.name} ---\n")
        
        # 1. Parse Markdown & Build Context Map
        text_content = markdown_file.read_text(encoding='utf-8')
        parser = MarkdownContextParser()
        # 1. Parse Sections and Title (New Functionality)
        doc_title, sections = parser.parse_structure(text_content)
        all_images = parser.parse_images(text_content)
        all_tables = parser.parse_tables(text_content)
        
        # V6: Detect review articles
        review_keywords = ['review', 'survey', 'overview', 'perspective', 'progress', 'recent advances', 'state of the art', 'state-of-the-art']
        is_review_article = any(kw in doc_title.lower() for kw in review_keywords)
        if is_review_article:
            print(f"   📋 Detected REVIEW ARTICLE: {doc_title}")
        
        # 2. Linking (The Magic Step)
        parser.link_assets_to_sections(sections, all_images, all_tables)

        # 2.1 Table De-duplication: Replace MD tables with placeholders in section contents
        if all_tables:
            print(f"   ✂️ De-duplicating {len(all_tables)} tables from Markdown text...")
            for table_info in all_tables:
                for sec in sections:
                    if table_info.content in sec.content:
                        placeholder = f"\n\n[{table_info.id}: {table_info.caption} processed separately]\n\n"
                        sec.content = sec.content.replace(table_info.content, placeholder)
                        print(f"       - Replaced '{table_info.id}' in section '{sec.title}'")

        # 2.5 Initialize SciFigureParser - [OPTIMIZED] save_debug=False for speed
        sf_parser = SciFigureParser(api_key=api_key, model_name=VISION_MODEL, debug=True, save_debug=False)

        # 3. Reporting
        print(f"   --- Document: {doc_title}")
        print(f"   --- Found: {len(sections)} Sections")
        print(f"   --- Found: {len(all_images)} Figures, {len(all_tables)} Tables")

        # check the length of all images found are the same as the image files in the asset directory
        img_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        img_files = []
        for pattern in img_patterns:
            img_files.extend(list(asset_dir.glob(pattern)))
        img_files = sorted(img_files)
        if len(all_images) != len(img_files):
            print("   \033[91m[Warning]\033[0m Number of images found does not match number of image files in asset directory")

        # Example Output
        print("\n   --- Found Assets in Sections:")
        for sec in sections:
            has_assets = sec.images or sec.tables
            if has_assets:
                print(f"\n   >>> Section '{sec.title}' contains:")
                if sec.images: print(f"       - {len(sec.images)} Images: {[img.id for img in sec.images]}")
                if sec.tables: print(f"       - {len(sec.tables)} Tables: {[tab.id for tab in sec.tables]}")
        print("\n   --- Found Assets in Sections ---\n\n")
        
        # Build Image Context Registry
        # We need to map filename -> {caption, section_info} so the image processor can find it.
        image_context_registry = {}
        
        # Initialize with basic info from all_images
        for img in all_images:
            image_context_registry[img.filename.lower()] = {
                "id": img.id,
                "caption": img.caption,
                "section_title": "Unassigned", # Default
                "section_summary": ""
            }

        # Enrich with Section Data
        # (Since sections "own" images now, we reverse-lookup to fill the registry)
        for sec in sections:
            for img in sec.images:
                if img.filename.lower() in image_context_registry:
                    image_context_registry[img.filename.lower()]["section_title"] = sec.title
                    # We can pass the first 500 chars of the section as "background context" for the image
                    nearby_text = parser._extract_nearby_text(img, sec.content, window_lines=5, max_chars=1000)
                    image_context_registry[img.filename.lower()]["section"] = nearby_text
                    # If markdown parser couldn't find fig ID, scan nearby section text for it
                    if image_context_registry[img.filename.lower()]["id"] == "Unknown":
                        fig_match = parser.REF_PATTERN.search(nearby_text)
                        if fig_match:
                            recovered_id = parser._normalize_id(fig_match.group(1), fig_match.group(2), fig_match.group(3))
                            image_context_registry[img.filename.lower()]["id"] = recovered_id
                            print(f"   [Context] Recovered figure ID '{recovered_id}' from nearby text for {img.filename}")

        # Fix #4: Build figure ID map and update registry with resolved figure IDs
        figure_id_map = parser.build_figure_id_map(all_images, sections)
        if figure_id_map:
            print(f"   🗺️ Figure ID map: {figure_id_map}")
            for filename_lower, fig_id in figure_id_map.items():
                if filename_lower in image_context_registry:
                    current_id = image_context_registry[filename_lower].get("id", "Unknown")
                    if current_id == "Unknown" or "(inferred)" not in fig_id:
                        image_context_registry[filename_lower]["id"] = fig_id

        # V7: Extract paper-level context for cross-section injection
        print("   🔄 Extracting paper-level context for cross-section injection...")
        paper_context = await extract_paper_context(client, doc_title, sections, model_name="gemini-2.5-flash")

        # V7: Build methods context string for injection into image/table calls
        methods_context_str = ""
        if paper_context:
            proc_summary = paper_context.get('experimental_procedure_summary', '')
            nomenclature = paper_context.get('nomenclature_key', '')
            systems_overview = paper_context.get('material_systems_overview', '')
            parts = []
            if proc_summary:
                parts.append(f"PROCESSING: {proc_summary[:1000]}")
            if nomenclature:
                parts.append(f"NOMENCLATURE: {nomenclature[:500]}")
            if systems_overview:
                parts.append(f"MATERIALS: {systems_overview[:500]}")
            methods_context_str = " | ".join(parts)

        # V7: Inject methods context into image context registry
        if methods_context_str:
            print(f"   📋 Injecting methods context ({len(methods_context_str)} chars) into image/table extraction")
            for key in image_context_registry:
                image_context_registry[key]["methods_context"] = methods_context_str

        # Build tasks
        tasks = []
        task_sources = []  # V6: Track provenance for each task
        # Extract from text (section-by-section)
        all_measurements = []
        material_defs = [] # You might want to accumulate these or merge them later

        # We process sections in parallel batches (or sequentially if rate limits matter)
        skip_sections = ['acknowledgements', 'references']
        for sec in sections:
            if any(skip in sec.title.lower() for skip in skip_sections):
                print(f"   \033[91m[Warning]\033[0m Skipping section: {sec.title}")
                continue
            if PROCESS_TEXT:
                tasks.append(safe_text_call_with_retry(sec, client, TEXT_MODEL, sem, paper_context=paper_context))
                task_sources.append(('text', sec))

        # # 3. Extract from Images (Parallel)
        # Add Image Tasks
        img_files = []
        for pattern in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            # Only grab original assets, skip debug/cropped/detected files
            candidates = list(asset_dir.glob(pattern))
            for cand in candidates:
                if "debug" not in cand.name.lower() and "cropped" not in cand.name.lower() and "detected" not in cand.name.lower():
                    img_files.append(cand)
        
        for img_path in sorted(img_files):
            # Retrieve context from registry (as you already do)
            filename = img_path.name

            # this is a temporary fix
            try:
                context = image_context_registry[filename]
            except:
                context = image_context_registry.get('_'+filename, {"id": "Unknown", "caption": "No caption"})
            if PROCESS_IMAGE:
                tasks.append(safe_image_call_with_retry(img_path, context, client, VISION_MODEL, sem, sf_parser=sf_parser))
                task_sources.append(('image', context))

        # Add Table Tasks
        for table_info in all_tables:
            table_data = {
                'caption': f"{table_info.id}: {table_info.caption}",
                'content': table_info.content
            }
            # V6: Find which section this table belongs to
            table_section_title = 'Unknown'
            for sec in sections:
                if table_info in sec.tables:
                    table_section_title = sec.title
                    break
            if PROCESS_TABLE:
                tasks.append(safe_table_call_with_retry(table_data, client, TEXT_MODEL, sem, paper_context=paper_context))
                task_sources.append(('table', {'section_title': table_section_title, 'table_id': table_info.id}))

        # 4. Execute Simultaneously
        print(f"   ... Processing {len(tasks)} items (Text + Images) simultaneously ...")
        results = await asyncio.gather(*tasks)

        # Merge Image Results
        all_measurements = []
        failed_count = 0
        skipped_count = 0

        for i, res in enumerate(results):
            # Handle exceptions from gather
            if isinstance(res, Exception):
                print(f"   ❌ Task {i} raised exception: {res}")
                failed_count += 1
                continue
            
            # V6: Get provenance info for this task
            task_type, task_meta = task_sources[i] if i < len(task_sources) else ('unknown', None)
            
            # Check if this is an image result (tuple) or text result (object)
            if isinstance(res, tuple): # this is for image results and table results
                result, success = res
                if success:
                    if len(result.measurements) == 0:
                        skipped_count += 1  # No data found (valid)
                    else:
                        # V6: Tag image/table measurements with section provenance
                        for m in result.measurements:
                            if task_type == 'image' and isinstance(task_meta, dict):
                                sec_title = task_meta.get('section_title', 'Unknown')
                                m.source_section = sec_title
                                m.source_section_type = classify_section_type(sec_title)
                            elif task_type == 'table' and isinstance(task_meta, dict):
                                sec_title = task_meta.get('section_title', 'Unknown')
                                m.source_section = sec_title
                                m.source_section_type = classify_section_type(sec_title)
                        all_measurements.extend(result.measurements)
                else:
                    failed_count += 1  # Actual failure
            elif hasattr(res, 'measurements'):
                # Text extraction result
                if len(res.measurements) == 0:
                    skipped_count += 1
                else:
                    # V6: Tag text-extracted measurements with section provenance
                    if task_type == 'text' and task_meta is not None:
                        sec = task_meta
                        sec_type = classify_section_type(sec.title)
                        for m in res.measurements:
                            m.source_section = sec.title
                            m.source_section_type = sec_type
                    all_measurements.extend(res.measurements)
                    # Accumulate material definitions from text extractions
                    for m in res.measurements:
                        if m.material_definitions:
                            material_defs.extend(m.material_definitions)
            elif res is None:
                failed_count += 1

        # 3.5 Extract Abbreviation Map (NEW - runs once per paper)
        print("   ... Extracting abbreviation map from paper text...")
        doc_abstract = ""
        for sec in sections:
            if 'abstract' in sec.title.lower() or 'introduction' in sec.title.lower():
                doc_abstract = sec.content[:1000]
                break
        abbreviation_map = await extract_abbreviation_map(client, doc_title, sections, model_name="gemini-2.5-flash")

        # 3.6 Extract Processing Info (NEW)
        print("   ... Extracting processing info from paper text...")
        processing_map = await extract_processing_info(client, doc_title, sections, model_name="gemini-2.5-flash")

        # 4. Canonicalize Names (Solves Problem 3)
        # We pass the definitions found in text + document-level context + abbreviation map + processing map
        print("   ... Canonicalizing Materials...")
        all_measurements = await canonicalize_materials(client, all_measurements, material_defs, model_name=TEXT_MODEL, doc_title=doc_title, doc_abstract=doc_abstract, abbreviation_map=abbreviation_map, processing_map=processing_map)
        
        # V7 Option A: Post-hoc processing_method recovery via canonical_formula matching
        if processing_map:
            null_proc_count_before = sum(1 for m in all_measurements if not m.processing_method and m.source != 'cited_text')
            general_method = processing_map.get("ALL_SAMPLES")
            for m in all_measurements:
                if m.processing_method or m.source == 'cited_text':
                    continue
                # Try matching canonical_formula against processing_map keys
                matched = False
                if m.canonical_formula:
                    for k, v in processing_map.items():
                        if k == "ALL_SAMPLES" or k.startswith('__'):
                            continue
                        k_lower = k.lower().strip()
                        formula_lower = m.canonical_formula.lower().strip()
                        # Exact match
                        if k_lower == formula_lower:
                            m.processing_method = v
                            matched = True
                            break
                        # Word-boundary match (key >= 3 chars)
                        if len(k_lower) >= 3 and re.search(r'\b' + re.escape(k_lower) + r'\b', formula_lower):
                            m.processing_method = v
                            matched = True
                            break
                # Fallback to general method
                if not matched and general_method:
                    m.processing_method = general_method
            null_proc_count_after = sum(1 for m in all_measurements if not m.processing_method and m.source != 'cited_text')
            recovered = null_proc_count_before - null_proc_count_after
            if recovered > 0:
                print(f"   🔧 V7 Option A: Recovered processing_method for {recovered} measurements via canonical_formula matching")

        # 4.5 De-duplicate Text Measurements (Solves Problem 4: Duplicate Extractions)
        print("   ... De-duplicating Text Measurements...")
        all_measurements = deduplicate_text_measurements(all_measurements)

        # V8: Post-processing — classify material_class using LLM for unset measurements
        print("   ... V8: Classifying material_class via LLM...")
        all_measurements = await classify_material_classes_llm(client, all_measurements, paper_context)

        mc_counts = {}
        for m in all_measurements:
            mc = m.material_class or 'Unknown'
            mc_counts[mc] = mc_counts.get(mc, 0) + 1
        print(f"      material_class distribution: {mc_counts}")

        # # 5. Normalize Values (Solves Problem 2)
        print("   ... Normalizing Units & Temperatures...")
        for m in all_measurements:
            
            #check if it has already been normalized
            if m.normalized_conductivity and m.normalized_temperature_c:
                continue
            
            norm = calculate_standard_units(m.raw_conductivity, m.raw_conductivity_unit, m.raw_temperature, m.raw_temperature_unit)
            m.normalized_conductivity = norm['cond']
            m.normalized_temperature_c = norm['temp']
            
            # Fallback: if canonical name is still empty, copy raw
            if not m.canonical_formula:
                m.canonical_formula = m.raw_composition

        # Fix #5: Populate source_paragraph_id from source_paragraph_indices
        for m in all_measurements:
            if m.source_paragraph_indices:
                m.source_paragraph_id = ",".join(f"P{i}" for i in m.source_paragraph_indices)

        # Fix #6: Flag cited data from intro/review sections for ML exclusion
        excluded_count = 0
        for m in all_measurements:
            if m.source in ('cited_text', 'cited_figure', 'cited_table'):
                sec_type = m.source_section_type or ''
                if sec_type in ('intro', 'meta', 'conclusion'):
                    m.exclude_from_ml = True
                    excluded_count += 1
        if excluded_count:
            print(f"   🚫 Flagged {excluded_count} cited measurements from intro/meta sections for ML exclusion")

        # Report statistics
        print(f"\n   📊 Extraction Summary:")
        print(f"      - Extracted: {len(all_measurements)} measurements")
        print(f"      - Skipped (no data): {skipped_count} images")
        print(f"      - Failed: {failed_count} items")

        pipeline_stats = {
            "extracted_count": len(all_measurements),
            "skipped_images": skipped_count,
            "failed_items": failed_count,
            "total_sections_processed": len(sections),
            "total_images_processed": len(img_files)
        }

        # [CHANGED] Return stats along with data + paper_context
        return all_measurements, material_defs, pipeline_stats, is_review_article, paper_context
    finally:
        # [NEW] Ensure the Gemini client is closed gracefully to prevent SSL/Event loop errors on exit
        if 'client' in locals() and client:
            await client.aio.aclose()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('markdown_file', type=Path)
    parser.add_argument('--asset_dir', type=Path)
    parser.add_argument('--model', default='gemini-3-flash-preview')
    args = parser.parse_args()

    start_time = time.time()
    measurements, material_definitions, stats, is_review_article, paper_context = asyncio.run(run_pipeline(args.markdown_file, args.asset_dir or args.markdown_file.parent, args.model))

    def _is_missing_temp(value) -> bool:
        if value is None:
            return True
        text = str(value).strip().lower()
        return text in {"", "null", "none", "n/a", "unknown", "not specified"}

    def _has_conductivity_signal(m: MeasuredPoint) -> bool:
        if m.normalized_conductivity is not None:
            return True
        raw_c = str(m.raw_conductivity).strip().lower() if m.raw_conductivity is not None else ""
        return raw_c not in {"", "null", "none", "n/a", "unknown", "not specified"}

    def apply_missing_temperature_assumptions(measurements_list: List[MeasuredPoint], assumed_rt_c: float = 25.0) -> int:
        """Fill missing temperatures with assumed room temperature and annotate warnings."""
        assumed_count = 0
        assumption_warning = f"Assumed room temperature ({assumed_rt_c:.0f} °C) because temperature was missing."

        for measurement in measurements_list:
            if measurement.normalized_temperature_c is not None:
                continue
            if not _has_conductivity_signal(measurement):
                continue

            measurement.normalized_temperature_c = assumed_rt_c
            if _is_missing_temp(measurement.raw_temperature):
                measurement.raw_temperature = str(int(assumed_rt_c))
            if _is_missing_temp(measurement.raw_temperature_unit):
                measurement.raw_temperature_unit = "room temperature"

            if measurement.warnings is None:
                measurement.warnings = []
            if assumption_warning not in measurement.warnings:
                measurement.warnings.append(assumption_warning)

            assumed_count += 1

        return assumed_count

    assumed_temp_count = apply_missing_temperature_assumptions(measurements)
    if assumed_temp_count:
        print(f"   ℹ️  Applied room-temperature assumption to {assumed_temp_count} measurement(s) with missing temperature.")
    
    elapsed_time = time.time() - start_time

    # Save
    out_path = args.markdown_file.parent / "robust_results_v8.json"
    output_data = {
        'doc_name': args.markdown_file.stem,
        'extraction_version': 'v8',
        'is_review_article': is_review_article,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'execution_time_seconds': round(elapsed_time, 2),
        
        # 1. Configuration & Provenance
        'config': {
            'vision_model': VISION_MODEL,
            'text_model': TEXT_MODEL,
            'normalization_engine': "v2_hybrid_llm_python",
        },
        
        # 2. Cost & Usage (Crucial for tracking)
        'cost_summary': {
            'total_input_tokens': tracker.total_input_tokens,
            'total_output_tokens': tracker.total_output_tokens,
            'total_cost_usd': round(tracker.total_cost_usd, 4),
            'call_counts': tracker.call_counts
        },
        
        # 3. Pipeline Health Stats
        'extraction_stats': stats,
        
        # V7: Paper-level context for downstream use
        'paper_context': paper_context or {},
        
        # 4. The Data
        'material_count': len(measurements),
        'measurements': [m.model_dump() for m in measurements],
        'material_definitions': material_definitions
    }

    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDone! Extracted {len(measurements)} points.")
    print(f"Saved to: {out_path}")
    tracker.print_summary()

if __name__ == "__main__":
    main()