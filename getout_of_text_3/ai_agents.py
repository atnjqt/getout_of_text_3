"""AI Agent Tools for getout_of_text_3
=====================================

This module provides LangChain-compatible Tool classes for analyzing a DIY
SCOTUS corpus using keyword-in-context (KWIC) extraction plus optional
post-filtered analysis. They are adapted from the example notebook
(`examples/ai/langchain.ipynb`) for library use.

Two complementary tools are exposed:

1. ScotusAnalysisTool
   - Performs an internal keyword search over a SCOTUS corpus dict
	 (volume -> DataFrame[case_id, text]) using the public
	   got3.search_keyword_corpus
   - Dynamically shrinks context window if the prompt would exceed model
	 token constraints (character length heuristic)
   - Returns raw model output (narrative text)

2. ScotusFilteredAnalysisTool
   - Accepts already filtered JSON (string or dict) produced by upstream
	 code (e.g. a previous search + human curation)
   - Offers extraction_strategy: 'first' | 'all' | 'raw_json'
   - Provides debug metrics (raw vs extracted chars, approx token counts)
   - Optional max_contexts cap
   - Optional structured JSON output with salvage of malformed responses
   - Preflight approximate token rejection (char/4 heuristic)

Both tools enforce a strict "ONLY use provided contexts" rule to reduce
hallucinations beyond the user-supplied snippets.

NOTE: These tools rely on the caller to supply an already initialized
LangChain chat model instance (e.g. via `langchain.chat_models.init_chat_model`
for AWS Bedrock). The library intentionally does not own provider creds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type, Union
import json
import re

try:
	from langchain.tools import BaseTool
except ImportError as e:  # pragma: no cover - optional dependency guard
	raise ImportError(
		"LangChain is required for ai_agents functionality. Install with: pip install langchain"
	) from e

# Use native Pydantic v2 imports (LangChain >=0.3 aligned with pydantic v2)
from pydantic import BaseModel, Field

# External imports from package for search function will happen lazily

# ============================================================================
# Live SCOTUS keyword search + analysis tool
# ============================================================================

class ScotusAnalysisInput(BaseModel):
	"""Input schema for performing a fresh keyword search over SCOTUS corpus."""

	keyword: str = Field(description="Keyword/phrase to search in SCOTUS cases")
	analysis_focus: Optional[str] = Field(
		default="general",
		description="Focus of analysis: 'general', 'evolution', 'judicial_philosophy', or 'custom'",
	)


class ScotusAnalysisTool(BaseTool):
	"""Tool that searches the SCOTUS corpus then analyzes retrieved contexts.

	Implementation notes:
	- Expects `db_dict_formatted` mapping volume -> DataFrame(case_id, text)
	- Uses getout_of_text_3.search_keyword_corpus for KWIC retrieval
	- Applies dynamic context window shrinking based on prompt char length
	- `max_tokens` is passed via model initialization (Bedrock provider)
	- Returns the model's raw response text
	"""

	name: str = "scotus_analysis"
	description: str = (
		"Analyzes SCOTUS cases for a given keyword after performing an internal search. "
		"Do NOT provide pre-filtered results to this tool."
	)
	args_schema: Type[BaseModel] = ScotusAnalysisInput
	# Excluded runtime-only attributes (not part of tool argument schema)
	model: Any = Field(default=None, exclude=True)  # LangChain chat model instance
	db_dict_formatted: Any = Field(default=None, exclude=True)

	def __init__(self, model: Any, db_dict_formatted: Any, **kwargs):
		super().__init__(**kwargs)
		self.model = model
		self.db_dict_formatted = db_dict_formatted

	# Sync run (primary in notebooks / scripts)
	def _run(self, keyword: str, analysis_focus: str = "general") -> str:  # noqa: D401
		try:
			return self._execute(keyword, analysis_focus)
		except Exception as e:  # pragma: no cover - defensive
			msg = f"Error analyzing SCOTUS results: {e}"
			return msg

	async def _arun(self, keyword: str, analysis_focus: str = "general") -> str:  # noqa: D401
		return self._run(keyword, analysis_focus)

	def _execute(self, keyword: str, analysis_focus: str) -> str:
		import getout_of_text_3 as got3
		# Fallback to large token allowance if model missing attribute
		max_tokens = getattr(getattr(self.model, "_lc_kwargs", {}), "get", lambda _k, _d=None: None)
		try:  # attempt to pull from model kwargs
			max_tokens = self.model._lc_kwargs.get("max_tokens", 128000)  # type: ignore[attr-defined]
		except Exception:  # pragma: no cover
			max_tokens = 128000

		base_context_words = 20
		context_words = base_context_words
		attempts = 0
		final_prompt: Optional[str] = None
		results_dict: Optional[Dict[str, Any]] = None

		while True:
			search_results = got3.search_keyword_corpus(
				keyword=keyword,
				db_dict=self.db_dict_formatted,
				case_sensitive=False,
				show_context=True,
				context_words=context_words,
				output="json",
			)
			results_dict = {
				k: v for k, v in sorted(search_results.items(), key=lambda item: int(item[0])) if v
			}
			if not results_dict:
				return f"No results found for keyword '{keyword}'."
			total_cases = sum(len(cases) for cases in results_dict.values())
			volumes = list(results_dict.keys())
			prompt = self._build_prompt(results_dict, keyword, analysis_focus, volumes, total_cases)
			if len(prompt) <= max_tokens or context_words <= 5:
				final_prompt = prompt
				break
			attempts += 1
			ratio = max_tokens / max(len(prompt), 1)
			new_context_words = max(5, int(context_words * ratio * 0.9))
			if new_context_words >= context_words:
				final_prompt = prompt
				break
			context_words = new_context_words

		if not final_prompt:  # safety
			return "Failed to build prompt."
		response = self.model.invoke([{"role": "user", "content": final_prompt}])
		content = getattr(response, "content", str(response))
		return content

	def _build_prompt(
		self,
		results_dict: Dict[str, Any],
		keyword: str,
		analysis_focus: str,
		volumes: List[str],
		total_cases: int,
	) -> str:
		import json as _json
		analysis_prompts = {
			"general": f"""
Instructions:
You are an AI Agent inside the forensic linguistic tool `getout_of_text_3`.
Analyze these SCOTUS case search results for the keyword \"{keyword}\" ONLY using the provided data.
Data summary:
- Volumes: {', '.join(sorted(volumes, key=int))}
- Total case occurrences: {total_cases}
Provide insights on:
1. Temporal evolution
2. Contextual variation
3. Notable intra-dataset patterns (no outside knowledge)
4. Interpretive themes relevant to ordinary meaning
Results (truncated JSON): {_json.dumps(results_dict, indent=2)}...
""",
			"evolution": f"Focus on change over volumes for '{keyword}'.\nData: {_json.dumps(results_dict, indent=2)}...",
			"judicial_philosophy": f"Assess usage patterns hinting at differing interpretive approaches for '{keyword}'. Data: {_json.dumps(results_dict, indent=2)}...",
			"custom": f"Comprehensive analysis for '{keyword}'. Data: {_json.dumps(results_dict, indent=2)}...",
		}
		return analysis_prompts.get(analysis_focus, analysis_prompts["general"]).strip()


# ============================================================================
# Filtered SCOTUS JSON analysis tool
# ============================================================================

class ScotusFilteredAnalysisInput(BaseModel):
	"""Input schema for analyzing already-filtered SCOTUS keyword JSON."""

	keyword: str = Field(description="Label only; no searching performed")
	results_json: Union[str, Dict[str, Any]] = Field(
		description="Pre-filtered JSON/dict from got3.search_keyword_corpus (after user curation)."
	)
	analysis_focus: Optional[str] = Field(
		default="general", description="'general', 'evolution', 'judicial_philosophy', or 'custom'"
	)
	max_contexts: Optional[int] = Field(
		default=None, description="Optional cap on number of context snippets. If None/0 => ALL."
	)
	return_json: bool = Field(
		default=False, description="If True, attempt strict JSON output (validated / salvaged)."
	)
	extraction_strategy: str = Field(
		default="first",
		description="Context extraction: 'first' | 'all' | 'raw_json' (embed entire JSON).",
	)
	debug: bool = Field(
		default=False, description="If True, include debug metrics (chars, approx tokens, ratios)."
	)


class ScotusFilteredAnalysisTool(BaseTool):
	"""Analyze ONLY the supplied pre-filtered SCOTUS keyword result JSON.

	Features:
	- extraction_strategy: first/all/raw_json
	- debug metrics & transparency
	- token preflight rejection (approx char/4 heuristic)
	- salvage of malformed JSON when return_json=True
	"""

	name: str = "scotus_filtered_analysis"
	description: str = (
		"Analyzes pre-filtered SCOTUS keyword search JSON (from got3) without performing any new retrieval."
	)
	args_schema: Type[BaseModel] = ScotusFilteredAnalysisInput
	model: Any = Field(default=None, exclude=True)

	def __init__(self, model: Any, **kwargs):
		super().__init__(**kwargs)
		self.model = model

	# ---------------- Public entry points ----------------
	def _run(
		self,
		keyword: str,
		results_json: Union[str, Dict[str, Any]],
		analysis_focus: str = "general",
		max_contexts: Optional[int] = None,
		return_json: bool = False,
		extraction_strategy: str = "first",
		debug: bool = False,
	) -> Union[str, Dict[str, Any]]:  # noqa: D401
		try:
			return self._execute(
				keyword,
				results_json,
				analysis_focus,
				max_contexts,
				return_json,
				extraction_strategy,
				debug,
			)
		except Exception as e:  # pragma: no cover - defensive
			msg = f"Error (filtered analysis): {e}"
			return {"error": msg} if return_json else msg

	async def _arun(
		self,
		keyword: str,
		results_json: Union[str, Dict[str, Any]],
		analysis_focus: str = "general",
		max_contexts: Optional[int] = None,
		return_json: bool = False,
		extraction_strategy: str = "first",
		debug: bool = False,
	) -> Union[str, Dict[str, Any]]:  # noqa: D401
		return self._run(
			keyword,
			results_json,
			analysis_focus,
			max_contexts,
			return_json,
			extraction_strategy,
			debug,
		)

	# ---------------- Core logic ----------------
	def _execute(
		self,
		keyword: str,
		results_json: Union[str, Dict[str, Any]],
		analysis_focus: str,
		max_contexts: Optional[int],
		return_json: bool,
		extraction_strategy: str,
		debug: bool,
	) -> Union[str, Dict[str, Any]]:
		if extraction_strategy not in {"first", "all", "raw_json"}:
			raise ValueError("extraction_strategy must be one of: 'first','all','raw_json'")
		results_dict = self._coerce_results(results_json)
		stats = self._compute_stats(results_dict, keyword, extraction_strategy)

		# Model max tokens heuristic
		try:  # attempt to read from model internal kwargs if present
			max_tokens = self.model._lc_kwargs.get("max_tokens", 128000)  # type: ignore[attr-defined]
		except Exception:  # pragma: no cover
			max_tokens = 128000

		raw_json_str = json.dumps(results_dict, sort_keys=True)
		raw_chars = len(raw_json_str)
		if extraction_strategy == "raw_json":
			extracted_for_metrics = self._sample_contexts(results_dict, max_contexts, "all")
		else:
			extracted_for_metrics = self._sample_contexts(results_dict, max_contexts, extraction_strategy)
		extracted_chars = sum(len(c) for c in extracted_for_metrics)
		approx_tokens_raw = raw_chars / 4
		approx_tokens_extracted = extracted_chars / 4
		reduction_ratio = (extracted_chars / raw_chars) if raw_chars else 0
		if debug:
			print(
				f"[DEBUG] raw_chars={raw_chars} extracted_chars={extracted_chars} "
				f"reduction_ratio={reduction_ratio:.3f} raw≈{approx_tokens_raw:.0f}tok extracted≈{approx_tokens_extracted:.0f}tok "
				f"strategy={extraction_strategy} limit={max_contexts}"
			)

		prompt = self._build_prompt(
			keyword,
			results_dict,
			stats,
			analysis_focus,
			max_contexts,
			return_json,
			extraction_strategy,
			debug,
			{
				"raw_chars": raw_chars,
				"extracted_chars": extracted_chars,
				"approx_tokens_raw": approx_tokens_raw,
				"approx_tokens_extracted": approx_tokens_extracted,
				"reduction_ratio": reduction_ratio,
				"extraction_strategy": extraction_strategy,
			},
		)

		approx_prompt_tokens = len(prompt) / 4
		if approx_prompt_tokens > max_tokens:
			msg = (
				"Preflight rejection: prompt would exceed model max_tokens. "
				f"approx_prompt_tokens={approx_prompt_tokens:.0f} > max_tokens={max_tokens}. "
				f"Strategy='{extraction_strategy}' raw_tokens≈{approx_tokens_raw:.0f} extracted_tokens≈{approx_tokens_extracted:.0f}."
			)
			if return_json:
				return {
					"error": "prompt_too_large",
					"message": msg,
					"keyword": keyword,
					"total_contexts": stats["total_contexts"],
					"extraction_strategy": extraction_strategy,
					"raw_chars": raw_chars,
					"extracted_chars": extracted_chars,
					"approx_tokens_prompt": approx_prompt_tokens,
				}
			return msg

		response = self.model.invoke([{"role": "user", "content": prompt}])
		raw = getattr(response, "content", None)
		if raw is None and hasattr(response, "text"):
			try:  # pragma: no cover
				raw = response.text()
			except Exception:
				raw = response.text
		content = self._normalize_model_content(raw)
		if return_json:
			return self._postprocess_json(content, results_dict, stats)
		return content

	# ---------------- Helpers ----------------
	def _coerce_results(self, results_json: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
		if isinstance(results_json, str):
			data = json.loads(results_json)
		else:
			data = results_json
		if not isinstance(data, dict) or not data:
			raise ValueError("results_json must be a non-empty dict or JSON string")
		return data

	def _extract_contexts_from_case(self, occs, extraction_strategy: str) -> List[str]:
		contexts: List[str] = []
		if isinstance(occs, str):
			contexts.append(occs)
		elif isinstance(occs, dict):
			keys_to_check = (
				"context",
				"text",
				"snippet",
				"kwic",
				"content",
				"full_text",
				"body",
			)
			if extraction_strategy == "first":
				for k in keys_to_check:
					if k in occs and isinstance(occs[k], str):
						contexts.append(occs[k])
						break
			else:  # 'all' or metrics for raw_json
				for k in keys_to_check:
					v = occs.get(k)
					if isinstance(v, str):
						contexts.append(v)
		elif isinstance(occs, list):
			for o in occs:
				contexts.extend(self._extract_contexts_from_case(o, extraction_strategy))
		return contexts

	def _compute_stats(
		self, results_dict: Dict[str, Any], keyword: str, extraction_strategy: str
	) -> Dict[str, Any]:
		def _vol_key(x):
			return int(x) if str(x).isdigit() else str(x)

		volumes = sorted(results_dict.keys(), key=_vol_key)
		case_counts: Dict[str, int] = {}
		total_contexts = 0
		occurrences_per_case: List[Dict[str, Any]] = []
		for vol, cases in results_dict.items():
			if not isinstance(cases, dict):
				continue
			case_counts[vol] = len(cases)
			for case_id, occs in cases.items():
				contexts = self._extract_contexts_from_case(occs, extraction_strategy)
				occ_count = len(contexts)
				total_contexts += occ_count
				occurrences_per_case.append(
					{"volume": vol, "case_id": case_id, "occurrences": occ_count}
				)
		return {
			"volumes": volumes,
			"case_counts": case_counts,
			"total_cases": sum(case_counts.values()),
			"total_contexts": total_contexts,
			"occurrences_per_case": occurrences_per_case,
			"keyword": keyword,
		}

	def _sample_contexts(
		self,
		results_dict: Dict[str, Any],
		max_contexts: Optional[int],
		extraction_strategy: str,
	) -> List[str]:
		limit = max_contexts if isinstance(max_contexts, int) and max_contexts > 0 else None
		samples: List[str] = []
		if extraction_strategy == "raw_json":
			return samples
		def _vol_key(x):
			return int(x) if str(x).isdigit() else str(x)
		for vol in sorted(results_dict.keys(), key=_vol_key):
			cases = results_dict[vol]
			if not isinstance(cases, dict):
				continue
			for case_id, occs in cases.items():
				contexts = self._extract_contexts_from_case(occs, extraction_strategy)
				for ctx in contexts:
					cleaned = " ".join(ctx.split())
					samples.append(f"[{vol}:{case_id}] {cleaned}")
					if limit and len(samples) >= limit:
						return samples
		return samples

	def _build_prompt(
		self,
		keyword: str,
		results_dict: Dict[str, Any],
		stats: Dict[str, Any],
		analysis_focus: str,
		max_contexts: Optional[int],
		return_json: bool,
		extraction_strategy: str,
		debug: bool,
		metrics: Dict[str, Any],
	) -> str:
		if extraction_strategy == "raw_json":
			raw_block = json.dumps(results_dict, indent=2)
			contexts_section = (
				f"RAW_JSON_MODE: Entire filtered JSON below (chars={metrics['raw_chars']} approx_tokens={metrics['approx_tokens_raw']:.0f}).\n"  # noqa: E501
				"Use ONLY this data.\n---\n" + raw_block + "\n---\n"
			)
		else:
			sample_contexts = self._sample_contexts(results_dict, max_contexts, extraction_strategy)
			if not sample_contexts:
				sample_contexts = ["(No context strings extracted — verify input JSON structure)"]
			contexts_section = (
				f"Sample Contexts ({len(sample_contexts)}) strategy={extraction_strategy} (max_contexts={max_contexts}):\n---\n"
				+ "\n".join(sample_contexts)
				+ "\n---\n"
			)

		focus_map = {
			"general": "Overview of usage patterns, semantic ranges, interpretive variability.",
			"evolution": "Describe shifts across volumes (volume order as temporal proxy).",
			"judicial_philosophy": "Identify internal usage patterns hinting at interpretive strategies (ONLY data given).",
			"custom": "Comprehensive structured analysis (frequency, contextual clusters, senses).",
		}
		occ_lines = sorted(
			[f"{o['volume']}:{o['case_id']}={o['occurrences']}" for o in stats["occurrences_per_case"]]
		)[:80]
		debug_block = ""
		if debug:
			debug_block = (
				"DEBUG METRICS (do NOT just restate):\n"
				f"raw_chars={metrics['raw_chars']} extracted_chars={metrics['extracted_chars']} reduction_ratio={metrics['reduction_ratio']:.3f}\n"
				f"approx_tokens_raw={metrics['approx_tokens_raw']:.0f} approx_tokens_extracted={metrics['approx_tokens_extracted']:.0f} strategy={metrics['extraction_strategy']}\n"
			)
		base = f"""
You are an AI analysis component of `getout_of_text_3`.
STRICT RULE: Use ONLY the provided contexts / JSON. NO external cases, doctrines, or speculation.
Keyword: "{keyword}"
Volumes: {', '.join(stats['volumes'])}
Total Cases: {stats['total_cases']} | Total Context Snippets (strategy='{extraction_strategy}'): {stats['total_contexts']}
Occurrences Per Case (sample): {'; '.join(occ_lines)}
Analysis Focus: {analysis_focus} → {focus_map.get(analysis_focus, focus_map['general'])}
{debug_block}
{contexts_section}
"""
		if return_json:
			base += (
				"Return ONLY valid JSON with this exact top-level structure (no extra prose):\n"
				"{\n"
				"  \"keyword\": string,\n"
				"  \"total_contexts\": number,\n"
				"  \"occurrences_summary\": string,\n"
				"  \"reasoning_content\": [string, ...],\n"
				"  \"summary\": string,\n"
				"  \"limitations\": string\n"
				"}\n"
				"Populate reasoning_content with 3–6 concise steps.\n"
				"If only one occurrence, note insufficient data for variation."
			)
		else:
			base += (
				"Required Output Sections:\n1. Usage Summary\n2. Contextual Patterns / Proto-senses\n3. Frequency & Distribution Observations\n4. Interpretability Notes\n5. Open Questions / Ambiguities\nGround all claims ONLY in the contexts above."
			)
		if stats["total_contexts"] == 1:
			base += "\nNOTE: Only one occurrence detected."
		return base.strip()

	def _normalize_model_content(self, raw: Any) -> str:
		if isinstance(raw, str):
			return raw
		if isinstance(raw, list):
			parts: List[str] = []
			for block in raw:
				if isinstance(block, str):
					parts.append(block)
				elif isinstance(block, dict):
					for key in ("text", "content", "value", "message"):
						val = block.get(key)
						if isinstance(val, str):
							parts.append(val)
							break
					else:  # no break
						parts.append(str(block))
				else:
					parts.append(str(block))
			return "\n".join(parts)
		if isinstance(raw, dict):
			for key in ("text", "content", "value"):
				if key in raw and isinstance(raw[key], str):
					return raw[key]
			return json.dumps(raw)
		return str(raw)

	def _postprocess_json(
		self, content: str, results_dict: Dict[str, Any], stats: Dict[str, Any]
	) -> Dict[str, Any]:
		parsed: Optional[Dict[str, Any]] = None
		try:
			parsed = json.loads(content)
		except Exception:
			if isinstance(content, str):
				match = re.search(r'{[\s\S]*}', content)
				if match:
					try:
						parsed = json.loads(match.group(0))
					except Exception:
						parsed = None
		if not isinstance(parsed, dict):
			parsed = {
				"keyword": stats["keyword"],
				"total_contexts": stats["total_contexts"],
				"occurrences_summary": f"{stats['total_contexts']} context snippet(s) across {stats['total_cases']} case(s)",
				"reasoning_content": [
					"Model did not return valid JSON; wrapped raw text.",
					(
						"Single occurrence limits distributional inference."
						if stats["total_contexts"] == 1
						else "Multiple contexts allow limited comparative analysis."
					),
				],
				"summary": content[:4000] if isinstance(content, str) else str(content)[:4000],
				"limitations": "Auto-wrapped due to invalid JSON from model.",
			}
		for k, default in [
			("reasoning_content", []),
			("summary", ""),
			(
				"occurrences_summary",
				f"{stats['total_contexts']} snippet(s) across {stats['total_cases']} case(s)",
			),
			("limitations", ""),
		]:
			if k not in parsed:
				parsed[k] = default
		return parsed


__all__ = [
	"ScotusAnalysisTool",
	"ScotusFilteredAnalysisTool",
	"ScotusAnalysisInput",
	"ScotusFilteredAnalysisInput",
]
