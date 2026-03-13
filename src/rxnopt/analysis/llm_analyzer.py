"""LLM-based constraint analyzer for reaction optimization.

This module provides functionality to analyze previous reaction data
and generate constraints for optimization using LLM-based analysis.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import json
from openai import OpenAI


class LLMAnalyzer:
    """LLM-based analyzer for generating optimization constraints.

    This class analyzes previous reaction data and reaction space to determine
    which reagents/conditions should be constrained or eliminated in subsequent
    optimization rounds using LLM.
    """

    def __init__(self):
        """Initialize the LLMAnalyzer."""
        self.analysis_results = {}

    def analyze(self, prev_rxn: pd.DataFrame, condition_dict: Dict[str, List[Any]], **kwargs) -> Optional[Dict[str, List[Any]]]:
        """Analyze previous reactions and generate constraints using LLM.

        This method uses LLM to analyze the previous reaction data and determine
        which reagents/conditions should be kept or eliminated.

        Args:
            prev_rxn: DataFrame containing previous reaction data
            condition_dict: Dictionary of condition types and their possible values
            **kwargs: Additional parameters:
                - reduce_ratio: Ratio of reagents to eliminate (default: 0.3)
                - api_key: OpenAI API key (required)
                - base_url: OpenAI API base URL (optional, default: OpenAI's default)
                - model: LLM model to use (default: "gpt-4")
                - temperature: Temperature for LLM generation (default: 0.7)

        Returns:
            Dictionary of constraints in format {condition_type: [allowed_values]}
            Returns None if no constraints are needed

        Example:
            >>> analyzer = LLMAnalyzer()
            >>> constraints = analyzer.analyze(
            ...     prev_rxn=prev_data,
            ...     condition_dict=condition_dict,
            ...     reduce_ratio=0.3,
            ...     api_key="your-api-key",
            ...     model="gpt-4"
            ... )
        """
        # Extract parameters
        reduce_ratio = kwargs.get("reduce_ratio", 0.1)
        api_key = kwargs.get("api_key", None)
        base_url = kwargs.get("base_url", None)
        model = kwargs.get("model", "gpt-4")
        temperature = kwargs.get("temperature", 0.0)

        # Validate required parameters
        if api_key is None:
            raise ValueError("api_key is required for LLM analysis")

        print(f"Analyzing {len(prev_rxn)} reactions using LLM: {model}")
        print(f"Target reduction ratio: {reduce_ratio}")

        # Generate constraints using LLM
        constraints = self._llm_analysis(
            prev_rxn=prev_rxn,
            condition_dict=condition_dict,
            reduce_ratio=reduce_ratio,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

        if constraints:
            print(f"\nGenerated constraints:")
            for condition_type, allowed_values in constraints.items():
                total_values = len(condition_dict[condition_type])
                removed = total_values - len(allowed_values)
                removal_ratio = removed / total_values
                print(f"  {condition_type}: {len(allowed_values)}/{total_values} values " f"({removal_ratio:.1%} removed)")
        else:
            print("\nNo constraints generated - keeping all values")

        return constraints

    def _llm_analysis(
        self,
        prev_rxn: pd.DataFrame,
        condition_dict: Dict[str, List[Any]],
        reduce_ratio: float,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
    ) -> Optional[Dict[str, List[Any]]]:
        """Perform LLM-based analysis for constraint generation.

        This method:
        1. Prepares a prompt describing the reaction data and optimization goal
        2. Sends to an LLM (via OpenAI API)
        3. Parses the LLM's response to extract constraint recommendations
        4. Validates and formats the constraints

        Args:
            prev_rxn: DataFrame containing previous reaction data
            condition_dict: Dictionary of condition types and their possible values
            reduce_ratio: Ratio of reagents to eliminate
            api_key: OpenAI API key
            base_url: Optional custom base URL for OpenAI API
            model: LLM model name
            temperature: Temperature for generation

        Returns:
            Dictionary of constraints or None
        """
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary(prev_rxn, condition_dict)

        # Prepare reaction space summary
        space_summary = self._prepare_space_summary(condition_dict, reduce_ratio)

        # Create the prompt
        prompt = self._create_prompt(data_summary, space_summary, condition_dict, reduce_ratio)
        print(prompt)
        # Call LLM API
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)

            print(f"Sending request to LLM API...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert chemist specializing in reaction optimization. You analyze experimental data to identify which reagents and conditions are performing poorly and should be eliminated from future optimization rounds.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=10000,
            )

            # Extract response content
            response_text = response.choices[0].message.content
            from IPython import embed

            embed()
            print(f"Received response from LLM")

            # Parse the response
            constraints = self._parse_llm_response(response_text, condition_dict)

            return constraints

        except Exception as e:
            print(f"Error calling LLM API: {e}")
            return None

    def _prepare_data_summary(self, prev_rxn: pd.DataFrame, condition_dict: Dict[str, List[Any]]) -> str:
        """Prepare a summary of the reaction data for the LLM.

        Args:
            prev_rxn: DataFrame containing previous reaction data
            condition_dict: Dictionary of condition types and their possible values

        Returns:
            String summary of the data
        """
        # Find metric columns (columns that are not condition types)
        metric_columns = [col for col in prev_rxn.columns if col not in condition_dict and col != "batch"]

        summary = f"Previous Reaction Data Summary:\n"
        summary += f"- Total reactions: {len(prev_rxn)}\n"
        summary += f"- Condition types: {', '.join(condition_dict.keys())}\n"

        if metric_columns:
            summary += f"- Optimization metrics: {', '.join(metric_columns)}\n"  # TODO: this one is wrong.

        # Show top 5 reactions by first metric
        if metric_columns and pd.api.types.is_numeric_dtype(prev_rxn[metric_columns[0]]):
            top_reactions = prev_rxn.nlargest(5, metric_columns[0])
            summary += "\nTop 5 performing reactions:\n"
            for idx, row in top_reactions.iterrows():
                summary += f"  - {metric_columns[0]}: {row[metric_columns[0]]:.2f}, "
                for cond in list(condition_dict.keys())[:3]:  # Show first 3 conditions
                    summary += f"{cond}={row[cond]}, "
                summary = summary.rstrip(", ") + "\n"

        return summary

    def _prepare_space_summary(self, condition_dict: Dict[str, List[Any]], reduce_ratio: float) -> str:
        """Prepare a summary of the reaction space for the LLM.

        Args:
            condition_dict: Dictionary of condition types and their possible values
            reduce_ratio: Ratio of reagents to eliminate

        Returns:
            String summary of the reaction space
        """
        summary = f"\nReaction Space Summary:\n"
        summary += f"- Goal: Eliminate approximately {reduce_ratio*100:.0%} of reagents per condition type\n"
        summary += f"- Condition types and their options:\n\n"

        for cond_type, values in condition_dict.items():
            num_to_keep = int(len(values) * (1 - reduce_ratio))
            summary += f"{cond_type}:\n"
            summary += f"  - Total options: {len(values)}\n"
            summary += f"  - Target to keep: ~{num_to_keep}\n"
            summary += f"  - All options: {', '.join(map(str, values[:10]))}"
            if len(values) > 10:
                summary += f" ... and {len(values)-10} more"
            summary += "\n\n"

        return summary

    def _create_prompt(self, data_summary: str, space_summary: str, condition_dict: Dict[str, List[Any]], reduce_ratio: float) -> str:
        """Create the prompt for the LLM.

        Args:
            data_summary: Summary of reaction data
            space_summary: Summary of reaction space
            condition_dict: Dictionary of condition types and their possible values
            reduce_ratio: Ratio of reagents to eliminate

        Returns:
            String prompt
        """
        prompt = f"""Based on the following reaction optimization data, please identify which reagents/conditions should be eliminated from the reaction space.

{data_summary}

{space_summary}

Your task:
1. For each condition type, analyze the data to identify reagents that are performing poorly or are unlikely to lead to optimal reactions
2. Select approximately {reduce_ratio*100:.0%} of reagents to eliminate from each condition type
3. Focus on eliminating the least promising reagents based on:
   - Poor performance in the existing data
   - Chemical incompatibility
   - Unlikely combinations
4. Return your answer in the following JSON format:

{{
    "base": ["base_value_1", "base_value_2"],
    "ligand": ["ligand_value_1", "ligand_value_2"],
    "solvent": ["solvent_value_1", "solvent_value_2"]
}}

The keys should be the condition types, and the values should be lists of reagents to KEEP (not eliminate).

Please provide only the JSON response, no additional text."""

        return prompt

    def _parse_llm_response(self, response_text: str, condition_dict: Dict[str, List[Any]]) -> Optional[Dict[str, List[Any]]]:
        """Parse the LLM response to extract constraints.

        Args:
            response_text: Text response from LLM
            condition_dict: Dictionary of condition types and their possible values

        Returns:
            Dictionary of constraints or None
        """
        try:
            # Try to extract JSON from response
            # Find JSON content between { and }
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx == -1 or end_idx == 0:
                print("Warning: Could not find JSON in LLM response")
                print(f"Response: {response_text}")
                return None

            json_str = response_text[start_idx:end_idx]
            constraints_data = json.loads(json_str)

            # Validate and format constraints
            constraints = {}
            for cond_type in condition_dict.keys():
                if cond_type in constraints_data:
                    keep_values = constraints_data[cond_type]

                    # Validate that the values exist in condition_dict
                    valid_values = []
                    for value in keep_values:
                        if value in condition_dict.get(cond_type, []):
                            valid_values.append(value)

                    # Only add constraint if we're removing some values
                    if valid_values and len(valid_values) < len(condition_dict[cond_type]):
                        constraints[cond_type] = valid_values
                        print(f"  {cond_type}: {len(valid_values)}/{len(condition_dict[cond_type])} values kept")

            if not constraints:
                print("Warning: No valid constraints generated from LLM response")
                return None

            return constraints

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from LLM response: {e}")
            print(f"Response: {response_text}")
            return None
        except Exception as e:
            print(f"Error processing LLM response: {e}")
            return None

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the last analysis performed.

        Returns:
            Dictionary containing analysis statistics and results
        """
        return self.analysis_results
