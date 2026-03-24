"""LLM-based constraint analyzer for reaction optimization.

This module provides functionality to analyze previous reaction data
and generate constraints for optimization using LLM-based analysis.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import json
from openai import OpenAI


class LLMAnalyzer:
    """LLM-based analyzer for generating optimization constraints.

    This class analyzes previous reaction data and reaction space to determine
    which reagents/conditions should be constrained or eliminated in subsequent
    optimization rounds using LLM.
    """

    def __init__(self, opt_metrics: List, opt_metric_settings: List[Dict], prev_rxn: pd.DataFrame, condition_dict: Dict[str, List[Any]], existing_prohibited: Optional[Dict[str, List[Any]]] = None):
        """Initialize the LLMAnalyzer.

        Args:
            opt_metrics: List of optimization metric names
            opt_metric_settings: List of optimization metric settings
            prev_rxn: DataFrame containing previous reaction data
            condition_dict: Dictionary of condition types and their possible values
            existing_prohibited: Optional dictionary of already prohibited reagents
        """
        self.opt_metrics = opt_metrics
        self.opt_metric_settings = opt_metric_settings
        self.prev_rxn = prev_rxn
        self.condition_dict = condition_dict
        self.existing_prohibited = existing_prohibited

    def analyze(self, **kwargs) -> Optional[Dict[str, List[Any]]]:
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
            Dictionary of constraints in format {condition_type: [prohibited_values]}
            Returns None if no constraints are needed

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

        print(f"Analyzing {len(self.prev_rxn)} reactions using LLM: {model}")
        print(f"Target reduction ratio: {reduce_ratio}")

        # Generate constraints using LLM
        constraints = self._llm_analysis(
            reduce_ratio=reduce_ratio,
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

        if constraints:
            print(f"\nGenerated constraints:")
            for condition_type, prohibited_values in constraints.items():
                total_values = len(self.condition_dict[condition_type])
                removed = len(prohibited_values)
                removal_ratio = removed / total_values
                print(f"  {condition_type}: {len(prohibited_values)}/{total_values} values " f"({removal_ratio:.1%} removed)")
        else:
            print("\nNo constraints generated - keeping all values")

        return constraints

    def _llm_analysis(
        self, reduce_ratio: float, api_key: str, base_url: Optional[str] = None, model: str = "gpt-4", temperature: float = 0.7
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
        data_summary = self._prepare_data_summary()

        # Prepare reaction space summary
        space_summary = self._prepare_space_summary()

        # Create the prompt
        prompt = self._create_prompt(data_summary, space_summary, reduce_ratio)
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
                max_tokens=40000,
            )

            # Extract response content
            response_text = response.choices[0].message.content

            print(f"Received response from LLM")

            # Parse the response
            constraints = self._parse_llm_response(response_text)

            return constraints

        except Exception as e:
            raise Exception(f"Error calling LLM API: {e}")

    def _prepare_data_summary(self) -> str:
        """Prepare a summary of the reaction data for the LLM.

        Args:
            prev_rxn: DataFrame containing previous reaction data
            condition_dict: Dictionary of condition types and their possible values

        Returns:
            String summary of the data
        """
        summary = f"### Previous Reaction Data Summary:\n"

        opt_info = []
        for i in zip(self.opt_metrics, self.opt_metric_settings):
            opt_info.append(f"{i[0]}({i[1]['opt_direct']}imum)")

        summary += f"- Optimization metrics: {', '.join(opt_info)}\n"
        summary += f"- Condition types: {', '.join(self.condition_dict.keys())}\n"

        prev_rxn_df = self.prev_rxn[["batch", "index"] + list(self.condition_dict.keys()) + self.opt_metrics]
        prev_rxn_df.sort_values(by=["batch", "index"], inplace=True)
        summary += f"### Previous Reaction Optimization Results:"
        summary += prev_rxn_df.to_markdown(index=False)

        return summary

    def _prepare_space_summary(self) -> str:
        """Prepare a summary of reaction space for the LLM.

        Args:
            condition_dict: Dictionary of condition types and their possible values
            reduce_ratio: Ratio of reagents to eliminate

        Returns:
            String summary of the reaction space
        """
        summary = f"\n### Reaction Space Summary:\n"
        summary += f"- Condition types and their options:\n\n"

        for cond_type, values in self.condition_dict.items():
            summary += f"  - {cond_type} ({len(values)} options):\n"
            summary += f"  - All options: {', '.join(map(str, values))}"
            summary += "\n\n"

        # Add existing prohibited reagents information
        if self.existing_prohibited:
            summary += f"\n### Previously Prohibited Reagents:\n"
            summary += f"NOTE: The following reagents have already been eliminated in previous rounds. DO NOT recommend them again.\n\n"
            for cond_type, prohibited_values in self.existing_prohibited.items():
                if prohibited_values:
                    summary += f"  - {cond_type}: {', '.join(map(str, prohibited_values))}\n"
            summary += "\n"

        return summary

    def _create_prompt(self, data_summary: str, space_summary: str, reduce_ratio: float) -> str:
        """Create the prompt for the LLM.

        Args:
            data_summary: Summary of reaction data
            space_summary: Summary of reaction space
            condition_dict: Dictionary of condition types and their possible values
            reduce_ratio: Ratio of reagents to eliminate

        Returns:
            String prompt
        """

        reduce_reagents_num = int(sum(len(values) for values in self.condition_dict.values()) * reduce_ratio)

        prohibited_note = ""
        if self.existing_prohibited:
            prohibited_note = f"\nIMPORTANT: Some reagents have already been prohibited in previous rounds (see 'Previously Prohibited Reagents' section above). DO NOT include these in your response - only recommend NEW reagents to prohibit."

        prompt = f"""Based on the following reaction optimization data, please identify which reagents/conditions should be eliminated from the reaction space.

{data_summary}

{space_summary}{prohibited_note}

Your task:
1. For each condition type, analyze the data to identify reagents that are performing poorly or are unlikely to lead to optimal reactions.
2. Select approximately {reduce_ratio:.0%} of reagents to eliminate. That is, reduce totally {reduce_reagents_num} reagents from the reaction space.
3. Focus on eliminating the least promising reagents based on:
   - Poor performance in the existing data
   - Chemical incompatibility
 But DO NOT remove reagents that are not present in the data. They are likely to be tests for future rounds.

IMPORTANT - RIGHT TO REFUSE SPATIAL EXCLUSION:
You have the right to refuse performing spatial exclusion if either of the following conditions is met:
   - You determine that there are no conditions that need to be excluded based on the current data
   - Further exclusion would result in an empty candidate list for any condition type
In these cases, return an empty JSON object {{}} instead of attempting to eliminate reagents.

4. Return your answer in the following `JSON` format like:

{{
    "reagent1": ["reagent1_value1", "reagent1_value2"],
    "reagent2": ["reagent2_value1", "reagent2_value2"],
    ...
}}

Or if you decide to refuse spatial exclusion, return an empty JSON:

{{}}

The keys should be the condition types, and the values should be lists of reagents to ELIMINATE (NOT KEEP).

Please provide only the JSON response, no additional text."""

        return prompt

    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, List[Any]]]:
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
            constraints = json.loads(json_str)

            return constraints

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from LLM response: {e}")
            print(f"Response: {response_text}")
            return None
        except Exception as e:
            print(f"Error processing LLM response: {e}")
            return None
