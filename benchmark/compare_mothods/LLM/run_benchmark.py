"""
LLM Benchmark for Reaction Optimization

This script uses an LLM API to perform reaction optimization through iterative experiments.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Optional
import requests
from io import StringIO
import time


class LLMBenchmark:
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://aihubmix.com/v1/chat/completions",
        model: str = "gemini-3-flash-preview",
        dataset_path: str = "benchmark/datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv",
        prompt_path: str = "benchmark/compare_mothods/LLM/prompt.md",
        output_dir: str = "benchmark/compare_mothods/LLM/results",
        batch_size: int = 5,
        max_rounds: int = 10,
    ):
        """
        Initialize the LLM benchmark.

        Args:
            api_key: API key for the LLM service
            api_url: URL for the LLM API endpoint
            model: Model name to use
            dataset_path: Path to the reference dataset
            prompt_path: Path to the prompt template
            output_dir: Directory to save results
            batch_size: Number of experiments per batch
            max_rounds: Maximum number of conversation rounds
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.dataset_path = dataset_path
        self.prompt_path = prompt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_rounds = max_rounds

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load reference dataset
        self.reference_df = pd.read_csv(dataset_path)

        # Load system prompt
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        # Configuration for optimization
        self.config = {
            "batch_size": batch_size,
            "opt_metrics": ["yield", "cost"],
            "opt_metric_settings": [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}],
            "condition_dict": [
                {"condition_type": "base", "condition_candidates": ["CsOAc", "CsOPiv", "KOAc", "KOPiv"]},
                {
                    "condition_type": "ligand",
                    "condition_candidates": [
                        "BrettPhos",
                        "CgMe-PPh",
                        "GorlosPhos HBF4",
                        "JackiePhos",
                        "PCy3 HBF4",
                        "P(fur)3",
                        "PPh2Me",
                        "PPh3",
                        "PPhMe2",
                        "PPhtBu2",
                        "tBPh-CPhos",
                        "X-Phos",
                    ],
                },
                {"condition_type": "solvent", "condition_candidates": ["BuCN", "BuOAc", "DMAc", "p-Xylene"]},
                {"condition_type": "concentration", "condition_candidates": [0.1, 0.057, 0.153]},
                {"condition_type": "temperature", "condition_candidates": [90, 105, 120]},
            ],
        }

        # Store all results
        self.all_results = None
        self.conversation_history = []

    def call_llm_api(self, messages: List[Dict[str, str]], max_retries: int = 3, timeout: int = 300) -> str:
        """
        Call LLM API with given messages and retry mechanism.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Request timeout in seconds (default: 300)

        Returns:
            The assistant's response text
        """
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

        payload = {"model": self.model, "messages": messages, "temperature": 0.7, "max_tokens": 10000}

        for attempt in range(max_retries):
            try:
                print(f"  Attempt {attempt + 1}/{max_retries} (timeout: {timeout}s)...")
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except requests.exceptions.Timeout as e:
                print(f"  Timeout error: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API call failed after {max_retries} attempts: {e}")
            except requests.exceptions.RequestException as e:
                print(f"  Request error: {e}")
                # Try to get more details from response
                if hasattr(e, "response") and e.response is not None:
                    try:
                        print(f"  Response status: {e.response.status_code}")
                        print(f"  Response content: {e.response.text[:500]}")
                    except:
                        pass
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API call failed after {max_retries} attempts: {e}")
            except Exception as e:
                print(f"  Unexpected error: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"  Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API call failed after {max_retries} attempts: {e}")

    def lookup_yield_cost(self, row: pd.Series) -> tuple:
        """
        Look up yield and cost from the reference dataset based on condition combination.

        Args:
            row: A row from the experiments DataFrame

        Returns:
            Tuple of (yield, cost) values from the reference dataset
        """
        # Create query conditions
        query = (
            (self.reference_df["base"] == row["base"])
            & (self.reference_df["ligand"] == row["ligand"])
            & (self.reference_df["solvent"] == row["solvent"])
            & (self.reference_df["concentration"] == row["concentration"])
            & (self.reference_df["temperature"] == row["temperature"])
        )

        matches = self.reference_df[query]

        if len(matches) > 0:
            # Return the first match
            return matches.iloc[0]["yield"], matches.iloc[0]["cost"]
        else:
            # No match found, return None values
            return None, None

    def parse_llm_response(self, response: str) -> pd.DataFrame:
        """
        Parse the LLM's CSV response into a DataFrame.

        Args:
            response: The text response from the LLM

        Returns:
            DataFrame containing the parsed experiments
        """
        # Extract CSV content from response
        lines = response.strip().split("\n")
        csv_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith("```") and "," in line:
                csv_lines.append(line)

        if not csv_lines:
            raise ValueError("No CSV content found in LLM response")

        # Parse CSV
        csv_content = "\n".join(csv_lines)
        df = pd.read_csv(StringIO(csv_content), index_col=None)
        # from IPython import embed; embed()
        return df

    def fill_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill yield and cost columns using the reference dataset.

        Args:
            df: DataFrame with experiment conditions

        Returns:
            DataFrame with filled yield and cost columns
        """
        yields = []
        costs = []

        for _, row in df.iterrows():
            yield_val, cost_val = self.lookup_yield_cost(row)
            yields.append(yield_val)
            costs.append(cost_val)

        df["yield"] = yields
        df["cost"] = costs

        return df

    def construct_user_message(self, round_num: int) -> str:
        """
        Construct the user message for the current round.

        Args:
            round_num: Current round number

        Returns:
            The user message as a string
        """
        if round_num == 0 or self.all_results is None:
            # First round - send configuration only
            message = f"For this optimization:\n\n{json.dumps(self.config, indent=4)}\n"
        else:
            # Subsequent rounds - send configuration and recent results only
            # Only send the most recent 2 batches to avoid exceeding API limits
            # if len(self.all_results) > 2 * self.batch_size:
            #     recent_results = self.all_results.iloc[-2 * self.batch_size :]
            # else:
            recent_results = self.all_results

            csv_str = recent_results.to_csv(index=False)
            message = f"For this optimization:\n\n{json.dumps(self.config, indent=4)}\n\n"
            message += f"Previous optimization results (most recent batches):\n\n{csv_str}\n"

        return message

    def run_benchmark(self) -> pd.DataFrame:
        """
        Run the complete benchmark for the specified number of rounds.

        Returns:
            DataFrame containing all results from all rounds
        """
        print(f"Starting LLM benchmark with {self.max_rounds} rounds...")
        print(f"Batch size: {self.batch_size}")
        print(f"Model: {self.model}")
        print("-" * 80)

        for round_num in range(self.max_rounds):
            print(f"\n=== Round {round_num + 1}/{self.max_rounds} ===")

            # Construct user message
            user_message = self.construct_user_message(round_num)

            # Build messages list
            # Only send system prompt and current user message
            # Do not add conversation history to avoid "assistant message prefill" error
            # Previous results are already included in the user message
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user_message}]

            # Call LLM API
            print(f"Calling LLM API...")
            assistant_response = self.call_llm_api(messages)

            print("--------response-------")
            print(assistant_response)
            print("----end response-------")

            # Store conversation
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})

            # Parse response
            print(f"Parsing LLM response...")
            try:
                new_experiments = self.parse_llm_response(assistant_response)
                print(f"Received {len(new_experiments)} experiment suggestions")
            except Exception as e:
                print(f"Error parsing response: {e}")
                print(f"Response content: {assistant_response}")
                continue

            # Fill metrics from reference dataset
            print(f"Looking up yield and cost values...")

            new_experiments = self.fill_metrics(new_experiments)

            # Check for missing values
            missing_count = new_experiments["yield"].isna().sum()
            if missing_count > 0:
                print(f"Warning: {missing_count} experiments had no match in reference dataset")

            # Concatenate with existing results
            if self.all_results is None:
                self.all_results = new_experiments
            else:
                self.all_results = pd.concat([self.all_results, new_experiments], ignore_index=True)

            print(f"Total experiments so far: {len(self.all_results)}")
            print(f"Batch statistics:")
            print(f"  - Mean yield: {new_experiments['yield'].mean():.2f}")
            print(f"  - Max yield: {new_experiments['yield'].max():.2f}")
            print(f"  - Mean cost: {new_experiments['cost'].mean():.4f}")

            # Save intermediate results
            intermediate_file = os.path.join(self.output_dir, f"round_{round_num + 1}_results.csv")
            self.all_results.to_csv(intermediate_file, index=False)
            print(f"Saved intermediate results to: {intermediate_file}")

        # Save final results
        final_file = os.path.join(self.output_dir, "final_results.csv")
        self.all_results.to_csv(final_file, index=False)
        print(f"\n{'=' * 80}")
        print(f"Benchmark complete!")
        print(f"Total experiments: {len(self.all_results)}")
        print(f"Final results saved to: {final_file}")

        return self.all_results


def main():
    """
    Main function to run the benchmark.
    """
    # Configuration - Update these with your actual API credentials
    API_KEY = os.environ.get("OPENAI_API_KEY", "sk-Pnmf5IgIJYMBEY8Z7078E31cAbC8437e83B4DdE3CaA72e78")
    API_URL = os.environ.get("OPENAI_API_URL", "https://aihubmix.com/v1/chat/completions")
    MODEL = os.environ.get("OPENAI_MODEL", "gemini-3-pro-preview")

    # You can also use other LLM providers by changing the API_URL and format
    # For example, for a local LLaMA server:
    # API_URL = "http://localhost:8000/v1/chat/completions"
    # MODEL = "llama-2-7b"

    # Create and run benchmark
    benchmark = LLMBenchmark(
        api_key=API_KEY,
        api_url=API_URL,
        model=MODEL,
        dataset_path="../../datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv",
        prompt_path="prompt.md",
        output_dir="results",
        batch_size=5,
        max_rounds=10,
    )

    # Run the benchmark
    results = benchmark.run_benchmark()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total experiments: {len(results)}")
    print(
        f"Unique condition combinations tested: {results[['base', 'ligand', 'solvent', 'concentration', 'temperature']].drop_duplicates().shape[0]}"
    )
    print(f"\nOverall statistics:")
    print(f"  - Mean yield: {results['yield'].mean():.2f}")
    print(f"  - Max yield: {results['yield'].max():.2f}")
    print(f"  - Min yield: {results['yield'].min():.2f}")
    print(f"  - Mean cost: {results['cost'].mean():.4f}")
    print(f"  - Min cost: {results['cost'].min():.4f}")

    # Show top 5 best yields
    print(f"\nTop 5 best yields:")
    top_5 = results.nlargest(5, "yield")[["batch", "index", "base", "ligand", "solvent", "concentration", "temperature", "yield", "cost"]]
    print(top_5.to_string(index=False))


if __name__ == "__main__":
    main()
