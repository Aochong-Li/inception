import os
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

class PromptLoader:
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.template_dir = os.path.join(base_dir, "templates")
        self.example_dir = os.path.join(base_dir, "examples")
        
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape()
        )

    def _load_examples(self, filename):
        path = os.path.join(self.example_dir, filename)
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def render_biosecurity_prompt(self, reasoning_trace, user_query="", example_ids=None):
        """
        Renders the biosecurity evaluation prompt.
        :param reasoning_trace: The reasoning trace to evaluate.
        :param user_query: The original user query (optional, defaults to empty string).
        :param example_ids: Optional list of IDs to filter few-shot examples.
                            If None, all 10 examples are used.
        """
        examples = self._load_examples("bio_historical_cases.yaml")
        if example_ids:
            examples = [e for e in examples if e['id'] in example_ids]

        template = self.env.get_template("biosecurity_judge.xml.j2")
        return template.render(
            reasoning_trace=reasoning_trace,
            user_query=user_query,
            examples=examples
        )

    def render_chemical_prompt(self, reasoning_trace, user_query="", example_ids=None):
        """
        Renders the chemical warfare evaluation prompt.
        :param reasoning_trace: The reasoning trace to evaluate.
        :param user_query: The original user query (optional, defaults to empty string).
        :param example_ids: Optional list of IDs to filter few-shot examples.
                            If None, all 10 examples are used.
        """
        examples = self._load_examples("chem_historical_cases.yaml")
        if example_ids:
            examples = [e for e in examples if e['id'] in example_ids]

        template = self.env.get_template("chemical_judge.xml.j2")
        return template.render(
            reasoning_trace=reasoning_trace,
            user_query=user_query,
            examples=examples
        )

# Example usage:
# loader = PromptLoader()
# prompt = loader.render_biosecurity_prompt(reasoning_trace="...", example_ids=[1, 4, 8])
