from typing import Any, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

_MODEL_CACHE : dict[tuple[str, str], tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

def _get_model_and_tokenizer(model_name: str, device: str = "cpu") -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if (model_name, device) not in _MODEL_CACHE:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        _MODEL_CACHE[(model_name, device)] = (model, tokenizer)
    return _MODEL_CACHE[(model_name, device)]


class LLMWrapper:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model, self.tokenizer = _get_model_and_tokenizer(model_name, device)
        self.device = device

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        message_history: list[dict] | None = None,
        max_new_tokens: int = 128) -> str:

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if message_history:
            messages.extend(message_history)
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        all_generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
        all_output = self.tokenizer.batch_decode(all_generated_ids, skip_special_tokens=True)[0]

        answer_generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, all_generated_ids)
        ]
        answer = self.tokenizer.decode(answer_generated_ids, skip_special_tokens=True)[0]

        return answer, {"all_output": all_output}


class LLMTool:
    def __init__(
        self,
        model_name: str,
        description: str,
        arguments: dict[str, str],        
        output: str,
        objective: str = "",
        additional_instructions: str = "",
        argument_encoders: dict[str, Callable[[Any], str]] | None = None,
        device: str = "cpu"):
        self._model = LLMWrapper(model_name, device)
        self._argument_encoders = argument_encoders or {}
        self._argument_keys = set(arguments.keys())

        if additional_instructions:
            additional_instructions = f"### ADDITIONAL INSTRUCTIONS\n\n{additional_instructions}"

        if objective:
            objective = f"### OBJECTIVE\n\n{objective}"

        arguments = "\n\n".join([
            f"##### {key}\n\n{value}"
            for key, value in arguments.items()
        ])

        self._system_prompt = f"""
        {description}

        ### ARGUMENTS
        
        You are provided with the following arguments in markdown format:
        {arguments}

        {objective}

        ### OUTPUT
        
        You must return the following output. You **MUST** return the output in the exact format specified.
        {output}

        {additional_instructions}
        """

    def generate(self, arguments: dict[str, Any], max_new_tokens: int = 128) -> tuple[str, dict[str, Any]]:

        for k in self._argument_keys:
            if k not in arguments:
                raise ValueError(f"Argument {k} is required but not provided.")

        encoded_arguments = {
            key: self._argument_encoders[key](value)
            if key in self._argument_encoders else str(value)
            for key, value in arguments.items()
        }
        
        arguments_md = "\n\n".join([
            f"#### {key}\n\n{value}"
            for key, value in encoded_arguments.items()
        ])

        prompt = f"""
        Solve the task using these arguments and return the output in the exact format specified.

        {arguments_md}
        """

        answer, llm_additional_info = self._model.generate(
            prompt=prompt,
            system_prompt=self._system_prompt,            
            max_new_tokens=max_new_tokens
        )

        return answer, llm_additional_info


class AnswerShortener(LLMTool):
    def __init__(self, model_name: str, device: str = "cpu"):
        description = """
        You are an answer shortener tool.

        Given a question and answer, shorten the answer to be
        as concise as possible while still answering the question.        
        """

        arguments = {
            "question": "The question to answer.",
            "original_answer": "The answer to shorten.",
        }

        output = "The shortened answer. Plain text."

        additional_instructions = """
        - The answer does *NOT* need to be a complete sentences.
        - The answer should not be longer than 4 words.
        - The answer *MUST* contain exactly the information required to answer the question.

        ### EXAMPLES
        question: What is the capital of France?
        original_answer: The capital of France is Paris.
        output: Paris

        question: Who wrote the book "1984"?
        original_answer: George Orwell wrote the book "1984".
        output: George Orwell
        """

        super().__init__(
            model_name,
            device=device,
            description=description,
            arguments=arguments,
            output=output,
            additional_instructions=additional_instructions,
        )


maybe_empty_string = lambda s: s if s else "<empty>"

class AnswerSynthesizer(LLMTool):
    def __init__(self, model_name: str, device: str = "cpu"):
        description = """
        You are a question answering tool.

        Given a question and data query results, synthesize an answer to the question.
        """
        
        arguments = {
            "context": "The context of the question. Can be <empty> if no context is provided.",
            "query_outputs":
                "A list of JSON objects, each containing a `query` and `result` field.\n"
                " The `query` field contains the query that was executed.\n"
                " The `result` field contains the result of the query as a document snippet.\n",
            "question": "The question to answer.",
        }

        objective = """
        Synthesize an answer to the question based on the provided query outputs.
        """
        
        output = "The synthesized answer. Plain text."
        
        additional_instructions = """
        - The answer should be concise and to the point.
        - The answer should be no more than 5 words.
        - The answer *MUST* contain exactly the information required to answer the question, and no more.

        ### EXAMPLES
        context: Helium and Hydrogen are two elements in the periodic table.
        question: Does Helium have higher atomic mass than Hydrogen?
        query_outputs: [
            {
                "query": "What is the atomic mass of Helium?",
                "result": "2.0141 u",
            },
            {
                "query": "What is the atomic mass of Hydrogen?",
                "result": "1.0079 u",
            },
        ]
        output: Yes
        """
        
        super().__init__(
            model_name,
            device=device,
            description=description,
            arguments=arguments,
            output=output,
            objective=objective,
            additional_instructions=additional_instructions,
            argument_encoders={
                "context": maybe_empty_string,
            },
        )


class QueryGenerator(LLMTool):
    def __init__(self, model_name: str, device: str = "cpu"):
        description = """
        You are a query planner tool.
        Given a question, and previous queries, generate a list of queries to answer the question.
        """
        
        arguments = {
            "context": "The context of the question. Can be <empty> if no context is provided.",
            "question": "The question to answer.",
            "previous_query_outputs":
                "A list of JSON objects, each containing a `query` and `result` field.\n"
                " The `query` field contains the query that was executed.\n"
                " The `result` field contains the result of the query as a document snippet.\n",
        }

        objective = """
        Generate a list of useful, non-redundant queries that help obtain missing information
        needed to answer the question.
        """

        output = """
        A JSON list of queries to answer the question separated by newlines, or final answer.
        If the previous query outputs already contain enough information to answer the question:
        return exactly:
        ```
        ANSWER: <final_answer>
        ```
        """

        additional_instructions = """
        **RELEVANCE**: Each query must directly contribute to answering the question.
        **NON-REDUNDANCY**: Do NOT repeat or paraphrase any previous query.
        **NEW-INFORMATION**: Do NOT request information already clearly present in previous results.
        **GAP-FOCUSED**: Identify missing pieces of information and generate queries specifically to fill those gaps.
        **DIVERSE**: Prefer queries that explore different aspects or angles of the problem.
        **MINIMUM-OUTPUT**: Output at least one query unless the answer is already fully known.
        
        If the previous query outputs already contain enough information to answer the question:
        return exactly:
        ```
        ANSWER: <final_answer>
        ```

        ### EXAMPLES
        
        #### Example 1
        Input
        ```
        question: Does Helium have higher atomic mass than Hydrogen?
        context: Helium and Hydrogen are two elements in the periodic table.
        previous_query_outputs: []
        ```

        Output:
        [
            "What is the atomic mass of Helium?",
            "What is the atomic mass of Hydrogen?",
        ]

        #### Example 2
        Input
        ```
        question: What is the capital of France?
        context: France is a country in Europe.
        previous_query_outputs:
        [
            {
                "query": "What is the capital of France?",
                "result": "The capital of France is Paris.",
            },
        ]
        ```

        Output:        
        ["ANSWER: Paris"]
        """
        
        super().__init__(
            model_name,
            device=device,
            description=description,
            arguments=arguments,
            output=output,
            argument_encoders={
                "context": maybe_empty_string,
            },
            objective=objective,
            additional_instructions=additional_instructions,
        )



def parse_string_list(text: str) -> list[str]:    
    text = text.replace("```json", "").replace("```", "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item.strip() for item in parsed]
        elif isinstance(parsed, dict):
            return [item.strip() for item in list(parsed.values())]
        else:
            return [str(parsed)]
    except:
        return [text]