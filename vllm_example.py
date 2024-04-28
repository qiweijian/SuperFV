from vllm import LLM, SamplingParams

def greedy_inference(model: LLM, prompts: List[str], greedy_params: SamplingParams = SamplingParams(**GREEDY_PARAMS)) -> List[str, List[int]]]:
    assert greedy_params.n == 1 and greedy_params.best_of == 1
    requests = model.generate(prompts, sampling_params=greedy_params)
    greedy_outputs = [
        (one_request.outputs[0].text, one_request.outputs[0].token_ids)
        for one_request in requests
    ]
    return greedy_outputs
    

GREEDY_PARAMS = {
    "temperature": 0.0, "top_p": 1.0, "max_tokens": 100, "n": 1,
    "stop": ["\n\n"]
}
model = LLM(
    model=model_name_or_path, tensor_parallel_size=tensor_parallel_size,
    gpu_memory_utilization=0.85, max_model_len=2048 # avoid OOM
)
greedy_params = SamplingParams(**GREEDY_PARAMS)

