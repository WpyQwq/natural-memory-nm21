| model | model_type | hidden | layers | container | 2nd positional | identity | surgery |
|---|---|---|---|---|---|---|---|
| tiny-random-LlamaForCausalLM | llama | 16 | 2 | model.layers | OTHER | - | FAIL ValueError: layer_indices must be inside [0, 2) |
| tiny-random-Qwen2ForCausalLM | - | - | - | - | OTHER | - | - |
| tiny-random-OlmoeForCausalLM | olmoe | 64 | 2 | model.layers | OTHER | - | FAIL ValueError: layer_indices must be inside [0, 2) |
