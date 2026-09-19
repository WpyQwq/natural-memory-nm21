| model | model_type | hidden | layers | container | 2nd positional | identity | surgery |
|---|---|---|---|---|---|---|---|
| tiny-random-LlamaForCausalLM | llama | 16 | 2 | model.layers | OTHER | identical | ok (no shims) |
| tiny-random-MistralForCausalLM | mistral | 32 | 2 | model.layers | OTHER | identical | ok (no shims) |
| tiny-Qwen2ForCausalLM-2.5 | qwen2 | 8 | 2 | model.layers | OTHER | identical | ok (no shims) |
| tiny-Qwen3ForCausalLM | qwen3 | 8 | 2 | model.layers | OTHER | identical | ok (no shims) |
| tiny-random-Gemma2ForCausalLM | gemma2 | 32 | 1 | model.layers | position_embeddings | identical | ok (no shims) |
| tiny-random-Gemma3ForCausalLM | gemma3_text | 16 | 2 | model.layers | position_embeddings | identical | ok (no shims) |
| tiny-random-Starcoder2ForCausalLM | starcoder2 | 32 | 2 | model.layers | OTHER | identical | ok (no shims) |
| tiny-random-OlmoeForCausalLM | olmoe | 64 | 2 | model.layers | OTHER | identical | ok (no shims) |
| tiny-random-GraniteMoeForCausalLM | granitemoe | 32 | 2 | model.layers | OTHER | identical | ok (no shims) |
| tiny-random-MixtralForCausalLM | mixtral | 64 | 2 | model.layers | position_embeddings | identical | ok (no shims) |
| tiny-random-CohereForCausalLM | cohere | 32 | 2 | model.layers | OTHER | identical | ok (no shims) |
| tiny-random-PhiForCausalLM | phi | 32 | 2 | model.layers | OTHER | identical | ok (no shims) |
| tiny-random-OPTForCausalLM | opt | 16 | 5 | model.decoder.layers | OTHER | identical | ok (no shims) |
| tiny-random-BartForCausalLM | bart | 16 | 2 | model.decoder.layers | OTHER | identical | ok (no shims) |
