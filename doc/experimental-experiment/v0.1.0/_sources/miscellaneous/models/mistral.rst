=======
Mistral
=======

`Mistral <https://huggingface.co/docs/transformers/en/model_doc/mistral>`_

.. runpython::
    :showcode:

    import numpy as np
    import torch
    from transformers import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralModel
    from experimental_experiment.helpers import pretty_onnx
    from experimental_experiment.torch_interpreter import to_onnx, ExportOptions


    def ids_tensor(shape, vocab_size):
        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(np.random.randint(0, vocab_size - 1))

        return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


    config = MistralConfig(
        hidden_size=32,
        num_hidden_layers=2,
        vocab_size=1024,
        intermediate_size=16,
        max_position_embeddings=512,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    config._attn_implementation = "eager"

    with torch.no_grad():
    
        model = MistralModel(config)

        batch, seq, vocab_size = 2, 1024, 1024

        input_ids = ids_tensor([batch, seq], vocab_size)
        input_mask = torch.tril(torch.ones(batch, seq, dtype=torch.float32))

        model(input_ids, input_mask)

        try:
            onx = to_onnx(
                model,
                (input_ids, input_mask),
                export_options=ExportOptions(decomposition_table="default"),
            )
            print(pretty_onnx(onx))
        except Exception as e:
            print(f"conversion is broken due to {e}")
