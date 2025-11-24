# PT_HPU_LAZY_MODE=1

import argparse
import torch
import habana_frameworks.torch.core as htcore
# htcore.hpu_set_env()
# !!!!!!!!!!!!!!!!!!! fix mark scales as const
htcore.hpu_inference_set_env()

from neural_compressor.torch.quantization import FP8Config, convert, finalize_calibration, prepare

torch.manual_seed(1)

# 1. python sample_two_steps.py --quant_config=maxabs_measure.json
# 2. python sample_two_steps.py --quant_config=maxabs_quant.json




class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 10)
        # self.register_buffer("scale_input", torch.ones(1))
        # self.input_scale = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        init_tensor = torch.tensor(1.0, dtype=torch.float32)
        self.input_scale = torch.nn.Parameter(init_tensor, requires_grad=False)

    def forward(self, inp):
        x1 = self.fc1(inp)
        x2 = self.fc2(x1)
        return x2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Habana FP8 sample code.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--quant_config", type=str, help="json file of quantization config")
    args = parser.parse_args()

    model = M().eval()
    
    # config = FP8Config.from_json_file(args.quant_config)

    # if config.measure:
    #     model = prepare(model, config)

    # if config.quantize:
    #     htcore.hpu_initialize()
    #     model = convert(model, config)
    #     print(model)

    with torch.no_grad():
        # htcore.hpu_initialize()
        model.to("hpu")
        output = model(torch.randn(1, 10).to("hpu"))
        torch.hpu.synchronize()
        htcore.hpu_inference_initialize(
            model,
            mark_scales=True,
            mark_non_scales=False,
            #   mark_only_scales_as_const=True
        )
        print(output)

    # if config.measure:
    #     finalize_calibration(model)
