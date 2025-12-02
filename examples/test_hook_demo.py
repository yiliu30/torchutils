import torch
import torch.nn as nn

import unittest
class ModelForTest(nn.Module):
    def __init__(self, x=1000):
        super().__init__()
        self.linear1 = nn.Linear(3 * x, 4 * x)
        self.batchnorm = nn.BatchNorm1d(4*x)
        self.linear2 = nn.Linear(4 * x, 5*x)

    def forward(self, x):
        return self.linear2(self.batchnorm(self.linear1(x)))


class HooksModelTester(unittest.TestCase):


    def test_align_devices_as_model_parallelism(self):
        
        def show_input_hook(module, input):
            print(f"Input device for module {module._name}: {[i.device for i in input]}")
            return input
        
        def show_output_hook(module, input, output):
            print(f"Output device for module {module._name}: input: { [i.device for i in input]} output: { [output.device]}")
            return output
        
        

        # Ensure previous operations on cuda:0 are complete
        # torch.cuda.synchronize(src_device)

        # 2. Run Profiler with Stack Tracing
        # The crucial part is 'with_stack=True' and CUDA/CPU activities
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True, # Recommended to trace allocations
            with_stack=True      # <--- This is what captures the Python call stack
        ) as prof:
        
        
            tensor1 = torch.ones((16, 1024, 1024))
            tensor1_cuda0 = tensor1.to("cuda:0")
            tensor1_cuda0 += 100
            print(f"Tensor on cuda:1: {tensor1_cuda0.device}, sum: {tensor1_cuda0.sum().item()}")
            tensor2_cuda1 = tensor1_cuda0.to("cuda:1")
            tensor2_cuda1 -= 100
            print(f"Tensor on cuda:1: {tensor2_cuda1.device}, sum: {tensor2_cuda1.sum().item()}")
            torch.cuda.synchronize()
            # model = ModelForTest()
            # def set_name(model):
            #     for name, module in model.named_modules():
            #         module._name = name
            # set_name(model)
            # # Everything is on CPU
            # assert model.linear1.weight.device == torch.device("cpu")
            # assert model.batchnorm.weight.device == torch.device("cpu")
            # assert model.linear2.weight.device == torch.device("cpu")

            # # This will move each submodule on different devices
            # add_hook_to_module(model.linear1, AlignDevicesHook(execution_device=0))
            # add_hook_to_module(model.batchnorm, AlignDevicesHook(execution_device=0))
            # add_hook_to_module(model.linear2, AlignDevicesHook(execution_device=1))

            # assert model.linear1.weight.device == torch.device(torch_device)
            # assert model.batchnorm.weight.device == torch.device(torch_device)
            # assert model.batchnorm.running_mean.device == torch.device(torch_device)
            # assert model.linear2.weight.device == torch.device(torch_device.replace(":0", ":1"))
            # torch.nn.modules.module.register_module_forward_pre_hook(show_input_hook)
            # torch.nn.modules.module.register_module_forward_hook(show_output_hook)
            # # We can still make a forward pass. The input does not need to be on any particular device
            # x = torch.randn(2, 3 * 1000)
            # output = model(x)
            # assert output.device == torch.device(torch_device.replace(":0", ":1"))

            # # We can add a general hook to put back output on same device as input.
            # add_hook_to_module(model, AlignDevicesHook(io_same_device=True))
            # x = torch.randn(2, 3 * 1000).to(torch_device)
            # output = model(x)
            # assert output.device == torch.device(torch_device)
            torch.cuda.synchronize()
        prof.export_chrome_trace("tensor_p2p_stack_trace_4.json")
        

# nsys profile -o p2p.copy.async.nsys-rep --trace-fork-before-exec=true --cuda-graph-trace=node   pytest -svvv test_hooks.py -k test_align_devices_as_model_parallelism 