# Reproduced https://pytorch.org/tutorials/recipes/regional_compilation.html


"""
This feature is available starting with the 2.5 release.
If you are using version 2.4, you can enable the configuration flag torch._dynamo.config.inline_inbuilt_nn_modules=True to prevent recompilations during regional compilation.
In version 2.5, this flag is enabled by default.
"""

from time import perf_counter
import torch
import torch.nn as nn


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(10, 10)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        a = self.linear1(x)
        a = self.relu1(a)
        a = torch.sigmoid(a)
        b = self.linear2(a)
        b = self.relu2(b)
        return b


class Model(torch.nn.Module):
    def __init__(self, apply_regional_compilation):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        # Apply compile only to the repeated layers.
        if apply_regional_compilation:
            self.layers = torch.nn.ModuleList([torch.compile(Layer()) for _ in range(64)])
        else:
            self.layers = torch.nn.ModuleList([Layer() for _ in range(64)])

    def forward(self, x):
        # In regional compilation, the self.linear is outside of the scope of `torch.compile`.
        x = self.linear(x)
        for layer in self.layers:
            x = layer(x)
        return x


model = Model(apply_regional_compilation=False).cuda()
full_compiled_model = torch.compile(model)


regional_compiled_model = Model(apply_regional_compilation=True).cuda()


def measure_latency(fn, input):
    # Reset the compiler caches to ensure no reuse between different runs
    torch.compiler.reset()
    with torch._inductor.utils.fresh_inductor_cache():
        start = perf_counter()
        fn(input)
        torch.cuda.synchronize()
        end = perf_counter()
        return end - start


input = torch.randn(10, 10, device="cuda")
full_model_compilation_latency = measure_latency(full_compiled_model, input)
print(f"Full model compilation time = {full_model_compilation_latency:.2f} seconds")

regional_compilation_latency = measure_latency(regional_compiled_model, input)
print(f"Regional compilation time = {regional_compilation_latency:.2f} seconds")

assert regional_compilation_latency < full_model_compilation_latency
