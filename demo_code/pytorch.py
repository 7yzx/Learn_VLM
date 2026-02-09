import torch
import torch.fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

from torch.overrides import TorchFunctionMode, resolve_name
from torch.utils._python_dispatch import TorchDispatchMode


# https://docs.pytorch.org/docs/2.9/notes/extending.html#extending-all-torch-api-with-modes
class FunctionLog(TorchFunctionMode):

    def __torch_function__(self, func, types, args, kwargs=None):
        # print(f"Function Log: {resolve_name(func)}(*{args}, **{kwargs})")
        print(f"Function Log: {resolve_name(func)}")
        return func(*args, **(kwargs or {}))


class DispatchLog(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        # print(f"Dispatch Log: {func}(*{args}, **{kwargs})")
        print(f"Dispatch Log: {func}")
        return func(*args, **(kwargs or {}))


class MLP(torch.nn.Module):

    def __init__(self, input_dim=16, hidden_dim=32, output_dim=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":

    input_dim = 16
    hidden_dim = 32
    output_dim = 16

    module = MLP(input_dim=input_dim,
                 hidden_dim=hidden_dim,
                 output_dim=output_dim)
    module.eval()

    # Torch FX Symbolic Tracing
    torch_symbolic_traced = torch.fx.symbolic_trace(module)
    torch_symbolic_traced_graph_drawer = FxGraphDrawer(
        torch_symbolic_traced, "mlp_symbolic_traced_graph_drawer")
    print("Torch FX Symbolic Traced Graph:")
    print(torch_symbolic_traced.graph)

    # Get the graph object and write to a file
    with open("mlp_symbolic_traced_graph.svg", "wb") as f:
        f.write(
            torch_symbolic_traced_graph_drawer.get_dot_graph().create_svg())

    # Torch Export to ATen IR
    args = (torch.randn(1, input_dim), )
    exported_program = torch.export.export(module, args)
    exported_aten_graph_module = exported_program.graph_module
    print("MLP Exported ATen Graph:")
    print(exported_aten_graph_module)

    exported_aten_graph_drawer = FxGraphDrawer(
        exported_aten_graph_module, "mlp_exported_aten_graph_drawer")
    with open("mlp_exported_aten_graph.svg", "wb") as f:
        f.write(exported_aten_graph_drawer.get_dot_graph().create_svg())

    # Torch Export to Core ATen IR
    core_aten_exported_program = exported_program.run_decompositions()
    core_aten_exported_aten_graph_module = core_aten_exported_program.graph_module
    print("MLP Core ATen Exported Graph:")
    print(core_aten_exported_aten_graph_module)

    core_aten_exported_aten_graph_drawer = FxGraphDrawer(
        core_aten_exported_aten_graph_module,
        "mlp_core_aten_exported_aten_graph_drawer")
    with open("mlp_core_aten_exported_aten_graph.svg", "wb") as f:
        f.write(
            core_aten_exported_aten_graph_drawer.get_dot_graph().create_svg())

    print("TorchFunctionMode logging:")
    with torch.inference_mode(), FunctionLog():
        # result = module(*args)
        # result = torch_symbolic_traced(*args)
        # result = exported_program.module()(*args)
        result = core_aten_exported_program.module()(*args)

    print("TorchDispatchMode logging:")
    with torch.inference_mode(), DispatchLog():
        # result = module(*args)
        # result = torch_symbolic_traced(*args)
        # result = exported_program.module()(*args)
        result = core_aten_exported_program.module()(*args)