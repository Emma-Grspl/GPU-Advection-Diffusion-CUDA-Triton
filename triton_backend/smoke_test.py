import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = (
        tl.program_id(axis=0)
        * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )

    mask = offsets < n_elements

    x = tl.load(
        x_ptr + offsets,
        mask=mask,
    )

    y = tl.load(
        y_ptr + offsets,
        mask=mask,
    )

    tl.store(
        out_ptr + offsets,
        x + y,
        mask=mask,
    )


def main():
    print("PyTorch:", torch.__version__)
    print("Triton:", triton.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Compute capability:",
        torch.cuda.get_device_capability(0),
    )

    n = 1024

    x = torch.randn(
        n,
        device="cuda",
        dtype=torch.float32,
    )

    y = torch.randn_like(x)

    out = torch.empty_like(x)

    grid = (
        triton.cdiv(n, 256),
    )

    add_kernel[grid](
        x,
        y,
        out,
        n_elements=n,
        BLOCK_SIZE=256,
    )

    torch.cuda.synchronize()

    error = (
        out
        - (x + y)
    ).abs().max().item()

    print("max error:", error)

    if error == 0.0:
        print("TRITON SMOKE TEST: PASS")
    else:
        raise RuntimeError(
            "Triton numerical mismatch."
        )


if __name__ == "__main__":
    main()
