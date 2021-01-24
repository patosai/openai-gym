#!/usr/bin/env python3

import torch

if __name__ == "__main__":
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(2, 2),
    #     torch.nn.Linear(2, 1)
    # )
    # inputs = [torch.randn(2) for _ in range(20)]
    # for input in inputs:
    #     output = model(input)
    #     output.backward()
    # for param in model.parameters():
    #     print("Param")
    #     print(param)
    #     print(param.grad)

    # frequencies = torch.Tensor([0.3, 0.5, 0.2])
    # result = torch.multinomial(frequencies, 1, replacement=True)
    # print(result)
    # print(result.grad)

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2)
    )
    input = torch.Tensor([3, 5])
    output = model(input)
    print(output)
    result = torch.multinomial(output, 1, replacement=True)
    print(result)
    print(result.grad)
