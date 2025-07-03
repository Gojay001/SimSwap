import torch
import torch.nn as nn

from models.fs_networks_fix import Generator_Adain_Upsample, Generator_Adain_Upsample_DS2


if __name__ == "__main__":
    model = Generator_Adain_Upsample_DS2(input_nc=3, output_nc=3)
    input_tensor = torch.randn(1, 3, 256, 256)
    dlatents = torch.randn(1, 512)

    mode = 'profile'
    # mode = 'onnx'

    if mode == 'profile':
        try:
            from thop import profile
            from thop.vision.basic_hooks import count_parameters

            custom_ops = {}
            def count_relu(m, x, y):
                x = x[0]
                nelements = x.numel()
                m.total_ops += nelements

            custom_ops[nn.ReLU] = count_relu

            flops, params = profile(model, inputs=(input_tensor, dlatents), custom_ops=custom_ops)
            # print(f'FLOPs: {flops}, Params: {params}')
            print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")
        except Exception as e:
            print(f"Error in thop: {e}")

    elif mode == 'onnx':
        model.eval()
        input_names = ['input_image', 'latent_code']
        output_names = ['output_image']

        dynamic_axes = {
            'input_image': {0: 'batch_size'},
            'latent_code': {0: 'batch_size'},
            'output_image': {0: 'batch_size'}
        }

        try:
            torch.onnx.export(model,
                            (input_tensor, dlatents),
                            "./models/SimSwap.onnx",
                            verbose=True,
                            input_names=input_names,
                            output_names=output_names,
                            opset_version=11,
                            do_constant_folding=True,
                            dynamic_axes=dynamic_axes)
            print("Success to export onnx.")
        except Exception as e:
            print(f"Error in onnx: {e}")

    else:
        print("Unsupported mode. Use 'profile' or 'onnx'.")
