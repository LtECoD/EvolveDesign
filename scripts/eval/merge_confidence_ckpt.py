import torch
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./output/sequence/train_mask_msa/version_1/checkpoints/epoch=2-step=4689.ckpt")
    parser.add_argument("--conf_ckpt", type=str, 
                        default="/share/home/yangsen/.cache/huggingface/hub/models--boltz-community--boltz-1/snapshots/7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1/boltz1_conf.ckpt")
    parser.add_argument("--output", type=str, default="./output/sequence/train_mask_msa/version_1/checkpoints/epoch=2_w_conf.ckpt")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    conf = torch.load(args.conf_ckpt, map_location="cpu", weights_only=False)

    # update state dict
    state_dict = ckpt['state_dict']
    conf_params = {name: param
        for name, param in conf['state_dict'].items()
            if name.startswith("confidence_module") or \
                name.startswith("structure_module.out_token_feat_update")
    }
    state_dict.update(conf_params)
    
    # update hyper parameters
    hyper_parameters = ckpt['hyper_parameters']
    hyper_parameters['confidence_prediction'] = True
    hyper_parameters['load_confidence_from_trunk'] = True
    hyper_parameters['confidence_imitate_trunk'] = True
    
    # remove the optimizer state and ema
    ckpt.pop("optimizer_states", None)
    ckpt.pop("ema", None)
    
    ckpt['state_dict'] = state_dict
    ckpt['hyper_parameters'] = hyper_parameters
    torch.save(ckpt, args.output)    


if __name__ == "__main__":
    main()