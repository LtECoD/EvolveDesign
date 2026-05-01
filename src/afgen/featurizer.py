import torch
import numpy as np
from dataclasses import replace

from boltz.data import const 
from boltz.data.types import  Tokenized
from boltz.data.feature.featurizer import BoltzFeaturizer


class AFGenFeaturizer(BoltzFeaturizer):
    def __init__(self, check_mask=True):
        self.check_mask = check_mask
        
    def process(self, data: Tokenized, **kwargs):
        # to add masked features
        if self.check_mask:
            token_data = data.tokens
            res_types = token_data['res_type']
            mask_indices = np.where(res_types<0)[0]
            mask_target = -1*res_types[mask_indices]
            mask_rate = len(mask_indices) / len(res_types)
            assert len(mask_indices) > 0, "no masked tokens"

            # turn masked tokens to UNK for TokenData
            res_types[mask_indices] = const.token_ids['UNK']
            token_data['res_type'] = res_types
            # turn masked tokens to UNK in Structure
            structure = data.structure
            residues = structure.residues
            struc_res_types = residues['res_type']
            struc_res_types[struc_res_types<0] = const.token_ids['UNK']
            structure = replace(structure, residues=residues)
            data = replace(data, tokens=token_data, structure=structure)

            features = super().process(data, **kwargs)

            msa_tokens = torch.argmax(features["msa"], dim=-1)
            msa_mask_rate = torch.sum(msa_tokens[:, mask_indices] == const.unk_token_ids['PROTEIN']) / \
                (msa_tokens.size(0) * len(mask_indices))

            features.update({
                "mask_msa_rate": torch.Tensor([msa_mask_rate]),
                "mask_seq_rate": torch.Tensor([mask_rate]),
                "mask_indices": torch.LongTensor(mask_indices),
                "mask_target": torch.LongTensor(mask_target)
            })
        else:
            features = super().process(data, **kwargs)

        return features
