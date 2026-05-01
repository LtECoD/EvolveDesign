from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist

from boltz.data import const
from boltz.data.types import Structure


def pick_random_residue(
    mask: np.ndarray,
    random: np.random.RandomState,
) -> np.ndarray:
    true_indices = np.where(mask)[0]
    random_true_index = random.choice(true_indices)
    return random_true_index    


def pick_chain_residue(
    structure: Structure,
    chain_id: int,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random residue from a chain.

    Parameters
    ----------
    tokens : np.ndarray
        The token data.
    chain_id : int
        The chain ID.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    # Filter to chain
    chain = structure.chains[chain_id]    
    res_start, res_num = chain['res_idx'], chain['res_num']
    residues = structure.residues[res_start: res_start+res_num]
    # will not select an UNK residue as query

    if structure.mask[chain_id]:
        mask = residues['is_present'] & (residues['res_type'] != const.unk_token_ids['PROTEIN'])
    else:
        raise ValueError("mask of the sampled chain is False")

    # Pick from chain, fallback to all tokens
    if mask.any():
        query = pick_random_residue(mask, random)
        query = query + res_start
    else:
        raise ValueError("No residues found in chain")

    return query


def pick_interface_residue(
    structure: np.ndarray,
    interface: np.ndarray,
    random: np.random.RandomState,
) -> np.ndarray:
    """Pick a random token from an interface.

    Parameters
    ----------
    structure : Structure
        The token data.
    interface : int
        The interface ID.
    random : np.ndarray
        The random state for reproducibility.

    Returns
    -------
    np.ndarray
        The selected token.

    """
    # Sample random interface
    cid1 = int(interface["chain_1"])
    cid2 = int(interface["chain_2"])
    chain_1 = structure.chains[cid1]
    chain_2 = structure.chains[cid2]
    
    residues_1 = structure.residues[chain_1["res_idx"]: chain_1["res_idx"]+chain_1["res_num"]]
    residues_2 = structure.residues[chain_2["res_idx"]: chain_2["res_idx"]+chain_2["res_num"]]
    # select query only from protein chains
    if (structure.mask[cid1]) and (chain_1['mol_type'] == const.chain_type_ids['PROTEIN']):
        mask_1 = residues_1['is_present']
    else:
        mask_1 = np.zeros(chain_1['res_num'], dtype=bool)
    if (structure.mask[cid2]) and (chain_2['mol_type'] == const.chain_type_ids['PROTEIN']):
        mask_2 = residues_2['is_present']
    else:
        mask_2 = np.zeros(chain_2['res_num'], dtype=bool)
    # will not select an UNK residue as query
    if chain_1['mol_type'] == const.chain_type_ids['PROTEIN']:
        mask_1 = mask_1 & (residues_1['res_type'] != const.unk_token_ids['PROTEIN'])
    if chain_2['mol_type'] == const.chain_type_ids['PROTEIN']:
        mask_2 = mask_2 & (residues_2['res_type'] != const.unk_token_ids['PROTEIN'])
    
    atoms = structure.atoms
    # If no interface, pick from the chains
    if mask_1.any() and (not mask_2.any()):
        query = pick_random_residue(mask_1, random)
        query = query + chain_1['res_idx']
    elif mask_2.any() and (not mask_1.any()):
        query = pick_random_residue(mask_2, random)
        query = query + chain_2['res_idx']

    elif (not mask_1.any()) and (not mask_2.any()):
        raise ValueError("No valid residues found in interface")
    else:        
        residues_1_coords = atoms['coords'][residues_1['atom_center']]
        residues_2_coords = atoms['coords'][residues_2['atom_center']]
        
        dists = cdist(residues_1_coords, residues_2_coords)
        cuttoff = dists < const.interface_cutoff

        # In rare cases, the interface cuttoff is slightly
        # too small, then we slightly expand it if it happens
        if not np.any(cuttoff):
            cuttoff = dists < (const.interface_cutoff + 5.0)

        res_mask_1 = np.any(cuttoff, axis=1) & mask_1
        res_mask_2 = np.any(cuttoff, axis=0) & mask_2

        if (not res_mask_1.any()) and (not res_mask_2.any()):
            raise ValueError(f"No valid residues pairs found in interface")

        # Select random token
        candidates = np.concatenate([res_mask_1, res_mask_2])
        query = pick_random_residue(candidates, random)
        
        if query < len(res_mask_1):
            query = query + chain_1['res_idx']
        else:
            query = query - chain_1['res_num'] + chain_2['res_idx']

    return query


class AFGenCropper:
    def __init__(self, min_neighborhood: int = 0, max_neighborhood: int = 40) -> None:
        """Initialize the cropper.

        Modulates the type of cropping to be performed.
        Smaller neighborhoods result in more spatial
        cropping. Larger neighborhoods result in more
        continuous cropping. A mix can be achieved by
        providing a range over which to sample.

        Parameters
        ----------
        min_neighborhood : int
            The minimum neighborhood size, by default 0.
        max_neighborhood : int
            The maximum neighborhood size, by default 40.

        """
        sizes = list(range(min_neighborhood, max_neighborhood + 1, 2))
        self.neighborhood_sizes = sizes

    def crop(
        self,
        structure: Structure,
        max_tokens: int,
        random: np.random.RandomState,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,
        interface_id: Optional[int] = None,
    ) -> np.ndarray:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        structure : Structure
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : int, optional
            The maximum number of atoms to consider.
        chain_id : int, optional
            The chain ID to crop.
        interface_id : int, optional
            The interface ID to crop.
        """
        # Check inputs
        if chain_id is not None and interface_id is not None:
            msg = "Only one of chain_id or interface_id can be provided."
            raise ValueError(msg)
        if chain_id is not None:
            chain = structure.chains[chain_id]
            assert structure.mask[chain_id], "mask of the sampled chain is False"
            assert chain['mol_type'] == const.chain_type_ids['PROTEIN'], \
                "chain to crop is not a protein"
        if interface_id is not None:
            interface = structure.interfaces[interface_id]
            assert structure.mask[int(interface["chain_1"])] or structure.mask[int(interface["chain_2"])], \
                "mask of the sampled interface is False"
            assert (structure.chains[int(interface["chain_1"])]['mol_type'] == const.chain_type_ids['PROTEIN']) or \
                (structure.chains[int(interface["chain_2"])]['mol_type'] == const.chain_type_ids['PROTEIN']), \
                    "interface includes no protein"

        # Randomly select a neighborhood size
        neighborhood_size = random.choice(self.neighborhood_sizes)

        # Get token data
        mask = structure.mask
        chains = structure.chains
        interfaces = structure.interfaces
        residues = structure.residues
        atoms = structure.atoms
        chain_indexs = []
        for idx, chain in enumerate(chains):
            chain_indexs.extend([idx] * chain['res_num'])
        chain_indexs = np.array(chain_indexs)

        # Filter to valid interfaces
        valid_interfaces = interfaces
        valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_1"]]]
        valid_interfaces = valid_interfaces[mask[valid_interfaces["chain_2"]]]

        # Pick a random token, chain, or interface
        if chain_id is not None:
            query = pick_chain_residue(structure, chain_id, random)
        elif interface_id is not None:
            interface = interfaces[interface_id]
            query = pick_interface_residue(structure, interface, random)
        else:
            raise ValueError("No chain or interface ID provided")

        assert residues[query]['is_present'], "query residue is not present"
        assert chains[chain_indexs[query]]['mol_type'] == const.chain_type_ids['PROTEIN'], "query residue is not a protein"
        assert mask[chain_indexs[query]], "mask of chain contains query is false"

        # Sort all tokens by distance to query_coords
        dists = atoms['coords'][residues['atom_center']] - atoms['coords'][residues['atom_center'][query]]
        dists[residues['is_present'] == False] = np.inf
        indices = np.argsort(np.linalg.norm(dists, axis=1))

        # Select cropped indices
        cropped: set[int] = set()
        total_atoms = 0
        total_tokens = 0
        for idx in indices:
            cindex = chain_indexs[idx]
            #! non-standard residues (like ccd molecue) may be in cropped data
            if (not residues['is_present'][idx]) or (not mask[cindex]):
                continue

            rstart, rnum = chains[cindex]['res_idx'], chains[cindex]['res_num']
            
            # Pick the whole chain if possible, otherwise select
            # a contiguous subset centered at the query token
            if rnum <= neighborhood_size:
                new_indices = list(range(rstart, rstart + rnum))
            else:
                # First limit to the maximum set of tokens, with the
                # neighboorhood on both sides to handle edges. This
                # is mostly for efficiency with the while loop below.
                
                res_idxs = residues['res_idx'][rstart: rstart + rnum]
                res_idx = residues['res_idx'][idx]
                min_idx = res_idx - neighborhood_size
                max_idx = res_idx + neighborhood_size
                
                # Start by adding just the query token
                res_mask = np.array([0]*rnum, dtype=bool)
                res_mask[idx-rstart] = True
        
                # Expand the neighborhood until we have enough tokens, one
                # by one to handle some edge cases with non-standard chains.
                # We switch to the res_idx instead of the token_idx to always
                # include all tokens from modified residues or from ligands.
                min_idx = max_idx = res_idx
                while np.sum(res_mask) < neighborhood_size:
                    min_idx = min_idx - 1
                    max_idx = max_idx + 1
                    res_mask = (res_idxs >= min_idx) & (res_idxs <= max_idx)
                    
                new_indices = np.arange(rstart, rstart + rnum)[res_mask]

            # Compute new tokens and new atoms
            new_indices = set(new_indices) - cropped
            new_atoms = np.sum(residues["atom_num"][list(new_indices)])
            new_tokens = np.sum([1 if residues[_index]['is_standard'] else residues[_index]['atom_num'] for _index in new_indices])

            # Stop if we exceed the max number of tokens or atoms
            if ((max_tokens is not None) and ((total_tokens + new_tokens) > max_tokens)) or \
                ((max_atoms is not None) and ((total_atoms + new_atoms) > max_atoms)):
                break

            # Add new indices
            cropped.update(new_indices)
            total_atoms += new_atoms
            total_tokens += new_tokens

        cropped_residue_mask = np.zeros(len(residues), dtype=bool)
        cropped_residue_mask[list(cropped)] = True
        return cropped_residue_mask
