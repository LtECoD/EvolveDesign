import numpy as np
from dataclasses import replace, astuple

from boltz.data import const
from boltz.data.types import (
    Input, 
    Structure,
    Tokenized, 
    Atom,
    Residue,
    Chain,
    Bond,
    Connection,
    Token,
    TokenBond
)
from boltz.data.tokenize.boltz import TokenData


def convert_to_atom_name(name: tuple[int, int, int, int]) -> str:
    name = [n for n in name if n > 0]
    name = [chr(n+32) for n in name]
    name = "".join(name)
    return name


def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    name = name.strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)


UNK_CONFORMER = {
    "N"     : np.array([-0.7552, 0.8886, 1.3701]),
    "CA"    : np.array([-0.4630, 0.8565, -0.0663]),
    "C"     : np.array([-1.4027, -0.0760, -0.7827]),
    "O"     : np.array([-1.7186, 0.1534, -1.9810]),
    "CB"    : np.array([1.0176, 0.5193, -0.3604]),
}
element_dict = {'N': 7, 'CA': 6, 'C': 6, 'O': 8, 'CB': 6}
chirs_dict = {'N': 0, 'CA': 0, 'C': 0, 'O': 0, 'CB': 0}


class AFGenTokenizer:
    def __init__(
        self, 
        mask_seq=False, 
        mask_msa=False, 
        min_seq_mask=0.1, 
        max_seq_mask=1.0,
        min_msa_mask=0.1,
        max_msa_mask=1.0,
    ):
        self.mask_seq = mask_seq
        self.mask_msa = mask_msa
        self.min_seq_mask = min_seq_mask
        self.max_seq_mask = max_seq_mask
        self.min_msa_mask = min_msa_mask
        self.max_msa_mask = max_msa_mask
        
    def tokenize(self, data: Input, crop_mask: np.ndarray=None, max_tokens: int=np.inf, max_atoms: int=np.inf) -> Tokenized:
        structure = data.structure
        chain_indexs = []
        for idx, chain in enumerate(structure.chains):
            chain_indexs.extend([idx] * chain['res_num'])
        chain_indexs = np.array(chain_indexs)

        if crop_mask is  None:       # take all residues
            crop_mask = np.ones(structure.residues.shape[0], dtype=bool)
        sampled_chains = set(chain_indexs[crop_mask])

        # check crop_mask, sampled chains must has protein chain
        has_protein_chain = False
        for cid in sampled_chains:
            has_protein_chain = has_protein_chain | \
                (structure.chains[cid]['mol_type'] == const.chain_type_ids['PROTEIN'])
        assert has_protein_chain, "cropped data has no protein chain"

        # mask the sequence
        if self.mask_seq:
            seq_mask_rate = np.random.uniform(self.min_seq_mask, self.max_seq_mask)
            structure = mask_protein_chain(data.structure, seq_mask_rate, crop_mask, sampled_chains)
            data = replace(data, structure=structure)

        # mask the msa
        if self.mask_msa:
            msa_mask_rate = np.random.uniform(self.min_msa_mask, self.max_msa_mask)
            msa = mask_msa(data.msa, data.structure, msa_mask_rate, crop_mask, sampled_chains)
            data = replace(data, msa=msa)
    
        tokenized = self._tokenize(data, crop_mask, sampled_chains, max_tokens=max_tokens, max_atoms=max_atoms)
        return tokenized

    def _tokenize(self, data: Input, crop_mask: np.ndarray, sampled_chains: set, max_tokens: int, max_atoms: int) -> Tokenized:
        # Get structure data
        struct = data.structure

        # Create token data
        token_data = []

        # Keep track of atom_idx to token_idx
        token_idx = 0
        atom_to_token = {}

        tokens_cnt = atom_cnt = 0
        # Filter to valid chains only
        for cid, chain in enumerate(struct.chains):
            if (not struct.mask[cid]) or (cid not in sampled_chains):   # skip invalid chains
                continue

            # Get residue indices
            res_start = chain["res_idx"]
            res_end = chain["res_idx"] + chain["res_num"]

            for rid, res in enumerate(struct.residues[res_start:res_end]):
                if not crop_mask[res_start+rid]:
                    continue                
                
                # Get atom indices
                atom_start = res["atom_idx"]
                atom_end = res["atom_idx"] + res["atom_num"]

                # Standard residues are tokens
                if res["is_standard"]:
                    # Get center and disto atoms
                    center = struct.atoms[res["atom_center"]]
                    disto = struct.atoms[res["atom_disto"]]

                    # Token is present if centers are
                    is_present = res["is_present"] & center["is_present"]
                    is_disto_present = res["is_present"] & disto["is_present"]

                    # Apply chain transformation
                    c_coords = center["coords"]
                    d_coords = disto["coords"]

                    # Create token
                    token = TokenData(
                        token_idx=token_idx,
                        atom_idx=res["atom_idx"],
                        atom_num=res["atom_num"],
                        res_idx=res["res_idx"],
                        res_type=res["res_type"],
                        sym_id=chain["sym_id"],
                        asym_id=chain["asym_id"],
                        entity_id=chain["entity_id"],
                        mol_type=chain["mol_type"],
                        center_idx=res["atom_center"],
                        disto_idx=res["atom_disto"],
                        center_coords=c_coords,
                        disto_coords=d_coords,
                        resolved_mask=is_present,
                        disto_mask=is_disto_present,
                        cyclic_period=chain["cyclic_period"],
                    )
                    
                    # Update atom_idx to token_idx
                    if (tokens_cnt+1 <= max_tokens) and (atom_cnt+int(res['atom_num']) <= max_atoms):
                        token_data.append(astuple(token))
                        for atom_idx in range(atom_start, atom_end):
                            atom_to_token[atom_idx] = token_idx
                        tokens_cnt += 1
                        atom_cnt += int(res['atom_num'])
                        token_idx += 1
                    else:
                        break

                # Non-standard are tokenized per atom
                else:
                    # We use the unk protein token as res_type
                    unk_token = const.unk_token["PROTEIN"]
                    unk_id = const.token_ids[unk_token]

                    # Get atom coordinates
                    atom_data = struct.atoms[atom_start:atom_end]
                    atom_coords = atom_data["coords"]

                    if (tokens_cnt+len(atom_data) <= max_tokens) and (atom_cnt+len(atom_data) <= max_atoms):
                        # Tokenize each atom
                        for i, atom in enumerate(atom_data):
                            # Token is present if atom is
                            is_present = res["is_present"] & atom["is_present"]
                            index = atom_start + i

                            # Create token
                            token = TokenData(
                                token_idx=token_idx,
                                atom_idx=index,
                                atom_num=1,
                                res_idx=res["res_idx"],
                                res_type=unk_id,
                                sym_id=chain["sym_id"],
                                asym_id=chain["asym_id"],
                                entity_id=chain["entity_id"],
                                mol_type=chain["mol_type"],
                                center_idx=index,
                                disto_idx=index,
                                center_coords=atom_coords[i],
                                disto_coords=atom_coords[i],
                                resolved_mask=is_present,
                                disto_mask=is_present,
                                cyclic_period=chain["cyclic_period"]
                            )
                            # Update atom_idx to token_idx
                            token_data.append(astuple(token))
                            atom_to_token[index] = token_idx
                        
                            token_idx += 1

                        tokens_cnt += len(atom_data)
                        atom_cnt += len(atom_data)
                    else:
                        break

        # Create token bonds
        token_bonds = []

        # Add atom-atom bonds from ligands
        for bond in struct.bonds:
            if (
                bond["atom_1"] not in atom_to_token
                or bond["atom_2"] not in atom_to_token
            ):
                continue
            token_bond = (
                atom_to_token[bond["atom_1"]],
                atom_to_token[bond["atom_2"]],
            )
            token_bonds.append(token_bond)

        # Add connection bonds (covalent)
        for conn in struct.connections:
            if (
                conn["atom_1"] not in atom_to_token
                or conn["atom_2"] not in atom_to_token
            ):
                continue
            token_bond = (
                atom_to_token[conn["atom_1"]],
                atom_to_token[conn["atom_2"]],
            )
            token_bonds.append(token_bond)

        token_data = np.array(token_data, dtype=Token)
        token_bonds = np.array(token_bonds, dtype=TokenBond)
        tokenized = Tokenized(
            token_data,
            token_bonds,
            data.structure,
            data.msa,
        )
        return tokenized


def mask_msa(msa: dict, structure: Structure, mask_rate: float, crop_mask: np.ndarray, sampled_chains: set):
    # residues has been masked, of which the res_type is negative
    masked_dict = {chain['entity_id']: np.zeros(chain['res_num'], dtype=bool) for chain in structure.chains}

    for chain_id in msa:
        if chain_id not in sampled_chains:
            continue
        chain = [c for c in structure.chains if c['asym_id'] == chain_id][0]
        chain_res_start, chain_res_num = chain['res_idx'], chain['res_num']
        res_types = structure.residues['res_type'][chain_res_start: chain_res_start+chain_res_num]
        
        # to remember msa of which residues are masked
        indices = np.argwhere(res_types<0).squeeze(-1)
        masked = masked_dict[chain['entity_id']]
        indices = indices[~masked[indices]]  # not masked

        if len(indices) > 0:
            masked[indices] = True
            masked_dict[chain['entity_id']] = masked

            residues = msa[chain_id].residues
            sequences = msa[chain_id].sequences
            res_starts, res_ends = sequences['res_start'], sequences['res_end']
            
            assert res_starts[0] == 0, "the first msa sequence must starts from 0"
            for idx in indices:
                mask_indices = res_starts + idx
                assert mask_indices.shape == res_ends.shape
                mask_indices = mask_indices[mask_indices < res_ends]    # not exceed end
                mask_indices = mask_indices[
                    residues['res_type'][mask_indices] != const.token_ids['-']  # not mask "-"
                ]
                _rand = np.random.rand(len(mask_indices))
                _rand[0] = 0.       # the first msa squence must be masked
                mask_indices = mask_indices[_rand <= mask_rate]
                # residues['res_type'][mask_indices] = const.token_ids['UNK']
                residues['res_type'][mask_indices] = const.token_ids['-']

            msa[chain_id] = replace(msa[chain_id], residues=residues)

    return msa


def mask_protein_chain(structure: Structure, mask_rate: float, crop_mask: np.ndarray, sampled_chains: set) -> Structure:
    atoms = structure.atoms
    bonds = structure.bonds
    residues = structure.residues
    chains = structure.chains
    connections = structure.connections
    entity_ids = chains['entity_id']

    # build cancidate chains
    chain_candidate_mask = {}
    for cid in sampled_chains:
        if (not structure.mask[cid]) or (structure.chains[cid]['mol_type'] != const.chain_type_ids['PROTEIN']):
            continue
        # only mask standard in the cropped region
        chain = chains[cid]
        rstart, rnum = chain['res_idx'], chain['res_num']
        chain_crop_mask = crop_mask[rstart: rstart+rnum]
        res_types = residues['res_type'][rstart: rstart+rnum]
        is_standard = (res_types >= const.token_ids["ALA"]) & (res_types <= const.token_ids["VAL"])
        candidate_mask = chain_crop_mask & is_standard
        if np.sum(candidate_mask) > 0:
            chain_candidate_mask[cid] = candidate_mask
    assert len(chain_candidate_mask) > 0, "no valid protein chain"

    # randomly select one chain to be masked
    cid = np.random.choice(list(chain_candidate_mask.keys()), 1)[0]
    chain = chains[cid]
    res_start, res_num = chain['res_idx'], chain['res_num']
    candidate_mask = chain_candidate_mask[cid]
    candidate_num = np.sum(candidate_mask)

    # record residues to be masked
    chain_residue_is_masked = {}
    
    # make sure at least one residue is masked
    mask_num = int(candidate_num * mask_rate)
    if mask_num == 0:
        mask_num = 1
    
    # randomly select residues to be masked
    indices = np.argwhere(candidate_mask).squeeze(-1)
    indices = np.random.choice(indices, mask_num, replace=False)  # select randomly mask indices
    assert len(indices) > 0
    residue_is_masked = np.zeros(res_num, dtype=bool)
    residue_is_masked[indices] = True
    chain_residue_is_masked[cid] = residue_is_masked

    # mask same residues for chains with the same entity_id
    chain_indices = set(np.where(entity_ids == entity_ids[cid])[0].tolist())
    for _cid in chain_indices:
        if (not structure.mask[_cid]) or (_cid not in sampled_chains) or (_cid == cid):
            continue
        assert _cid not in chain_residue_is_masked
        assert chains[_cid]['mol_type'] == const.chain_type_ids['PROTEIN'], \
            f"chain {_cid} is not protein, but {const.chain_types[chains[_cid]['mol_type']]}"
        assert chains[_cid]['res_num'] == res_num, "chains share the same entity_id must have same number of atoms"
        _res_start = chains[_cid]['res_idx']
        
        # only mask standard residues
        _chain_crop_mask = crop_mask[_res_start: _res_start+res_num]
        _res_types = residues['res_type'][_res_start: _res_start+res_num]
        _is_standard = (_res_types >= const.token_ids["ALA"]) & (_res_types <= const.token_ids["VAL"])
        _candidate_mask = _chain_crop_mask & _is_standard
        chain_residue_is_masked[_cid] = residue_is_masked & _candidate_mask

    # turn masked residue to UNK, and reorganize atom, residue and chain
    new_chains = []
    new_residues = []
    new_atoms = []
    atom_map = {}
    res_map = {}

    cur_atom_idx = 0
    cur_res_idx = 0
    for cid in range(len(chains)):
        chain_atom_idx, chain_atom_num = cur_atom_idx, 0
        chain_res_idx, chain_res_num = cur_res_idx, 0

        chain = chains[cid]
        rstart, rnum = int(chain['res_idx']), int(chain['res_num'])
        chain_residues = residues[rstart: rstart+rnum]
        residue_is_masked = chain_residue_is_masked.get(cid, np.zeros(rnum, dtype=bool))
        assert len(residue_is_masked) == rnum, "residue mask length is not equal to residue num"

        # check all masked residues are standard AA
        masked_res_type = residues['res_type'][rstart: rstart+rnum][residue_is_masked]
        if len(masked_res_type) > 0:
            assert ((masked_res_type >= const.token_ids["ALA"]) & (masked_res_type <= const.token_ids["VAL"])).all(), \
                "masked residues must be standard residues"

        for rid in range(rnum):
            residue = chain_residues[rid]
            astart, anum = int(residue['atom_idx']), int(residue['atom_num'])
            residue_atoms = atoms[astart: astart+anum]
            atom_name_index = {convert_to_atom_name(name): _id for _id, name in enumerate(residue_atoms['name'])}
            assert anum > 0, "a residue has no atom"

            residue_atom_idx = cur_atom_idx
            if not residue_is_masked[rid]:      # if the residue is not masked
                new_atoms.extend(residue_atoms)
                new_residues.append(np.array((
                    residue['name'],
                    residue['res_type'],
                    residue['res_idx'],
                    residue_atom_idx,
                    anum,
                    residue['atom_center'] - astart + residue_atom_idx,
                    residue['atom_disto'] - astart + residue_atom_idx,
                    residue['is_standard'],
                    residue['is_present']
                ), dtype=Residue))

                _map = dict(zip(range(astart, astart+anum), range(residue_atom_idx, residue_atom_idx+anum)))
                atom_map.update(_map)
                chain_atom_num += anum
                cur_atom_idx += anum
            else:                               # the residue is masked
                assert const.chain_types[chain['mol_type']] == "PROTEIN", 'only protein can be masked'

                for name in ['N', 'CA', 'C', 'O', 'CB']:
                    aindex = atom_name_index.get(name, None)
                    new_atoms.append(np.array((
                        convert_atom_name(name),
                        element_dict[name],
                        0,
                        residue_atoms['coords'][aindex] if aindex else [0., 0., 0.],
                        UNK_CONFORMER[name],
                        True,
                        chirs_dict[name]
                    ), dtype=Atom))
                    if aindex:
                        assert int(residue_atoms['element'][aindex]) == element_dict[name] and \
                            int(residue_atoms['chirality'][aindex]) == chirs_dict[name], 'element or chirality is inconsistent'
                        atom_map[aindex+astart] = cur_atom_idx
                    cur_atom_idx += 1
            
                new_residues.append(np.array((
                    "UNK",
                    -1 * residue['res_type'],  #! to inform featurizer which token are masked 
                    residue['res_idx'],
                    residue_atom_idx,
                    5,
                    residue_atom_idx+1, 
                    residue_atom_idx+4,
                    True,
                    True
                ), dtype=Residue))
                chain_atom_num += 5

            res_map[rstart+rid] = cur_res_idx
            chain_res_num += 1
            cur_res_idx += 1

        cyclic_period = 0
        new_chains.append(np.array((
            chain['name'],
            chain['mol_type'],
            chain['entity_id'],
            chain['sym_id'],
            chain['asym_id'],
            chain_atom_idx,
            chain_atom_num,
            chain_res_idx,
            chain_res_num,
            cyclic_period
        ), dtype=Chain))
    
    # reorganize bond and connection 
    new_bonds = []
    new_connections = []
    for bond in bonds:
        a1, a2 = int(bond['atom_1']), int(bond['atom_2'])
        if a1 in atom_map and a2 in atom_map:
            new_bonds.append(np.array((atom_map[a1], atom_map[a2], bond['type']), dtype=Bond))
    for conn in connections:
        c1, c2 = int(conn['chain_1']), int(conn['chain_2'])
        r1, r2 = int(conn['res_1']), int(conn['res_2'])
        a1, a2 = int(conn['atom_1']), int(conn['atom_2'])
        if r1 in res_map and r2 in res_map and a1 in atom_map and a2 in atom_map:
            new_connections.append(
                np.array((c1, c2, res_map[r1], res_map[r2], atom_map[a1], atom_map[a2]), dtype=Connection))

    chain_data = np.array(new_chains, dtype=Chain)
    residue_data = np.array(new_residues, dtype=Residue)
    atom_data = np.array(new_atoms, dtype=Atom)
    bond_data = np.array(new_bonds, dtype=Bond)
    conn_data = np.array(new_connections, dtype=Connection)

    # check structure
    residue_atom_num_sum = np.sum(residue_data['atom_num'])
    chain_atom_num_sum = np.sum(chain_data['atom_num'])
    chain_residue_num_sum = np.sum(chain_data['res_num'])
    assert residue_atom_num_sum == chain_atom_num_sum == len(atom_data), "atom num is inconsistent"
    for k, v in res_map.items():
        assert k == v, "residue num changed"
    assert (residue_data['atom_num'] > 0).all(), 'some residue has no atom'
    assert (chain_data['res_num'] > 0).all(), "some chain has no residue"
    assert chain_residue_num_sum == len(residue_data)
    assert np.sum(residue_data['res_type'] < 0) > 0, "no residue is masked"
    
    structure = replace(structure, 
                        atoms=atom_data, 
                        bonds=bond_data, 
                        residues=residue_data, 
                        chains=chain_data, 
                        connections=conn_data,
                        )
    return structure