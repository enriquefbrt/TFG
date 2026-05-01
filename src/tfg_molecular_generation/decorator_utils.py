import re
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


DECORATOR_BLOCK_RE = re.compile(r"<R(\d+)>\s*(.*?)\s*</R\1>", re.DOTALL)


def is_decorator_sequence(text: str) -> bool:
    if not text:
        return False
    return bool(DECORATOR_BLOCK_RE.search(text))


def _find_dummy_with_label(mol: Chem.Mol, label: int) -> List[int]:
    return [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == label
    ]


def _single_neighbor_idx(mol: Chem.Mol, atom_idx: int) -> int:
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
    if len(neighbors) != 1:
        raise ValueError(
            f"Dummy atom at index {atom_idx} must have exactly one neighbor; got {len(neighbors)}."
        )
    return neighbors[0]


def _extract_labels_from_scaffold(scaffold_mol: Chem.Mol) -> List[int]:
    labels = []
    for atom in scaffold_mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            label = atom.GetAtomMapNum()
            if label <= 0:
                raise ValueError("Found dummy atom in scaffold without positive atom-map label.")
            labels.append(label)
    return sorted(set(labels))


def parse_decorator_sequence(decorators_text: str) -> Dict[int, str]:
    parsed: Dict[int, str] = {}
    for match in DECORATOR_BLOCK_RE.finditer(decorators_text or ""):
        label = int(match.group(1))
        smiles = (match.group(2) or "").strip()
        if not smiles:
            continue
        if label in parsed:
            raise ValueError(f"Duplicate decorator block for R{label}.")
        parsed[label] = smiles
    return parsed


def smiles_to_scaffold_and_decorators(smiles: str) -> Optional[Tuple[str, str]]:
    """
    Converts a molecule into:
    - input_text: scaffold with labeled attachment points [*:i]
    - target_text: decorator sequence: <R1> [*:1]... </R1> ...

    Returns None if decomposition is not possible or not useful for decoration training.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core is None or core.GetNumAtoms() == 0:
        return None

    # Side chains with dummy labels linked to core atom indices.
    side_chains = Chem.ReplaceCore(mol, core, labelByIndex=True)
    if side_chains is None:
        return None

    fragments = Chem.GetMolFrags(side_chains, asMols=True, sanitizeFrags=True)
    if not fragments:
        # Molecule equals the scaffold and has no decorations.
        return None

    # Each fragment should have exactly one attachment point for simple scaffold decoration.
    fragment_entries: List[Tuple[int, Chem.Mol]] = []
    for frag in fragments:
        dummies = [atom for atom in frag.GetAtoms() if atom.GetAtomicNum() == 0]
        if len(dummies) != 1:
            return None
        core_atom_idx = dummies[0].GetIsotope()
        if core_atom_idx is None or core_atom_idx < 0:
            return None
        fragment_entries.append((int(core_atom_idx), frag))

    unique_core_atom_indices = sorted({idx for idx, _ in fragment_entries})
    if not unique_core_atom_indices:
        return None

    core_idx_to_r_label = {core_idx: i + 1 for i, core_idx in enumerate(unique_core_atom_indices)}

    # Build scaffold with [*:i] labels.
    rw_core = Chem.RWMol(core)
    for core_atom_idx in unique_core_atom_indices:
        dummy_atom = Chem.Atom(0)
        dummy_atom.SetAtomMapNum(core_idx_to_r_label[core_atom_idx])
        dummy_idx = rw_core.AddAtom(dummy_atom)
        rw_core.AddBond(core_atom_idx, dummy_idx, Chem.BondType.SINGLE)

    scaffold_labeled = Chem.MolToSmiles(rw_core.GetMol(), canonical=True)

    # Build decorator sequence, rooting each fragment at dummy to start with [*:i].
    decorator_blocks: List[str] = []
    for core_atom_idx, frag in sorted(fragment_entries, key=lambda x: core_idx_to_r_label[x[0]]):
        label = core_idx_to_r_label[core_atom_idx]
        dummy_idx = None
        for atom in frag.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetIsotope(0)
                atom.SetAtomMapNum(label)
                dummy_idx = atom.GetIdx()
                break
        if dummy_idx is None:
            return None
        frag_smiles = Chem.MolToSmiles(frag, canonical=False, rootedAtAtom=dummy_idx)
        decorator_blocks.append(f"<R{label}> {frag_smiles} </R{label}>")

    decorators_text = " ".join(decorator_blocks)
    return scaffold_labeled, decorators_text


def attach_decorators_to_scaffold(scaffold_smiles: str, decorators_text: str) -> str:
    """
    Assembles final molecule by attaching each decorator [*:i] to scaffold [*:i].
    Raises ValueError on malformed inputs.
    """
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    if scaffold_mol is None:
        raise ValueError("Invalid scaffold SMILES.")

    scaffold_labels = _extract_labels_from_scaffold(scaffold_mol)
    if not scaffold_labels:
        raise ValueError("Scaffold has no labeled attachment points [*:i].")

    decorators = parse_decorator_sequence(decorators_text)
    if not decorators:
        raise ValueError("No valid decorator blocks found in generated text.")

    decorator_labels = sorted(decorators.keys())
    if decorator_labels != scaffold_labels:
        raise ValueError(
            f"Decorator labels {decorator_labels} do not match scaffold labels {scaffold_labels}."
        )

    current = scaffold_mol
    for label in decorator_labels:
        frag_mol = Chem.MolFromSmiles(decorators[label])
        if frag_mol is None:
            raise ValueError(f"Invalid decorator SMILES for R{label}: {decorators[label]}")

        scaffold_dummy_idxs = _find_dummy_with_label(current, label)
        if len(scaffold_dummy_idxs) != 1:
            raise ValueError(
                f"Expected exactly one scaffold dummy with label R{label}, found {len(scaffold_dummy_idxs)}."
            )

        frag_dummy_idxs = _find_dummy_with_label(frag_mol, label)
        if len(frag_dummy_idxs) != 1:
            raise ValueError(
                f"Expected exactly one decorator dummy with label R{label}, found {len(frag_dummy_idxs)}."
            )

        n_base = current.GetNumAtoms()
        combined = Chem.RWMol(Chem.CombineMols(current, frag_mol))
        scaffold_dummy_idx = scaffold_dummy_idxs[0]
        frag_dummy_idx = n_base + frag_dummy_idxs[0]

        scaffold_neighbor = _single_neighbor_idx(combined, scaffold_dummy_idx)
        frag_neighbor = _single_neighbor_idx(combined, frag_dummy_idx)

        combined.AddBond(scaffold_neighbor, frag_neighbor, Chem.BondType.SINGLE)

        # Remove dummies in descending order to keep index stability.
        for idx in sorted([scaffold_dummy_idx, frag_dummy_idx], reverse=True):
            combined.RemoveAtom(idx)

        current = combined.GetMol()
        Chem.SanitizeMol(current)

    return Chem.MolToSmiles(current, canonical=True)
