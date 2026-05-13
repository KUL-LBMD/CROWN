import gemmi

COMMON_ARTIFACTS = {'02U', '12P', '13P', '144', '15P', '16P', '1EM', '1PE', '1PG', '1PS', '2DP', '2JC', '2NV', '2OP', '2PE', '32M', '33O', '3HR', '3PG', 
                    '3SY', '3V3', '543', '6JZ', '6PE', '7E8', '7E9', '7I7', '7N5', '7PE', '7PG', '7PH', '90A', '9FO', '9JE', '9YU', 'AAE', 'ABA', 'AE3', 
                    'AE4', 'AGA', 'AKR', 'AUC', 'B3H', 'B3P', 'B4T', 'B4X', 'BAM', 'BCN', 'BDN', 'BE7', 'BEN', 'BET', 'BEZ', 'BGL', 'BHG', 'BNG', 'BNZ', 
                    'BOG', 'BTB', 'BU1', 'BXC', 'C10', 'C14', 'C8E', 'CAC', 'CAD', 'CAQ', 'CD4', 'CE1', 'CE9', 'CHT', 'CIT', 'CN3', 'CN6', 'CPS', 'CXE', 
                    'CXS', 'D10', 'D12', 'D1D', 'D22', 'DAO', 'DD9', 'DDQ', 'DDR', 'DEP', 'DET', 'DHB', 'DHJ', 'DIO', 'DKA', 'DMS', 'DMF', 'DMI', 'DMR', 'DOX', 
                    'DPG', 'DR6', 'DRE', 'DTD', 'DTT', 'DTU', 'DTV', 'E4N', 'EAP', 'EEE', 'EPE', 'ETE', 'ETF', 'ETX', 'F09', 'F4R', 'FJO', 'FTT', 'FW5', 
                    'GLV', 'GOL', 'GVT', 'GYF', 'HAE', 'HAI', 'HCA', 'HCS', 'HED', 'HEX', 'HEZ', 'HP6', 'HSG', 'HSH', 'HT3', 'HTG', 'HTH', 'HTO', 'HZA', 
                    'I3C', 'ICT', 'IHP', 'IHS', 'IMD', 'IPH', 'JDJ', 'K12', 'KDO', 'L1P', 'L2C', 'L2P', 'L3P', 'L4P', 'LAC', 'LDA', 'LI1', 'LMR', 'LMT', 
                    'LMU', 'LUT', 'M2M', 'MAC', 'MAE', 'MB3', 'MBN', 'MBO', 'MC3', 'ME2', 'MEG', 'MES', 'MLA', 'MLI', 'MLT', 'MPD', 'MPO', 'MRD', 
                    'MYR', 'N8E', 'NBN', 'NET', 'NEX', 'NHE', 'O4B', 'OCT', 'OES', 'OGA', 'OP2', 'OTE', 'OXM', 'P03', 'P15', 'P1O', 'P22', 'P25', 'P2K', 
                    'P33', 'P3G', 'P4C', 'P4G', 'P4K', 'P6G', 'PA8', 'PC8', 'PD7', 'PE3', 'PE4', 'PE5', 'PE6', 'PE7', 'PE8', 'PEG', 'PEP', 'PEU', 'PEX', 
                    'PG0', 'PG4', 'PG5', 'PG6', 'PG8', 'PGE', 'PGF', 'PGO', 'PGR', 'PHB', 'PHQ', 'PL9', 'PLC', 'PMS', 'PPI', 'PQ9', 'PQE', 'PTD', 'PUT', 
                    'PVO', 'PX2', 'PX4', 'QGT', 'QJE', 'QLB', 'RG1', 'RWB', 'SAR', 'SGM', 'SIN', 'SOG', 'SP5', 'SPD', 'SPJ', 'SPM', 'SPZ', 'SQU', 
                    'SRT', 'TAM', 'TAR', 'TAU', 'TBU', 'TCE', 'TCN', 'TEA', 'TFA', 'THE', 'TLA', 'TMA', 'TOE', 'TRD', 'TRS', 'UMQ', 'UND', 'V1J', 
                    'VX', 'XAT', 'XP4', 'XPA', 'XPE', 'Y69'}

def remove_artifacts_and_fix_quotes(structure: gemmi.Structure):
    """
    Read raw mmCIF, split chains so each gemmi.Chain corresponds to a single
    subchain (label_asym_id), and drop subchains whose residues are *entirely*
    crystallization artifacts. Subchains that mix artifact CCDs with standard
    residues (e.g. peptide ligands using ABA/SAR as monomers) are preserved.
    This makes chain.name-based entity lookup 1:1 in later steps.
    """
    structure.setup_entities()   # populates entities + assigns residue.subchain
    for model in structure:
        # Group residues by subchain, preserving first-seen order.
        # Artifact filtering is deferred until after grouping so we can
        # test "entire subchain is artifact" rather than residue-by-residue.
        groups: "dict[str, list[gemmi.Residue]]" = {}
        order: "list[str]" = []
        for chain in model:
            for residue in chain:
                # Fall back to chain name if setup_entities couldn't assign one
                sub_id = residue.subchain or chain.name
                if sub_id not in groups:
                    groups[sub_id] = []
                    order.append(sub_id)
                groups[sub_id].append(residue.clone())

        # Replace all chains in this model with one-subchain-per-chain,
        # skipping any subchain whose residues are all artifacts.
        while len(model) > 0:
            del model[0]
        for sub_id in order:
            residues = groups[sub_id]
            if not residues:
                continue
            if all(r.name in COMMON_ARTIFACTS for r in residues):
                continue
            new_chain = gemmi.Chain(sub_id)
            for res in residues:
                res.subchain = sub_id   # keep consistent post-split
                new_chain.add_residue(res)
            model.add_chain(new_chain)

    # Re-establish entity ↔ chain relationships on the restructured model
    structure.setup_entities()
    return structure
