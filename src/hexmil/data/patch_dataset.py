import os
import pandas as pd


# Multi-class modality labels — use (label > 0) for binary real/fake
MOD_LABEL = {'real': 0, 'pix2pix': 1, 'cycle': 2, 'diffusion': 3, 'ctgan': 4}


def load_split_table(data_dir: str, split: str, mods: list[str] | None = None) -> pd.DataFrame:
    if not os.path.isfile(os.path.join(data_dir, 'data.csv')):
        raise FileNotFoundError(
            f"data.csv not found under DATA_DIR={data_dir!r}. Configure the dataset "
            "path in config.py or via the HEXMIL_DATA_DIR environment variable "
            "(it must point to the M3DSynth root with data.csv and sets.csv)."
        )
    data = pd.read_csv(os.path.join(data_dir, 'data.csv'))
    sets = pd.read_csv(os.path.join(data_dir, 'sets.csv'))

    # Append CT-GAN metadata when requested
    if mods is not None and 'ctgan' in mods:
        ctgan_data_path = os.path.join(data_dir, 'ctgan_data.csv')
        ctgan_sets_path = os.path.join(data_dir, 'ctgan_sets.csv')
        if os.path.exists(ctgan_data_path) and os.path.exists(ctgan_sets_path):
            data = pd.concat([data, pd.read_csv(ctgan_data_path)], ignore_index=True)
            sets = pd.concat([sets, pd.read_csv(ctgan_sets_path)], ignore_index=True)
        else:
            import warnings
            warnings.warn(
                'ctgan requested but ctgan_data.csv / ctgan_sets.csv not found '
                f'in {data_dir}.',
                UserWarning, stacklevel=2,
            )

    tab = data.merge(sets, on='orig_id', how='inner')
    tab = tab[tab['set'] == split].reset_index(drop=True)
    if mods is not None:
        tab = tab[tab['mod'].isin(mods)].reset_index(drop=True)
    return tab
