from pathlib import Path
from random import shuffle

base_path = Path('/lustre1/projects/fs_ma8701_1/omniglot_processed')
folders = list(Path('').glob('*/**'))
valid_pct = 0.15


def read_and_write(img, folder, prefix):
    data = img.read_bytes()
    new_file_path = Path(base_path, prefix, '_'.join(folder.parts), img.name)
    if not new_file_path.parent.exists():
        new_file_path.parent.mkdir(parents=True)
    with new_file_path.open('wb') as ww:
        ww.write(data)


for folder in folders:
    pngs = list(folder.glob('*.png'))
    shuffle(pngs)
    n_images = len(pngs)
    n_valid = int(n_images * valid_pct)

    train_imgs = pngs[n_valid:]
    valid_imgs = pngs[:n_valid]

    for png in train_imgs:
        read_and_write(png, folder, 'train')
    for png in valid_imgs:
        read_and_write(png, folder, 'valid')
