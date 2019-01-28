from pathlib import Path

base_path = Path('/lustre1/projects/fs_ma8701_1/omniglot_processed')
pngs = list(Path('').glob('**/*.png'))

for png in pngs:
    data = png.read_bytes()
    new_file_path = Path('_'.join(png.parts[:2]), png.parts[-1])
    filepath = base_path.joinpath(new_file_path)
    if not filepath.parent.exists():
        print(filepath.parent)
        filepath.parent.mkdir()
    with filepath.open('wb') as ww:
        ww.write(data)
