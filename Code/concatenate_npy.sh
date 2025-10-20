python - <<'PY'
import numpy as np, os, pathlib, tempfile

a = np.load('Gen_VAE_2D_v2_develop_cond_class2_vae_cond.npy', mmap_mode='r')
b = np.load('Gen_VAE_2D_v2_develop_cond_class4_vae_cond.npy', mmap_mode='r')
c = np.concatenate([a, b], axis=0)   # (50, 400, 2048)

out = pathlib.Path('Gen_VAE_2D_v2_develop_cond_class2_4_vae_cond.npy').resolve()

# tmp con sufijo .npy para que np.save NO agregue otro .npy
with tempfile.NamedTemporaryFile(dir=out.parent, prefix=out.stem + '.', suffix='.npy', delete=False) as f:
    tmp_path = pathlib.Path(f.name)

np.save(tmp_path, c)
os.replace(tmp_path, out)

print("OK ->", out, "shape:", c.shape, "dtype:", c.dtype, "size:", out.stat().st_size)
PY
