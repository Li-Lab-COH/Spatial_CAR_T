from pathlib import Path
import torch

run_name = Path(
    "/coh_labs/yunroseli/Jona/CAR-T/data/zarr/fullDataset/annotating_references/"
    "c2l_run_output/c2l_lvl2_epochs_200_Ncells1_decalpha_100"
)

model_file = run_name / "model.pt"

print("Model file:", model_file, flush=True)
print("Size GB:", model_file.stat().st_size / 1e9, flush=True)

ckpt = torch.load(model_file, map_location="cpu", weights_only=False)

print("\nTop-level type:", type(ckpt), flush=True)

if isinstance(ckpt, dict):
    print("\nTop-level keys:", flush=True)
    for k in ckpt.keys():
        print(" -", k, type(ckpt[k]), flush=True)

    print("\nPotential training/model-state fields:", flush=True)
    for key in [
        "is_trained_",
        "is_trained",
        "history_",
        "attr_dict",
        "init_params_",
        "registry_",
        "model_state_dict",
        "var_names",
    ]:
        if key in ckpt:
            value = ckpt[key]
            print(f"\nFOUND {key}: {type(value)}", flush=True)
            if isinstance(value, (str, int, float, bool, type(None))):
                print("  value:", value, flush=True)
            elif isinstance(value, dict):
                print("  dict keys:", list(value.keys())[:30], flush=True)
            else:
                try:
                    print("  len:", len(value), flush=True)
                except Exception:
                    pass


def find_keys(obj, targets, path="root", max_depth=8):
    if max_depth < 0:
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}"
            if str(k) in targets:
                print(f"\nFOUND TARGET {k} at {new_path}", flush=True)
                print("type:", type(v), flush=True)
                try:
                    if isinstance(v, (str, int, float, bool, type(None))):
                        print("value:", v, flush=True)
                    elif isinstance(v, dict):
                        print("dict keys:", list(v.keys())[:30], flush=True)
                    else:
                        print("len:", len(v), flush=True)
                except Exception as e:
                    print("could not summarize:", repr(e), flush=True)
            find_keys(v, targets, new_path, max_depth - 1)

    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:50]):
            find_keys(v, targets, f"{path}[{i}]", max_depth - 1)


print("\nRecursive search:", flush=True)
find_keys(
    ckpt,
    targets={
        "is_trained_",
        "is_trained",
        "history_",
        "init_params_",
        "model_kwargs",
        "max_epochs",
        "train_size",
        "batch_size",
        "registry_",
        "attr_dict",
        "adata_attrs",
        "summary_stats",
    },
)

attr = ckpt["attr_dict"]

print("is_trained_:", attr.get("is_trained_", "MISSING"))
print("run_name_:", attr.get("run_name_", "MISSING"))
print("run_id_:", attr.get("run_id_", "MISSING"))

print("\nSaved history type:", type(attr.get("history_", None)))
if "history_" in attr:
    try:
        print(attr["history_"])
    except Exception as e:
        print("Could not print history:", repr(e))

print("\ninit_params_ type:", type(attr.get("init_params_", None)))
if "init_params_" in attr:
    print(attr["init_params_"])

print("\nregistry scvi version if present:")
reg = attr.get("registry_", None)
try:
    print(reg.get("scvi_version", "no scvi_version key"))
except Exception as e:
    print("Could not read registry scvi_version:", repr(e))
